"""Parameter-free, role-aware aggregation over audited historical views only."""
from collections import OrderedDict
from dataclasses import dataclass
import math

from tarqa.data.metadata import SchemaCatalog, ForeignKey
from tarqa.prediction.fk_graph import SchemaGraph
from tarqa.prediction.graph import quote_identifier as qi
from tarqa.prediction.history_views import open_history
from tarqa.types import EntityAnchor, Track
from tarqa.data.releases import stamp


@dataclass
class HistoricalFeatures:
    features: dict
    audit: dict


def step_condition(step, left, right):
    fk = step.foreign_key
    if step.from_table == fk.parent_table and step.to_table == fk.child_table:
        return f'{left}.{qi(fk.parent_column)} = {right}.{qi(fk.child_column)}'
    return f'{left}.{qi(fk.child_column)} = {right}.{qi(fk.parent_column)}'


def semijoin_query(path, anchor):
    """Count final physical rows once, including tables without primary keys.

    A normal multi-hop JOIN would multiply an order reached through two items.
    EXISTS retains row multiplicity in the final table, not in the traversal.
    """
    if not path.steps:
        return f'SELECT t.* FROM {qi(anchor.table)} t WHERE t.{qi(anchor.key)} = ?'
    joins = [f'FROM {qi(anchor.table)} a0']
    for i, step in enumerate(path.steps[:-1], 1):
        joins.append(f'JOIN {qi(step.to_table)} a{i} ON {step_condition(step, f"a{i-1}", f"a{i}")}')
    last = step_condition(path.steps[-1], f'a{len(path.steps)-1}', 't')
    return (f'SELECT t.* FROM {qi(path.end_table)} t WHERE EXISTS '
            f'(SELECT 1 {" ".join(joins)} WHERE a0.{qi(anchor.key)} = ? AND {last})')


class ReleaseAggregator:
    def __init__(self, paths, config):
        self.paths, self.config = paths, config
        self.catalog = SchemaCatalog.from_files(paths.domain, paths.metadata)
        self._contexts = OrderedDict()
        self._feature_cache = OrderedDict()
        self.max_hops = int(config.get("max_hops", 3))
        self.max_paths = int(config.get("max_paths_per_table", 4))
        self.table_policy = config.get("relation_table_policy", "relevant_only")
        if self.table_policy not in {"relevant_only", "full_schema"}:
            raise ValueError("Unsupported table policy")

    def close(self):
        for con, _, _ in self._contexts.values():
            con.close()
        self._contexts.clear()

    def _context(self, reference_time):
        if reference_time in self._contexts:
            self._contexts.move_to_end(reference_time)
            return self._contexts[reference_time]
        con = open_history(self.paths, reference_time)
        names = set(self.catalog.table_names)
        if self.paths.domain == "dallas":
            names.add("officer_entities")
        columns = {table: {r[0]: r[1] for r in con.execute(f'DESCRIBE SELECT * FROM {qi(table)}').fetchall()}
                   for table in sorted(names)}
        fks = [fk for fk in self.catalog.foreign_keys
               if fk.child_column in columns.get(fk.child_table, {})
               and fk.parent_column in columns.get(fk.parent_table, {})]
        if self.paths.domain == "dallas":
            fks.append(ForeignKey("officers", "full_name", "officer_entities", "full_name"))
        if self.paths.domain == "airline":
            fact = "on_time_on_time_performance_2016_1"
            fks.extend([ForeignKey(fact, "carrier", "l_unique_carriers", "code"),
                        ForeignKey(fact, "arrtimeblk", "l_deparrblk", "code")])
        # FK structure comes from declared schema, never future-value containment.
        graph = SchemaGraph(self.catalog, sorted(fks, key=lambda f: f.signature))
        self._contexts[reference_time] = (con, columns, graph)
        while len(self._contexts) > int(self.config.get("history_connections", 2)):
            self._contexts.popitem(last=False)[1][0].close()
        return con, columns, graph

    def aggregate(self, spec, binding, *, sample_key=""):
        if spec.track != Track.PREDICTION or spec.validate_dependencies():
            raise ValueError("Historical aggregation needs a valid predictive spec")
        spec = spec.model_copy(update={"reference_time": stamp(spec.reference_time)})
        anchor = binding.anchors[0]
        if self.paths.domain == "dallas" and (anchor.table, anchor.key) == ("officers", "full_name"):
            anchor = EntityAnchor(table="officer_entities", key="full_name", value=anchor.value)
        unknown = set(spec.relevant_tables) - set(self.catalog.table_names)
        if unknown:
            raise ValueError(f"Unknown relevant tables: {unknown}")
        key = (anchor.table, anchor.key, str(anchor.value), spec.reference_time,
               tuple(sorted(spec.relevant_tables)), self.table_policy)
        if key in self._feature_cache:
            return self._feature_cache[key]
        con, columns, graph = self._context(spec.reference_time)
        if anchor.key not in columns.get(anchor.table, {}):
            raise ValueError("Unknown or hidden anchor key")
        n = con.execute(f'SELECT COUNT(*) FROM {qi(anchor.table)} WHERE {qi(anchor.key)}=?', [anchor.value]).fetchone()[0]
        if n != 1:
            raise ValueError(f"Expected exactly one historically visible anchor; got {n}: {anchor}")
        targets = set(spec.relevant_tables) if self.table_policy == "relevant_only" else set(columns)
        targets.add(anchor.table)
        features, used, unreachable = {}, [], []
        key_columns = self.catalog.entity_key_columns()
        key_columns.add((anchor.table, anchor.key))
        for fk in graph.foreign_keys:
            key_columns.update(((fk.child_table, fk.child_column), (fk.parent_table, fk.parent_column)))
        if self.paths.domain == "airline":
            fact = "on_time_on_time_performance_2016_1"
            for column in ("cancelled", "arrdel15", "depdel15", "diverted", "divairportlandings"):
                key_columns.discard((fact, column))
        for table in sorted(targets):
            paths = graph.paths(anchor.table, table, max_hops=self.max_hops, max_paths=self.max_paths)
            if not paths:
                unreachable.append(table)
            for path in paths:
                prefix = f"relation {path.signature}"
                query = semijoin_query(path, anchor)
                expressions, suffixes = ["COUNT(*)"], ["row_count"]
                for column, dtype in columns[table].items():
                    if (table, column) in key_columns:
                        continue
                    ref = qi(column)
                    numeric = any(t in dtype for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "BOOL"))
                    temporal = any(t in dtype for t in ("DATE", "TIME"))
                    if numeric:
                        functions = {"count": f"COUNT({ref})", "sum": f"SUM({ref}::DOUBLE)",
                            "mean": f"AVG({ref}::DOUBLE)", "min": f"MIN({ref}::DOUBLE)",
                            "max": f"MAX({ref}::DOUBLE)", "std": f"STDDEV_POP({ref}::DOUBLE)"}
                        for stat, expr in functions.items():
                            expressions.append(expr); suffixes.append(f"{column}::{stat}")
                    elif temporal:
                        # Future estimates may be visible; preserve negative recency instead of inventing clipping.
                        for stat, op in (("recency_days", "MAX"), ("oldest_days", "MIN")):
                            expressions.append(f"epoch(TIMESTAMP '{spec.reference_time}' - {op}({ref})::TIMESTAMP)/86400.0")
                            suffixes.append(f"{column}::{stat}")
                    elif self.config.get("include_categorical", True):
                        expressions.extend([f"COUNT({ref})", f"COUNT(DISTINCT {ref})"])
                        suffixes.extend([f"{column}::count", f"{column}::distinct_count"])
                values = con.execute(f'SELECT {", ".join(expressions)} FROM ({query}) visible_rows', [anchor.value]).fetchone()
                for suffix, value in zip(suffixes, values):
                    full = f"{prefix}::{suffix}"
                    features[full] = float(value) if value is not None else 0.0
                    if value is None:
                        features[full + "::missing"] = 1.0
                    if not math.isfinite(features[full]):
                        raise ValueError(f"Nonfinite relational feature: {full}")
                if self.config.get("include_categorical", True):
                    for column, dtype in columns[table].items():
                        if (table, column) in key_columns or not any(t in dtype for t in ("VARCHAR", "ENUM")):
                            continue
                        ref = qi(column)
                        hist = con.execute(f"SELECT {ref}, COUNT(*) n FROM ({query}) visible_rows "
                            f"WHERE {ref} IS NOT NULL AND length(CAST({ref} AS VARCHAR)) <= 80 "
                            f"GROUP BY {ref} ORDER BY n DESC, CAST({ref} AS VARCHAR) LIMIT ?",
                            [anchor.value, int(self.config.get("categorical_top_k", 4))]).fetchall()
                        for category, count in hist:
                            descriptor = str(category).replace("::", " ")
                            features[f"{prefix}::{column} value {descriptor}::category_count"] = float(count)
                features[prefix + "::active"] = 1.0
                used.append(prefix)
        for table in unreachable:
            features[f"unreachable {table}::missing"] = 1.0
        result = HistoricalFeatures(features, dict(reference_time=spec.reference_time,
            runtime_anchor=anchor.model_dump(mode="json"), used_slots=used,
            unreachable_tables=unreachable, visibility_policy=self.paths.domain))
        self._feature_cache[key] = result
        if len(self._feature_cache) > 1024:
            self._feature_cache.popitem(last=False)
        return result
