"""Record-level typed graphs built ONLY from the audited historical views.

SQL enumerates rows and PK/FK edges; it does not compute graph embeddings.
Physical duplicate rows remain separate nodes, including tables without a PK.
The two directed versions of a FK have different relation identities.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import math

import numpy as np

from tarqa.data.releases import stamp
from tarqa.prediction.graph import quote_identifier as qi
from tarqa.prediction.release_aggregator import ReleaseAggregator
from tarqa.types import EntityAnchor, Track


@dataclass
class RowGraph:
    node_texts: list[str]
    node_tables: list[str]
    edges: dict[str, np.ndarray]  # [2, E]: source (sender), destination (receiver)
    anchor_index: int


@dataclass
class HistoricalGraph:
    features: RowGraph
    audit: dict


@dataclass(frozen=True)
class DirectedRelation:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    description: str


def directed_relations(foreign_keys):
    relations = []
    for fk in sorted(set(foreign_keys), key=lambda f: f.signature):
        for reverse in (False, True):
            source = (fk.parent_table, fk.parent_column) if reverse else (fk.child_table, fk.child_column)
            target = (fk.child_table, fk.child_column) if reverse else (fk.parent_table, fk.parent_column)
            direction = 'parent_to_child' if reverse else 'child_to_parent'
            text = (f'Foreign key {fk.signature}; direction {direction}; '
                    f'message from {source[0]}.{source[1]} to {target[0]}.{target[1]}')
            relations.append(DirectedRelation(*source, *target, text))
    return relations


def _value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite historical node attribute')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def render_node(table, columns, values, excluded, reference_time):
    """Column identity and value co-occur in a row; no label, QA, or raw ID input."""
    fields = []
    cutoff = datetime.fromisoformat(reference_time)
    for column, value in sorted(zip(columns, values)):
        if column in excluded:
            continue
        field = dict(column=column, dtype=columns[column], value=_value(value))
        if isinstance(value, (date, datetime)):
            observed = value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())
            # An already-visible estimated future date legitimately has negative age.
            field['age_days'] = (cutoff - observed).total_seconds() / 86400.0
        fields.append(field)
    return json.dumps(dict(table=table, fields=fields), ensure_ascii=False, separators=(',', ':'), allow_nan=False)


class RowGraphAggregator(ReleaseAggregator):
    """Reuse the audited history contexts; replace statistics with real row graphs.

Each needed visible table is indexed once per cached cutoff. Adjacency is
queried on demand, without constructing a whole-database edge tensor.
Resource limits fail explicitly; they never silently sample/truncate a graph.
"""
    ROW_ID = '__tarqa_graph_row_id__'

    def __init__(self, paths, config):
        super().__init__(paths, config)
        self.num_layers = int(config.get('num_layers', 2))
        if self.num_layers != 2:
            raise ValueError('This graph protocol is fixed to exactly 2 message-passing layers')
        if 'max_hops' in config and int(config['max_hops']) != self.num_layers:
            raise ValueError('For the graph backend max_hops must equal num_layers=2')
        self.max_nodes = int(config.get('max_nodes', 50000))
        self.max_edges = int(config.get('max_edges', 500000))
        if min(self.max_nodes, self.max_edges, int(config.get('history_connections', 2))) < 1:
            raise ValueError('Graph resource limits and history_connections must be positive')

    def _table(self, con, table, columns):
        if self.ROW_ID in columns[table]:
            raise ValueError('Reserved graph row identifier collides with a database column')
        name = '__tarqa_gnn_' + hashlib.sha256(table.encode()).hexdigest()[:20]
        # These SELECTs deliberately use the historical view, never the raw attachment.
        con.execute(f'CREATE TEMP TABLE IF NOT EXISTS {qi(name)} AS '
                    f'SELECT row_number() OVER () AS {qi(self.ROW_ID)}, * FROM {qi(table)}')
        return qi(name)

    def _allowed(self, anchor, spec, columns, relations):
        if self.table_policy == 'full_schema':
            return set(columns)
        targets = set(spec.relevant_tables) | {anchor.table}
        allowed = set(targets)
        # Retain bridge tables on a two-hop path from anchor to a requested table.
        # No max_paths cap: different FK roles must all be retained.
        for left in relations:
            if left.source_table != anchor.table:
                continue
            if any(right.source_table == left.target_table and right.target_table in targets - {anchor.table}
                   for right in relations):
                allowed.add(left.target_table)
        return allowed

    def aggregate(self, spec, binding, *, sample_key=''):
        if spec.track != Track.PREDICTION or spec.validate_dependencies():
            raise ValueError('Historical graph construction needs a valid predictive spec')
        if len(binding.anchors) != 1:
            raise ValueError('Historical graph construction requires exactly one anchor')
        reference_time = stamp(spec.reference_time)
        anchor = binding.anchors[0]
        if self.paths.domain == 'dallas' and (anchor.table, anchor.key) == ('officers', 'full_name'):
            anchor = EntityAnchor(table='officer_entities', key='full_name', value=anchor.value)
        unknown = set(spec.relevant_tables) - set(self.catalog.table_names)
        if unknown:
            raise ValueError(f'Unknown relevant tables: {unknown}')
        con, columns, schema_graph = self._context(reference_time)
        if anchor.key not in columns.get(anchor.table, {}):
            raise ValueError('Unknown or historically hidden anchor key')
        root_table = self._table(con, anchor.table, columns)
        roots = con.execute(f'SELECT {qi(self.ROW_ID)} FROM {root_table} WHERE {qi(anchor.key)}=?',
                            [anchor.value]).fetchmany(2)
        if len(roots) != 1:
            raise ValueError(f'Expected exactly one historically visible anchor: {anchor}')
        root_id = int(roots[0][0])
        visible_fks = [fk for fk in schema_graph.foreign_keys
                       if fk.child_column in columns.get(fk.child_table, {})
                       and fk.parent_column in columns.get(fk.parent_table, {})]
        all_relations = directed_relations(visible_fks)
        allowed = self._allowed(anchor, spec, columns, all_relations)
        relations = [r for r in all_relations if r.source_table in allowed and r.target_table in allowed]
        outgoing = defaultdict(list)
        for relation in relations:
            outgoing[relation.source_table].append(relation)
        visited = {anchor.table: {root_id}}
        frontier = {anchor.table: {root_id}}
        nodes_per_hop = [1]
        for _ in range(self.num_layers):
            following = defaultdict(set)
            for table, ids in sorted(frontier.items()):
                src = self._table(con, table, columns)
                for relation in outgoing[table]:
                    dst = self._table(con, relation.target_table, columns)
                    found = con.execute(
                        f'SELECT DISTINCT t.{qi(self.ROW_ID)} FROM {src} s JOIN {dst} t '
                        f'ON s.{qi(relation.source_column)}=t.{qi(relation.target_column)} '
                        f'WHERE s.{qi(self.ROW_ID)} IN (SELECT unnest(?)) LIMIT ?',
                        [sorted(ids), self.max_nodes + 1]).fetchall()
                    if len(found) > self.max_nodes:
                        raise ValueError(f'Graph exceeds max_nodes={self.max_nodes}: {sample_key}; no graph was sampled')
                    seen = visited.setdefault(relation.target_table, set())
                    new = {int(row[0]) for row in found} - seen
                    following[relation.target_table].update(new)
                    seen.update(new)
                    if sum(map(len, visited.values())) > self.max_nodes:
                        raise ValueError(f'Graph exceeds max_nodes={self.max_nodes}: {sample_key}; no graph was sampled')
            nodes_per_hop.append(sum(map(len, following.values())))
            frontier = {table: ids for table, ids in following.items() if ids}

        # IDs are structural only. Duplicate physical rows get distinct identifiers.
        node_keys = [(table, rid) for table in sorted(visited) for rid in sorted(visited[table])]
        index = {key: i for i, key in enumerate(node_keys)}
        edges = {}
        total_edges = 0
        for relation in relations:
            left, right = visited.get(relation.source_table), visited.get(relation.target_table)
            if not left or not right:
                continue
            src = self._table(con, relation.source_table, columns)
            dst = self._table(con, relation.target_table, columns)
            pairs = con.execute(
                f'SELECT s.{qi(self.ROW_ID)},t.{qi(self.ROW_ID)} FROM {src} s JOIN {dst} t '
                f'ON s.{qi(relation.source_column)}=t.{qi(relation.target_column)} '
                f'WHERE s.{qi(self.ROW_ID)} IN (SELECT unnest(?)) '
                f'AND t.{qi(self.ROW_ID)} IN (SELECT unnest(?)) '
                f'ORDER BY s.{qi(self.ROW_ID)},t.{qi(self.ROW_ID)} LIMIT ?',
                [sorted(left), sorted(right), self.max_edges - total_edges + 1]).fetchall()
            total_edges += len(pairs)
            if total_edges > self.max_edges:
                raise ValueError(f'Graph exceeds max_edges={self.max_edges}: {sample_key}; no graph was sampled')
            if pairs:
                edges[relation.description] = np.asarray([
                    [index[(relation.source_table, int(a))] for a, _ in pairs],
                    [index[(relation.target_table, int(b))] for _, b in pairs]], dtype=np.int64)

        key_columns = self.catalog.entity_key_columns() | {(anchor.table, anchor.key)}
        for fk in visible_fks:
            key_columns.update(((fk.child_table, fk.child_column), (fk.parent_table, fk.parent_column)))
        if self.paths.domain == 'airline':
            for column in ('cancelled', 'arrdel15', 'depdel15', 'diverted', 'divairportlandings'):
                key_columns.discard(('on_time_on_time_performance_2016_1', column))
        texts, tables = [], []
        for table in sorted(visited):
            if not visited[table]:
                continue
            actual = list(columns[table])
            selected = ','.join(qi(c) for c in actual)
            rows = con.execute(f'SELECT {selected} FROM {self._table(con, table, columns)} '
                               f'WHERE {qi(self.ROW_ID)} IN (SELECT unnest(?)) ORDER BY {qi(self.ROW_ID)}',
                               [sorted(visited[table])]).fetchall()
            excluded = {c for t, c in key_columns if t == table}
            texts.extend(render_node(table, columns[table], row, excluded, reference_time) for row in rows)
            tables.extend([table] * len(rows))
        graph = RowGraph(texts, tables, edges, index[(anchor.table, root_id)])
        audit = dict(backend='hetero_graph', graph_protocol_version=1, reference_time=reference_time,
            runtime_anchor=anchor.model_dump(mode='json'), num_layers=self.num_layers,
            node_count=len(texts), directed_edge_count=total_edges, nodes_per_hop=nodes_per_hop,
            node_counts_by_table=dict(Counter(tables)), edge_counts_by_relation={k: v.shape[1] for k, v in edges.items()},
            requested_tables=sorted(spec.relevant_tables), allowed_tables=sorted(allowed),
            requested_tables_without_nodes=sorted(set(spec.relevant_tables) - set(tables)),
            bridge_tables=sorted(allowed - set(spec.relevant_tables) - {anchor.table}),
            visibility_policy=self.paths.domain, row_serialization_version=1, graph_sampling='none')
        return HistoricalGraph(graph, audit)
