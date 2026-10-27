from __future__ import annotations

from collections import deque
from dataclasses import replace

from tarqa.data.metadata import ForeignKey, SchemaCatalog


def _adjacency(catalog: SchemaCatalog) -> dict[str, list[tuple[str, ForeignKey]]]:
    graph: dict[str, list[tuple[str, ForeignKey]]] = {name: [] for name in catalog.tables}
    for fk in catalog.foreign_keys:
        graph.setdefault(fk.child_table, []).append((fk.parent_table, fk))
        graph.setdefault(fk.parent_table, []).append((fk.child_table, fk))
    return graph


def shortest_table_path(catalog: SchemaCatalog, source: str, target: str) -> list[str] | None:
    if source == target:
        return [source]
    graph = _adjacency(catalog)
    queue = deque([[source]])
    seen = {source}
    while queue:
        path = queue.popleft()
        for nxt, _ in graph.get(path[-1], []):
            if nxt in seen:
                continue
            new_path = [*path, nxt]
            if nxt == target:
                return new_path
            seen.add(nxt)
            queue.append(new_path)
    return None


def expand_relevant_tables(catalog: SchemaCatalog, relevant_tables: list[str]) -> list[str]:
    selected = list(dict.fromkeys(relevant_tables))
    if len(selected) <= 1:
        return selected
    expanded = set(selected)
    anchor = selected[0]
    for table in selected[1:]:
        path = shortest_table_path(catalog, anchor, table)
        if path:
            expanded.update(path)
    return sorted(expanded)


def pruned_catalog(
    catalog: SchemaCatalog, relevant_tables: list[str], add_bridge_tables: bool = True
) -> SchemaCatalog:
    if not relevant_tables:
        raise ValueError("Task Spec has no relevant tables; refusing full-schema fallback")
    unknown = [name for name in relevant_tables if name not in catalog.tables]
    if unknown:
        raise ValueError(f"Task Spec references missing tables: {unknown}")
    names = (
        expand_relevant_tables(catalog, relevant_tables)
        if add_bridge_tables
        else list(dict.fromkeys(relevant_tables))
    )
    missing = [name for name in names if name not in catalog.tables]
    if missing:
        raise ValueError(f"Task Spec references missing tables: {missing}")
    return SchemaCatalog(
        domain=catalog.domain,
        description=catalog.description,
        tables={name: replace(catalog.tables[name],
            columns=dict(catalog.tables[name].columns),
            primary_keys=list(catalog.tables[name].primary_keys),
            foreign_keys=[fk for fk in catalog.tables[name].foreign_keys
                          if fk.child_table in names and fk.parent_table in names])
                for name in names},
    )
