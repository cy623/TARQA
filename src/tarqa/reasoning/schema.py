"""Render the retained database schema for SQL generation."""

def qi(name):
    return '"' + name.replace('"', '""') + '"'


def schema_ddl(catalog):
    statements = []
    for name in catalog.table_names:
        table = catalog.tables[name]
        columns = [f"  {qi(c.name)} {c.dtype}" for c in table.columns.values()]
        statements.append(f"CREATE TABLE {qi(name)} (\n" + ",\n".join(columns) + "\n);")
    statements.extend(f"-- Foreign key: {fk.signature}" for fk in catalog.foreign_keys)
    return "\n\n".join(statements)
