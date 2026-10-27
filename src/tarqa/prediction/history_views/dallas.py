"""DATE-safe incident views; undated participation rows inherit incident time."""
from pathlib import Path
import duckdb
from .common import quote,stamp


def open_history(database,reference_time,release_root=None):
    con=duckdb.connect(':memory:');con.execute('SET threads=2')
    con.execute(f'ATTACH {quote(Path(database).resolve())} AS snapshot (READ_ONLY)')
    cutoff='TIMESTAMP '+quote(stamp(reference_time))
    con.execute(f'''CREATE TEMP VIEW incidents AS
        SELECT case_number,date,location,subject_weapon,subject_count,officer_count,latitude,longitude
        FROM snapshot.main.incidents WHERE CAST(date AS TIMESTAMP)+INTERVAL '1 day'<={cutoff}''')
    for table in ('officers','subjects'):
        con.execute(f'''CREATE TEMP VIEW {table} AS SELECT p.* FROM snapshot.main.{table} p
            WHERE EXISTS(SELECT 1 FROM incidents i WHERE i.case_number=p.case_number)''')
    # Original officers.full_name identifies a name group across participation
    # rows. This derived view provides one row per *visible* original name.
    con.execute('CREATE TEMP VIEW officer_entities AS SELECT DISTINCT full_name FROM officers')
    return con
