"""Historical flight and lookup views under the declared operational replay lag."""
from pathlib import Path
import json
import duckdb
from .common import quote,stamp
FACT = 'on_time_on_time_performance_2016_1'


def open_history(database,reference_time,release_root=None):
    cfg=json.loads((Path(release_root)/'config.json').read_text());schema=json.loads((Path(release_root)/'source/metadata_Airline.json').read_text())
    con=duckdb.connect(':memory:');con.execute('SET threads=2');con.execute(f'ATTACH {quote(Path(database).resolve())} AS snapshot (READ_ONLY)')
    con.execute(f'''CREATE TEMP VIEW {FACT} AS SELECT * FROM snapshot.main.{FACT}
        WHERE CAST(flightdate AS TIMESTAMP)+INTERVAL '{cfg['event_availability_lag_days']} days'<=TIMESTAMP {quote(stamp(reference_time))}''')
    # Lookup vocabularies are reached only through already-visible flight rows.
    # Keep code identity, hiding unversioned descriptive attributes in features.
    references={table:[] for table in schema['tables'] if table!=FACT}
    for fk in schema['tables'][FACT]['foreign_keys']:
        target,key=fk['references'].split('.');references[target].append(fk['column'])
    # Schema omits arrtimeblk even though it shares the same static block codes.
    references['l_deparrblk'].append('arrtimeblk')
    for table,columns in references.items():
        assert columns
        condition=' OR '.join(f'f.{col}=d.code' for col in columns)
        con.execute(f'''CREATE TEMP VIEW {table} AS SELECT d.code FROM snapshot.main.{table} d
            WHERE EXISTS(SELECT 1 FROM {FACT} f WHERE {condition})''')
    return con
