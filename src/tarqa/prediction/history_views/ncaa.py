"""Historical game views at daily resolution. Never exposes target labels."""
from pathlib import Path
import duckdb
from .common import quote,stamp
GAME_TABLES = ['regular_season_compact_results','regular_season_detailed_results','tourney_compact_results','tourney_detailed_results']

def open_history(database,reference_time,release_root=None):
 con=duckdb.connect(':memory:');con.execute('SET threads=2');con.execute(f'ATTACH {quote(Path(database).resolve())} AS snapshot (READ_ONLY)')
 cut='TIMESTAMP '+quote(stamp(reference_time));src='snapshot.main.'
 con.execute(f'CREATE TEMP VIEW seasons AS SELECT season,dayzero FROM {src}seasons WHERE dayzero <= {cut}')
 for table in GAME_TABLES:
  con.execute(f'''CREATE TEMP VIEW {table} AS SELECT g.* FROM {src}{table} g
  JOIN seasons s ON s.season=g.season WHERE s.dayzero+(g.daynum+1)*INTERVAL '1 day' <= {cut}''')
 con.execute(f'''CREATE TEMP VIEW teams AS SELECT t.team_id FROM {src}teams t
 WHERE EXISTS(SELECT 1 FROM regular_season_compact_results g WHERE g.wteam=t.team_id OR g.lteam=t.team_id)
 OR EXISTS(SELECT 1 FROM tourney_compact_results g WHERE g.wteam=t.team_id OR g.lteam=t.team_id)''')
 for table in ('tourney_seeds','tourney_slots'):
  con.execute(f'''CREATE TEMP VIEW {table} AS SELECT t.* FROM {src}{table} t
  JOIN seasons s ON t.season=s.season WHERE s.dayzero+INTERVAL '155 days' <= {cut}''')
 con.execute(f'CREATE TEMP VIEW target AS SELECT * FROM {src}target WHERE FALSE')
 return con
