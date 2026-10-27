"""Conservative historical inputs: structural identifiers and timestamped events.

Unversioned profile text, post text/tags, final scores/counters/statuses are NOT
exposed. Snapshot ownership is retained as an explicit relational assumption.
"""
from pathlib import Path
import duckdb
from .common import quote,stamp

def open_history(database,reference_time,release_root=None):
 con=duckdb.connect(':memory:');con.execute('SET threads=2')
 con.execute(f'ATTACH {quote(Path(database).resolve())} AS snapshot (READ_ONLY)')
 cut='TIMESTAMP '+quote(stamp(reference_time));src='snapshot.main.'
 con.execute(f'CREATE TEMP VIEW users AS SELECT id,creationdate FROM {src}users WHERE creationdate < {cut}')
 con.execute(f'''CREATE TEMP VIEW posts AS
 SELECT p.id,p.posttypeid,p.creaiondate,
 CASE WHEN EXISTS(SELECT 1 FROM users u WHERE u.id=p.owneruserid) THEN p.owneruserid END AS owneruserid,
 CASE WHEN EXISTS(SELECT 1 FROM {src}posts parent WHERE parent.id=p.parentid AND parent.creaiondate < {cut}) THEN p.parentid END AS parentid
 FROM {src}posts p WHERE p.creaiondate < {cut}''')
 con.execute(f'''CREATE TEMP VIEW comments AS
 SELECT c.id,c.postid,c.creationdate,
 CASE WHEN EXISTS(SELECT 1 FROM users u WHERE u.id=c.userid) THEN c.userid END AS userid
 FROM {src}comments c WHERE c.creationdate < {cut}
 AND EXISTS(SELECT 1 FROM posts p WHERE p.id=c.postid)''')
 con.execute(f'''CREATE TEMP VIEW votes AS
 SELECT v.id,v.postid,v.votetypeid,v.creationdate,v.bountyamount,
 CASE WHEN EXISTS(SELECT 1 FROM users u WHERE u.id=v.userid) THEN v.userid END AS userid
 FROM {src}votes v WHERE v.creationdate < CAST({cut} AS DATE)
 AND EXISTS(SELECT 1 FROM posts p WHERE p.id=v.postid)''')
 con.execute(f'''CREATE TEMP VIEW badges AS SELECT b.id,b.userid,b.name,b.date
 FROM {src}badges b WHERE b.date < {cut} AND EXISTS(SELECT 1 FROM users u WHERE u.id=b.userid)''')
 con.execute(f'''CREATE TEMP VIEW posthistory AS
 SELECT h.id,h.posthistorytypeid,h.postid,h.creationdate,
 CASE WHEN EXISTS(SELECT 1 FROM users u WHERE u.id=h.userid) THEN h.userid END AS userid
 FROM {src}posthistory h WHERE h.creationdate < {cut}
 AND EXISTS(SELECT 1 FROM posts p WHERE p.id=h.postid)''')
 con.execute(f'''CREATE TEMP VIEW postlinks AS SELECT l.id,l.creationdate,l.postid,l.relatedpostid,l.linktypeid
 FROM {src}postlinks l WHERE l.creationdate < {cut}
 AND EXISTS(SELECT 1 FROM posts p WHERE p.id=l.postid)
 AND EXISTS(SELECT 1 FROM posts p WHERE p.id=l.relatedpostid)''')
 # Schema-compatible empty relation; final tag counts and wiki links have no history contract.
 con.execute(f'CREATE TEMP VIEW tags AS SELECT * FROM {src}tags WHERE FALSE')
 return con
