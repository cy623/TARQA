"""Conservative DATE-level historical views with witnessed undated links."""
from pathlib import Path
import duckdb
from .common import quote,stamp


def open_history(database,reference_time,release_root=None):
    con=duckdb.connect(':memory:');con.execute('SET threads=2')
    con.execute(f'ATTACH {quote(Path(database).resolve())} AS snapshot (READ_ONLY)')
    src='snapshot.main.';cut='TIMESTAMP '+quote(stamp(reference_time))
    def known(column):return f"CAST({column} AS TIMESTAMP)+INTERVAL '1 day'<={cut}"
    con.execute(f'CREATE TEMP VIEW account AS SELECT account_id,district_id,date FROM {src}account WHERE {known("date")}')
    con.execute(f'''CREATE TEMP VIEW trans AS SELECT t.* FROM {src}trans t
        WHERE {known('t.date')} AND EXISTS(SELECT 1 FROM account a WHERE a.account_id=t.account_id)''')
    con.execute(f'''CREATE TEMP VIEW loan AS SELECT l.loan_id,l.account_id,l.date,l.amount,l.duration,l.payments
        FROM {src}loan l WHERE {known('l.date')} AND EXISTS(SELECT 1 FROM account a WHERE a.account_id=l.account_id)''')
    # An old account alone does not date an undated disposition. A past card
    # references its exact disp_id and provides a dated witness of this link.
    con.execute(f'''CREATE TEMP VIEW disp AS SELECT d.disp_id,d.client_id,d.account_id,d.type FROM {src}disp d
        WHERE EXISTS(SELECT 1 FROM account a WHERE a.account_id=d.account_id)
        AND EXISTS(SELECT 1 FROM {src}card c WHERE c.disp_id=d.disp_id AND {known('c.issued')})''')
    con.execute(f'''CREATE TEMP VIEW card AS SELECT c.* FROM {src}card c
        WHERE {known('c.issued')} AND EXISTS(SELECT 1 FROM disp d WHERE d.disp_id=c.disp_id)''')
    con.execute(f'''CREATE TEMP VIEW client AS SELECT c.client_id,c.gender,c.birth_date,c.district_id FROM {src}client c
        WHERE EXISTS(SELECT 1 FROM disp d WHERE d.client_id=c.client_id) AND c.birth_date<CAST({cut} AS DATE)''')
    con.execute(f'''CREATE TEMP VIEW district AS SELECT d.district_id FROM {src}district d
        WHERE EXISTS(SELECT 1 FROM account a WHERE a.district_id=d.district_id)
        OR EXISTS(SELECT 1 FROM client c WHERE c.district_id=d.district_id)''')
    con.execute(f'CREATE TEMP VIEW orders AS SELECT * FROM {src}orders WHERE FALSE')
    return con
