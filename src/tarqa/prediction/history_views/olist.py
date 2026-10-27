"""Historical input views under the visibility assumptions stated in README.

These are feature-preparation views, not access controls: the read-only raw
attachment remains accessible. Label SQL must run against the complete source
snapshot in a separate label-generation step, never against these input views.
"""
from pathlib import Path
import duckdb
from .common import dt, quote, stamp


def open_history(database, reference_time, release_root=None):
    cutoff = "TIMESTAMP " + quote(stamp(dt(reference_time)))
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH {quote(Path(database).resolve())} AS source_snapshot (READ_ONLY)")
    src = "source_snapshot.main."
    actual_dates = ("order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date")
    replacements = ", ".join(f"CASE WHEN {column} <= {cutoff} THEN {column} ELSE NULL END AS {column}" for column in actual_dates)
    con.execute(f"""CREATE TEMP VIEW olist_orders_dataset AS
        SELECT * EXCLUDE (order_status) REPLACE ({replacements})
        FROM {src}olist_orders_dataset WHERE order_purchase_timestamp <= {cutoff}""")
    for table in ("olist_order_items_dataset", "olist_order_payments_dataset"):
        con.execute(f"""CREATE TEMP VIEW {table} AS
            SELECT d.* FROM {src}{table} d
            WHERE EXISTS (SELECT 1 FROM olist_orders_dataset o WHERE o.order_id=d.order_id)""")
    con.execute(f"""CREATE TEMP VIEW olist_order_reviews_dataset AS
        SELECT r.* FROM {src}olist_order_reviews_dataset r
        WHERE r.review_creation_date <= {cutoff} AND r.review_answer_timestamp <= {cutoff}
          AND EXISTS (SELECT 1 FROM olist_orders_dataset o WHERE o.order_id=r.order_id)""")
    for table, key in (("olist_sellers_dataset", "seller_id"), ("olist_products_dataset", "product_id")):
        con.execute(f"""CREATE TEMP VIEW {table} AS
            SELECT e.* FROM {src}{table} e
            WHERE EXISTS (SELECT 1 FROM olist_order_items_dataset i WHERE i.{key}=e.{key})""")
    con.execute(f"""CREATE TEMP VIEW olist_customers_dataset AS
        SELECT c.* FROM {src}olist_customers_dataset c
        WHERE EXISTS (SELECT 1 FROM olist_orders_dataset o WHERE o.customer_id=c.customer_id)""")
    for table in ("olist_geolocation_dataset", "product_category_name_translation"):
        con.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM {src}{table}")
    return con
