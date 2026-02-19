"""
etl.py — Extract, Transform, Load pipeline for E-com AI Analyst (V1)

Source : online_retail_II.xlsx  (Kaggle – UCI Online Retail II dataset)
Target : ecom.db               (SQLite, single table `orders`)

Cleaning rules
--------------
1. Combine both sheets (Year 2009-2010 & Year 2010-2011).
2. Drop rows where Customer ID is null  (anonymous → can't analyse).
3. Drop cancellations (Invoice starts with 'C').
4. Drop rows where Quantity <= 0 or Price <= 0 (bad data).
5. Create `Total_Sales = Quantity * Price`.
6. Normalise column names to snake_case (SQLite-friendly).
7. Write to SQLite `ecom.db` → table `orders`.
"""

import os
import sqlite3
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(
    BASE_DIR, "Online_retail(example to work with)", "online_retail_II.xlsx"
)
DB_PATH = os.path.join(BASE_DIR, "ecom.db")


def extract(xlsx_path: str) -> pd.DataFrame:
    """Read both sheets and concatenate into one DataFrame."""
    print("[1/4] Extracting data from Excel …")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype={"Invoice": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    print(f"       Raw rows loaded: {len(df):,}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning & enrichment rules."""
    print("[2/4] Transforming …")
    initial = len(df)

    # 1 — Drop null Customer IDs
    df = df.dropna(subset=["Customer ID"])
    print(f"       After dropping null Customer ID : {len(df):,}  (removed {initial - len(df):,})")

    # 2 — Drop cancellations (Invoice starts with 'C')
    before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    print(f"       After dropping cancellations    : {len(df):,}  (removed {before - len(df):,})")

    # 3 — Drop bad Quantity / Price
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"       After dropping bad qty/price    : {len(df):,}  (removed {before - len(df):,})")

    # 4 — Create Total_Sales
    df["Total_Sales"] = (df["Quantity"] * df["Price"]).round(2)

    # 5 — Normalise column names → snake_case
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # 6 — Ensure clean types
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    print(f"       Final clean rows                : {len(df):,}")
    return df


def load(df: pd.DataFrame, db_path: str) -> None:
    """Write DataFrame to SQLite, replacing any existing table."""
    print("[3/4] Loading into SQLite …")
    conn = sqlite3.connect(db_path)
    df.to_sql("orders", conn, if_exists="replace", index=False)

    # Quick sanity check
    row_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    print(f"       Rows in 'orders' table: {row_count:,}")

    conn.close()
    print(f"[4/4] Done → {db_path}")


def verify(db_path: str) -> None:
    """Print schema + sample queries so we know everything is solid."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    print("\n── Schema ──────────────────────────────────────────")
    for col in cur.execute("PRAGMA table_info(orders)").fetchall():
        print(f"    {col[1]:20s}  {col[2]}")

    print("\n── Sample rows ─────────────────────────────────────")
    for row in cur.execute("SELECT * FROM orders LIMIT 5").fetchall():
        print(f"    {row}")

    print("\n── Quick stats ─────────────────────────────────────")
    stats = cur.execute(
        """
        SELECT
            COUNT(*)                          AS total_orders,
            COUNT(DISTINCT invoice)           AS unique_invoices,
            COUNT(DISTINCT customer_id)       AS unique_customers,
            ROUND(SUM(total_sales), 2)        AS gross_revenue,
            MIN(invoicedate)                  AS first_date,
            MAX(invoicedate)                  AS last_date
        FROM orders
        """
    ).fetchone()
    labels = [
        "Total rows", "Unique invoices", "Unique customers",
        "Gross revenue", "First date", "Last date",
    ]
    for label, val in zip(labels, stats):
        print(f"    {label:20s}: {val}")

    conn.close()


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw = extract(XLSX_PATH)
    clean = transform(raw)
    load(clean, DB_PATH)
    verify(DB_PATH)
