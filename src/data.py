"""Download the Online Retail II dataset, clean it and save it as parquet.

Usage:  python -m src.data
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_PATH = Path("data/processed/transactions.parquet")

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def find_local(data_dir: Path = DATA_DIR) -> Path | None:
    """Look for an existing raw file (.xlsx or .csv) anywhere under data/."""
    for pattern in ("*.xlsx", "*.csv"):
        for path in sorted(data_dir.rglob(pattern)):
            if "retail" in path.name.lower():
                return path
    return None


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Return the local raw file if there is one, otherwise download and extract it."""
    local = find_local()
    if local:
        print(f"Using local file: {local}")
        return local
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {URL}")
    with urllib.request.urlopen(URL) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    archive.extractall(raw_dir)
    return next(raw_dir.glob("*.xlsx"))


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw file; for .xlsx, combine both sheets (2009-2010 and 2010-2011)."""
    dtype = {"Invoice": str, "StockCode": str}
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=dtype, encoding="unicode_escape")
    sheets = pd.read_excel(path, sheet_name=None, dtype=dtype)
    return pd.concat(sheets.values(), ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and drop rows without a customer or with invalid values.

    Return rows (invoice number starts with 'C') are kept and flagged with is_return,
    because the return rate is a useful churn feature.
    """
    df = df.rename(columns=COLUMNS)
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice"] = df["invoice"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed")
    df["is_return"] = df["invoice"].str.startswith("C")
    df = df[(df["price"] > 0) & (df["quantity"] != 0)]
    df = df.drop_duplicates()
    df["revenue"] = df["quantity"] * df["price"]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild the processed file even if it exists")
    args = parser.parse_args()

    if PROCESSED_PATH.exists() and not args.force:
        print(f"{PROCESSED_PATH} already exists, nothing to do (use --force to rebuild).")
        return

    df = clean(load_raw(download()))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(
        f"{len(df):,} rows, {df['customer_id'].nunique():,} customers, "
        f"{df['invoice_date'].min():%Y-%m-%d} to {df['invoice_date'].max():%Y-%m-%d}"
    )
    print(f"Saved: {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
