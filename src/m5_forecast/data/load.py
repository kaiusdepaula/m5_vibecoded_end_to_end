from pathlib import Path

import pandas as pd

from .schema import (
    CALENDAR_SCHEMA,
    SALES_DAY_VALUES_SCHEMA,
    SALES_WIDE_ID_SCHEMA,
    SELL_PRICES_SCHEMA,
    validate,
)

# Force plain object dtype for identifier columns at read time. pandas' newer
# default string dtype (and its internal dictionary/categorical encodings) has
# been observed to desync codes vs. categories when melting a wide, large,
# high-cardinality frame -- explicit object dtype sidesteps that path entirely.
_SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
_CALENDAR_STRING_COLUMNS = ["d"]
_SELL_PRICES_STRING_COLUMNS = ["store_id", "item_id"]


def melt_sales(wide: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    id_cols = [c for c in wide.columns if c not in day_cols]
    return wide.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")


def load_calendar(path: Path) -> pd.DataFrame:
    dtype = {col: object for col in _CALENDAR_STRING_COLUMNS}
    df = pd.read_csv(path, parse_dates=["date"], dtype=dtype)
    validate(df, CALENDAR_SCHEMA, "calendar")
    df["d"] = df["d"].astype("category")
    return df


def load_sell_prices(path: Path) -> pd.DataFrame:
    dtype = {col: object for col in _SELL_PRICES_STRING_COLUMNS}
    df = pd.read_csv(path, dtype=dtype)
    validate(df, SELL_PRICES_SCHEMA, "sell_prices")

    # Categorical to match sales' item_id/store_id dtype (see load_sales):
    # merging a categorical column against a plain string column forces
    # pandas to materialize one side in full, which is exactly the expensive
    # path we're avoiding by categorizing sales' id columns in the first place.
    for col in _SELL_PRICES_STRING_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def load_sales(path: Path, sample_series: int | None = None, random_state: int | None = None) -> pd.DataFrame:
    # Declare every day column's dtype explicitly rather than letting pandas
    # infer it: with ~1,913 undeclared-dtype columns, this file has repeatedly
    # triggered C-parser dtype-inference bugs (a low_memory=True chunking
    # KeyError, and separately a handful of day columns silently coming back
    # as object dtype instead of int64) -- inference is the common thread, so
    # removing it for every column is the actual fix rather than working
    # around each symptom.
    day_cols = [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("d_")]
    dtype = {col: object for col in _SALES_ID_COLUMNS} | {col: "int64" for col in day_cols}
    # low_memory=False: even with dtype declared, this file's ~1,913 columns
    # have shown chunk-concatenation issues under the chunked C-parser
    # (low_memory=True); reading in one pass avoids that entirely and easily
    # fits in memory at this row count.
    wide = pd.read_csv(path, dtype=dtype, low_memory=False)

    if sample_series is not None and sample_series < len(wide):
        wide = wide.sample(n=sample_series, random_state=random_state).reset_index(drop=True)

    validate(wide, SALES_WIDE_ID_SCHEMA, "sales")
    validate(wide, SALES_DAY_VALUES_SCHEMA, "sales")

    # Convert to category before melting: id columns replicate across every
    # day row (~1900x for a full M5 extract), and category dtype keeps that
    # replication a small int code instead of a full string object per cell.
    for col in _SALES_ID_COLUMNS:
        wide[col] = wide[col].astype("category")

    # Validated above as int64 (pandera requires it); downcast afterward, once,
    # on the compact wide frame -- so the melted long frame's sales column,
    # duplicated across ~1900 days, is int32 (4 bytes) from the start rather
    # than int64 (8 bytes). No need to re-validate "sales" post-melt: every
    # value it can hold was already checked above, melt only reshapes.
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    wide[day_cols] = wide[day_cols].astype("int32")

    long_df = melt_sales(wide)
    # melt's var_name column ("d") is built from the day-column labels, which
    # pandas would otherwise store as a full string per row (~1900x
    # replication); category dtype matches calendar's "d" and load_sell_prices'
    # id columns above.
    long_df["d"] = long_df["d"].astype("category")
    return long_df
