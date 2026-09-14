import pandas as pd

HIERARCHY_LEVELS = ("item_store", "store", "state", "total")


_ID_COLUMNS = ("item_id", "store_id", "state_id", "dept_id", "cat_id")


def build_modelling_table(
    sales: pd.DataFrame, calendar: pd.DataFrame, sell_prices: pd.DataFrame
) -> pd.DataFrame:
    table = sales.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    # "d" was only needed as the join key to pull in date/wm_yr_wk; nothing
    # downstream uses it, and keeping it around risks the same
    # categorical-vs-string merge upcast handled below for the id columns.
    table = table.drop(columns="d")
    table = table.merge(
        sell_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    # FR-05: a week with no recorded price keeps its sales row; price-dependent
    # features must check has_price rather than treating a missing price as zero.
    table["has_price"] = table["sell_price"].notna()

    # Merging on item_id/store_id against sell_prices (a plain string frame)
    # upcasts those columns back to full string dtype; re-categorize so the
    # rest of the pipeline (grouped by these columns) stays memory-cheap.
    for col in _ID_COLUMNS:
        if col in table.columns:
            table[col] = table[col].astype("category")
    return table


def aggregate_hierarchy(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bottom-up series at each of item_store/store/state/total, each a long
    frame of unique_id, date, sales, dollar_sales. dollar_sales is 0 for rows
    with no recorded price (FR-05) so it never contributes to dollar-weighting.

    Mutates `table` in place (adds dollar_sales) rather than copying it --
    at full M5 scale the modelling table is tens of millions of rows, and this
    is called once by the caller, which doesn't need the input afterward."""
    table["dollar_sales"] = (table["sales"] * table["sell_price"].fillna(0)).astype("float32")

    # item_store is already at (item_id, store_id, date) granularity, so no
    # groupby is needed -- just build unique_id and select the columns.
    item_store = table[["item_id", "store_id", "date", "sales", "dollar_sales"]].copy()
    item_id, store_id = item_store.pop("item_id"), item_store.pop("store_id")
    # item_id/store_id are already low-cardinality categoricals (~thousands of
    # item_ids, ~10 stores). `.astype(str)` on the full column would build one
    # Python string object per row (millions of them) just to re-discover the
    # same few thousand distinct combinations -- factorizing that many string
    # objects is what was blowing up memory. Combine the (small) integer codes
    # instead, factorize *those*, and only format strings for the combinations
    # that actually occur.
    n_stores = len(store_id.cat.categories)
    combo_codes, combo_keys = pd.factorize(
        item_id.cat.codes.astype("int64") * n_stores + store_id.cat.codes.astype("int64")
    )
    item_labels = pd.Series(item_id.cat.categories[combo_keys // n_stores]).astype(str)
    store_labels = pd.Series(store_id.cat.categories[combo_keys % n_stores]).astype(str)
    labels = (item_labels + "_" + store_labels).to_numpy()
    item_store["unique_id"] = pd.Categorical.from_codes(combo_codes, categories=labels)

    def _agg(group_cols: list[str]) -> pd.DataFrame:
        grouped = table.groupby(group_cols + ["date"], as_index=False).agg(
            sales=("sales", "sum"), dollar_sales=("dollar_sales", "sum")
        )
        grouped["unique_id"] = grouped[group_cols].astype(str).agg("_".join, axis=1).astype("category")
        return grouped[["unique_id", "date", "sales", "dollar_sales"]]

    total = table.groupby("date", as_index=False).agg(sales=("sales", "sum"), dollar_sales=("dollar_sales", "sum"))
    total["unique_id"] = "TOTAL"

    return {
        "item_store": item_store[["unique_id", "date", "sales", "dollar_sales"]],
        "store": _agg(["store_id"]),
        "state": _agg(["state_id"]),
        "total": total[["unique_id", "date", "sales", "dollar_sales"]],
    }


def dollar_weights(level_df: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int) -> dict[str, float]:
    """Share of each series' dollar sales in the trailing window, per FR-13/level_wrmsse
    weighting. Falls back to equal weights if no priced sales fall in the window."""
    window = level_df[(level_df["date"] > as_of - pd.Timedelta(days=lookback_days)) & (level_df["date"] <= as_of)]
    totals = window.groupby("unique_id")["dollar_sales"].sum()
    grand_total = totals.sum()
    if grand_total == 0:
        ids = level_df["unique_id"].unique()
        return {uid: 1.0 / len(ids) for uid in ids}
    return (totals / grand_total).to_dict()
