import pandas as pd


def reconcile_bottom_up(item_store_values: pd.DataFrame, id_map: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Sum item_store-level values up to a higher level. item_store_values has
    columns unique_id, date, value; id_map maps unique_id to store_id/state_id.
    group_cols=[] aggregates to a single TOTAL series."""
    merged = item_store_values.merge(id_map, on="unique_id", how="left")

    if not group_cols:
        total = merged.groupby("date", as_index=False)["value"].sum()
        total["unique_id"] = "TOTAL"
        return total[["unique_id", "date", "value"]]

    grouped = merged.groupby(group_cols + ["date"], as_index=False)["value"].sum()
    if len(group_cols) > 1:
        grouped["unique_id"] = grouped[group_cols].astype(str).agg("_".join, axis=1)
    else:
        grouped["unique_id"] = grouped[group_cols[0]]
    return grouped[["unique_id", "date", "value"]]


def build_id_map(modelling_table: pd.DataFrame) -> pd.DataFrame:
    id_map = modelling_table[["item_id", "store_id", "state_id"]].drop_duplicates().copy()
    id_map["unique_id"] = id_map["item_id"].astype(str) + "_" + id_map["store_id"].astype(str)
    return id_map[["unique_id", "store_id", "state_id"]]
