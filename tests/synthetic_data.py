import numpy as np
import pandas as pd


def make_synthetic_m5_extract(num_days: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)

    items = ["ITEM_1", "ITEM_2"]
    stores = [("CA_1", "CA"), ("TX_1", "TX")]

    dates = pd.date_range("2016-01-01", periods=num_days, freq="D")
    day_cols = [f"d_{i + 1}" for i in range(num_days)]

    rows = []
    for item in items:
        for store_id, state_id in stores:
            sales = rng.integers(0, 5, size=num_days)
            rows.append(
                {
                    "id": f"{item}_{store_id}",
                    "item_id": item,
                    "dept_id": "HOBBIES_1",
                    "cat_id": "HOBBIES",
                    "store_id": store_id,
                    "state_id": state_id,
                    **{d: int(v) for d, v in zip(day_cols, sales)},
                }
            )
    sales_wide = pd.DataFrame(rows)

    calendar = pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": [11600 + (i // 7) for i in range(num_days)],
            "d": day_cols,
        }
    )

    weeks = sorted(calendar["wm_yr_wk"].unique())
    price_rows = [
        {"store_id": store_id, "item_id": item, "wm_yr_wk": week, "sell_price": 2.5}
        for item in items
        for store_id, _ in stores
        for week in weeks
    ]
    sell_prices = pd.DataFrame(price_rows)

    return sales_wide, calendar, sell_prices
