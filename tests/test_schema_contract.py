from pathlib import Path

import pandas as pd
import pytest

from m5_forecast.data.build_features import build_modelling_table
from m5_forecast.data.load import melt_sales
from m5_forecast.data.schema import (
    SchemaContractError,
    validate,
    SALES_LONG_SCHEMA,
    CALENDAR_SCHEMA,
    SELL_PRICES_SCHEMA,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _synthetic_sales_wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["ITEM_1_CA_1", "ITEM_2_CA_1"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["HOBBIES_1", "HOBBIES_1"],
            "cat_id": ["HOBBIES", "HOBBIES"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "d_1": [0, 1],
            "d_2": [0, 2],
            "d_3": [0, 0],
            "d_4": [0, 3],
            "d_5": [0, 1],
            "d_6": [0, 0],
        }
    )


def _synthetic_calendar() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "wm_yr_wk": [11601] * 6,
            "d": [f"d_{i}" for i in range(1, 7)],
        }
    )


def _synthetic_sell_prices() -> pd.DataFrame:
    # ITEM_1 has no price row for wm_yr_wk 11601 -- the missing-price case (FR-05).
    return pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["ITEM_2"],
            "wm_yr_wk": [11601],
            "sell_price": [2.5],
        }
    )


def test_valid_frames_join_into_expected_modelling_table():
    sales = validate(_synthetic_sales_wide().pipe(melt_sales), SALES_LONG_SCHEMA, "sales")
    calendar = validate(_synthetic_calendar(), CALENDAR_SCHEMA, "calendar")
    prices = _synthetic_sell_prices()

    table = build_modelling_table(sales, calendar, prices)

    assert set(table.columns) >= {"item_id", "store_id", "state_id", "date", "sales", "has_price"}
    assert len(table) == 12  # 2 items x 6 days


def test_all_zero_series_is_retained_not_dropped():
    sales = validate(_synthetic_sales_wide().pipe(melt_sales), SALES_LONG_SCHEMA, "sales")

    item_1_rows = sales[sales["item_id"] == "ITEM_1"]
    assert len(item_1_rows) == 6
    assert (item_1_rows["sales"] == 0).all()


def test_missing_price_week_keeps_sales_row_with_has_price_false():
    sales = validate(_synthetic_sales_wide().pipe(melt_sales), SALES_LONG_SCHEMA, "sales")
    calendar = validate(_synthetic_calendar(), CALENDAR_SCHEMA, "calendar")
    prices = _synthetic_sell_prices()

    table = build_modelling_table(sales, calendar, prices)

    item_1_rows = table[table["item_id"] == "ITEM_1"]
    assert len(item_1_rows) == 6
    assert not item_1_rows["has_price"].any()
    assert item_1_rows["sell_price"].isna().all()

    item_2_rows = table[table["item_id"] == "ITEM_2"]
    assert item_2_rows["has_price"].all()


def test_corrupted_sell_price_column_fails_loudly_naming_column_and_rule():
    corrupted = pd.read_csv(FIXTURES / "corrupted_sell_prices.csv")

    with pytest.raises(SchemaContractError) as excinfo:
        validate(corrupted, SELL_PRICES_SCHEMA, "sell_prices")

    message = str(excinfo.value)
    assert "sell_price" in message
    assert "greater_than" in message
