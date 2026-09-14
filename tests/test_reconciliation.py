import pandas as pd
import pytest

from m5_forecast.models.reconcile import reconcile_bottom_up


def test_summed_item_store_forecasts_match_higher_level_forecasts_within_tolerance():
    dates = pd.to_datetime(["2016-02-01", "2016-02-02"])
    item_store_forecast = pd.DataFrame(
        {
            "unique_id": (["ITEM_1_CA_1"] * 2) + (["ITEM_2_CA_1"] * 2) + (["ITEM_1_TX_1"] * 2),
            "date": list(dates) * 3,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    id_map = pd.DataFrame(
        {
            "unique_id": ["ITEM_1_CA_1", "ITEM_2_CA_1", "ITEM_1_TX_1"],
            "store_id": ["CA_1", "CA_1", "TX_1"],
            "state_id": ["CA", "CA", "TX"],
        }
    )

    store = reconcile_bottom_up(item_store_forecast, id_map, ["store_id"])
    state = reconcile_bottom_up(item_store_forecast, id_map, ["state_id"])
    total = reconcile_bottom_up(item_store_forecast, id_map, [])

    ca1_day1 = store.loc[(store["unique_id"] == "CA_1") & (store["date"] == dates[0]), "value"].iloc[0]
    assert ca1_day1 == pytest.approx(1.0 + 3.0, abs=1e-6)  # ITEM_1_CA_1 + ITEM_2_CA_1

    ca_state_day1 = state.loc[(state["unique_id"] == "CA") & (state["date"] == dates[0]), "value"].iloc[0]
    assert ca_state_day1 == pytest.approx(1.0 + 3.0, abs=1e-6)

    total_day1 = total.loc[total["date"] == dates[0], "value"].iloc[0]
    assert total_day1 == pytest.approx(1.0 + 3.0 + 5.0, abs=1e-6)
