import pandera.pandas as pa
from pandera.pandas import Check, Column

CALENDAR_SCHEMA = pa.DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "wm_yr_wk": Column(int, nullable=False),
        "d": Column(str, Check.str_matches(r"^d_\d+$"), nullable=False),
    },
    strict=False,
)

SALES_LONG_SCHEMA = pa.DataFrameSchema(
    {
        "item_id": Column(str, nullable=False),
        "store_id": Column(str, nullable=False),
        "state_id": Column(str, nullable=False),
        "d": Column(str, nullable=False),
        "sales": Column(int, Check.ge(0), nullable=False),
    },
    strict=False,
)

# Validated pre-melt, on the compact wide frame -- id columns are about to be
# converted to `category` before melting (so their ~1900x replication across
# day rows stays a small int code, not a full string per cell), and pandera's
# `Column(str)` check requires an actual string dtype, not category.
SALES_WIDE_ID_SCHEMA = pa.DataFrameSchema(
    {
        "item_id": Column(str, nullable=False),
        "store_id": Column(str, nullable=False),
        "state_id": Column(str, nullable=False),
    },
    strict=False,
)

SALES_DAY_VALUES_SCHEMA = pa.DataFrameSchema(
    {r"^d_\d+$": Column(int, Check.ge(0), nullable=False, regex=True)},
    strict=False,
)

SELL_PRICES_SCHEMA = pa.DataFrameSchema(
    {
        "store_id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "wm_yr_wk": Column(int, nullable=False),
        "sell_price": Column(float, Check.gt(0), nullable=False),
    },
    strict=False,
)


class SchemaContractError(Exception):
    """Raised when a table violates its schema contract, naming the column and rule broken."""


def validate(df, schema: pa.DataFrameSchema, name: str):
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure = exc.failure_cases.iloc[0]
        raise SchemaContractError(
            f"{name}: column '{failure['column']}' violated rule '{failure['check']}' "
            f"(failure case: {failure['failure_case']!r})"
        ) from exc
