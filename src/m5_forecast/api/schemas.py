from typing import Literal, Optional

from pydantic import BaseModel, model_validator

Level = Literal["item_store", "store", "state", "total"]


class ForecastRequest(BaseModel):
    level: Level
    series_id: Optional[str] = None
    key: Optional[str] = None

    @model_validator(mode="after")
    def check_identifier(self) -> "ForecastRequest":
        if self.level == "item_store" and not self.series_id:
            raise ValueError("series_id is required when level is 'item_store'")
        if self.level in ("store", "state") and not self.key:
            raise ValueError(f"key is required when level is '{self.level}'")
        return self

    @property
    def lookup_key(self) -> str:
        if self.level == "item_store":
            return self.series_id
        if self.level == "total":
            return "TOTAL"
        return self.key


class ForecastPoint(BaseModel):
    date: str
    value: float


class ModelRef(BaseModel):
    name: str
    version: int


class ForecastResponse(BaseModel):
    level: Level
    series_id: Optional[str] = None
    key: Optional[str] = None
    model: ModelRef
    horizon: int
    forecast: list[ForecastPoint]
    request_id: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
