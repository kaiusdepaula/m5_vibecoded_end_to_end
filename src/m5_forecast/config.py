from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SETTINGS_YAML = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _yaml_defaults() -> dict:
    if not _SETTINGS_YAML.exists():
        return {}
    return yaml.safe_load(_SETTINGS_YAML.read_text()) or {}


class BacktestSettings(BaseSettings):
    window_count: int = 3
    step_days: int = 28


class FeatureSettings(BaseSettings):
    lags: list[int] = Field(default_factory=lambda: [7, 14, 28])
    rolling_means: list[int] = Field(default_factory=lambda: [7, 28])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    horizon: int = 28
    random_seed: int = 42
    sample_series: int | None = 3000

    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)

    hierarchy_weight_lookback_days: int = 28
    reconciliation_tolerance: float = 1e-6
    reproducibility_tolerance: float = 1e-4

    mlflow_tracking_uri: str = "http://localhost:5000"
    registered_model_name: str = "m5_hierarchical"
    model_alias: str = "champion"


def load_settings() -> Settings:
    return Settings(**_yaml_defaults())
