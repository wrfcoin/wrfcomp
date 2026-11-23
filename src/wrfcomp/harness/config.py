"""Typed, frozen view of `configs/splits.yaml`, the single source for splits and gate thresholds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# src/wrfcomp/harness/config.py -> repository root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "splits.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _naive_utc_ns(value: datetime) -> np.datetime64:
    """Aware datetime -> timezone-naive UTC `datetime64[ns]`, the form the data contract uses."""
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ns")


class SplitConfig(_Frozen):
    start_utc: datetime
    end_utc: datetime | None
    embargo_days_after: int | None = Field(default=None, ge=0)
    purpose: str

    @model_validator(mode="after")
    def _check(self) -> SplitConfig:
        for name in ("start_utc", "end_utc"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must carry a UTC offset (for example a trailing 'Z')")
        if self.end_utc is not None and self.end_utc < self.start_utc:
            raise ValueError("end_utc is before start_utc")
        return self

    @property
    def start(self) -> np.datetime64:
        return _naive_utc_ns(self.start_utc)

    @property
    def end(self) -> np.datetime64 | None:
        return None if self.end_utc is None else _naive_utc_ns(self.end_utc)


class SplitsConfig(_Frozen):
    train: SplitConfig
    validation: SplitConfig
    sealed_test: SplitConfig

    @model_validator(mode="after")
    def _check(self) -> SplitsConfig:
        if self.train.end_utc is None or self.validation.end_utc is None:
            raise ValueError("train and validation splits need an end_utc")
        if not self.train.end_utc < self.validation.start_utc:
            raise ValueError("train split must end before validation starts")
        if not self.validation.end_utc < self.sealed_test.start_utc:
            raise ValueError("validation split must end before sealed_test starts")
        return self


class LossWeights(_Frozen):
    crps_fair: float = Field(ge=0.0)
    twcrps_p95: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> LossWeights:
        if abs(self.crps_fair + self.twcrps_p95 - 1.0) > 1e-12:
            raise ValueError("loss weights must sum to 1")
        return self


class PromotionGateConfig(_Frozen):
    alpha_dm: float = Field(gt=0.0, lt=1.0)
    min_crpss_vs_baseline: float
    spread_skill_min: float = Field(gt=0.0)
    spread_skill_max: float = Field(gt=0.0)
    tail_quantile: float = Field(gt=0.0, lt=1.0)
    tail_threshold_source: Literal["train"]
    # Smaller ensembles are scored but can never pass the gate.
    min_members_for_gate: int = Field(default=8, ge=2)
    loss_weights: LossWeights

    @model_validator(mode="after")
    def _check(self) -> PromotionGateConfig:
        if not self.spread_skill_min < self.spread_skill_max:
            raise ValueError("spread_skill_min must be below spread_skill_max")
        return self


class VariablesConfig(_Frozen):
    primary: tuple[str, ...] = Field(min_length=1)
    secondary_upper_air: tuple[str, ...] = ()


class EvaluationConfig(_Frozen):
    init_hours_utc: tuple[int, ...] = Field(min_length=1)
    lead_hours: tuple[int, ...] = Field(min_length=1)
    extended_lead_hours: tuple[int, ...] = ()
    truth_tolerance_minutes: int = Field(ge=0)
    variables: VariablesConfig
    stations_file: str
    promotion_gate: PromotionGateConfig

    @model_validator(mode="after")
    def _check(self) -> EvaluationConfig:
        if any(not 0 <= h <= 23 for h in self.init_hours_utc):
            raise ValueError("init_hours_utc must be within 0..23")
        if any(h <= 0 for h in (*self.lead_hours, *self.extended_lead_hours)):
            raise ValueError("lead hours must be positive")
        for name in ("init_hours_utc", "lead_hours", "extended_lead_hours"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} has duplicates")
        return self


class HarnessConfig(_Frozen):
    protocol_version: str
    splits: SplitsConfig
    evaluation: EvaluationConfig

    @property
    def gate(self) -> PromotionGateConfig:
        return self.evaluation.promotion_gate


def load_config(path: str | Path | None = None) -> HarnessConfig:
    """Load and validate the splits file. Defaults to `configs/splits.yaml` of this checkout."""
    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"harness config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return HarnessConfig.model_validate(raw)
