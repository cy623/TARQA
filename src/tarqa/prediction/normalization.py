from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.preprocessing import RobustScaler


@dataclass
class FeaturePreprocessor:
    mode: Literal["none", "robust"] = "robust"
    scaler: RobustScaler | None = None

    def fit(self, values: np.ndarray) -> "FeaturePreprocessor":
        values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.mode == "robust":
            self.scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
        elif self.mode != "none":
            raise ValueError(f"Unknown feature preprocessing mode: {self.mode}")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(values).astype(np.float32) if self.scaler else values

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class RegressionTargetTransformer:
    """Target transform that does not require a per-question ID at inference.

    `signed_log1p` is the recommended main setting for heterogeneous scalar targets.
    The robust variants are available as ablations but fit one shared domain-level scaler.
    """

    mode: Literal["none", "signed_log1p", "robust", "signed_log1p_robust"] = "signed_log1p"
    scaler: RobustScaler | None = None

    @staticmethod
    def _signed_log1p(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _inverse_signed_log1p(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.expm1(np.minimum(np.abs(values), 40.0))

    def fit(self, values: np.ndarray) -> "RegressionTargetTransformer":
        transformed = values.reshape(-1, 1).astype(np.float64)
        if self.mode in {"signed_log1p", "signed_log1p_robust"}:
            transformed = self._signed_log1p(transformed)
        if self.mode in {"robust", "signed_log1p_robust"}:
            self.scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(transformed)
        elif self.mode not in {"none", "signed_log1p"}:
            raise ValueError(f"Unknown regression target transform: {self.mode}")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = values.reshape(-1, 1).astype(np.float64)
        if self.mode in {"signed_log1p", "signed_log1p_robust"}:
            transformed = self._signed_log1p(transformed)
        if self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return transformed.reshape(-1).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        transformed = values.reshape(-1, 1).astype(np.float64)
        if self.scaler is not None:
            transformed = self.scaler.inverse_transform(transformed)
        if self.mode in {"signed_log1p", "signed_log1p_robust"}:
            transformed = self._inverse_signed_log1p(transformed)
        return transformed.reshape(-1).astype(np.float64)
