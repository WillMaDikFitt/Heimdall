"""Gradient-boosting hourly forecaster with recursive multi-step prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.forecast.features import DEFAULT_LAGS, DEFAULT_ROLLING, add_features, feature_columns

WET_THRESHOLD_MM = 0.1


def _fill_exo_nans(featured: pd.DataFrame) -> pd.DataFrame:
    """Keep lag rows even when some NWP member fields are missing."""
    out = featured.copy()
    exo_cols = [col for col in out.columns if str(col).startswith("exo_")]
    if exo_cols:
        out[exo_cols] = out[exo_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


class HourlyGradientBooster:
    def __init__(
        self,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        rolling: tuple[int, ...] = DEFAULT_ROLLING,
        random_state: int = 42,
    ) -> None:
        self.lags = lags
        self.rolling = rolling
        self.model = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            random_state=random_state,
        )
        self.feature_names_: list[str] = []

    def fit(
        self,
        hourly: pd.DataFrame,
        exogenous: pd.DataFrame | None = None,
    ) -> HourlyGradientBooster:
        """Fit on a frame with DatetimeIndex and column `y` (training window only)."""
        featured = add_features(
            hourly[["y"]], lags=self.lags, rolling=self.rolling, exogenous=exogenous
        )
        featured = _fill_exo_nans(featured).dropna()
        if featured.empty:
            raise ValueError("Not enough training rows after feature engineering")
        self.feature_names_ = feature_columns(featured)
        self.model.fit(featured[self.feature_names_], featured["y"])
        self._exogenous = exogenous
        return self

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_index: pd.DatetimeIndex,
        exogenous: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Roll forward hour by hour, appending each prediction into the history."""
        exo = exogenous if exogenous is not None else getattr(self, "_exogenous", None)
        hist = history.dropna().astype(float).copy()
        if hist.empty:
            return np.full(len(forecast_index), np.nan)

        preds: list[float] = []
        for ts in forecast_index:
            temp = hist.copy()
            temp.loc[ts] = np.nan
            featured = _fill_exo_nans(
                add_features(
                    temp.to_frame(name="y"), lags=self.lags, rolling=self.rolling, exogenous=exo
                )
            )
            if ts not in featured.index:
                yhat = float(hist.iloc[-1])
            else:
                row = featured.loc[ts, self.feature_names_]
                if row.isna().any():
                    yhat = float(hist.iloc[-1])
                else:
                    yhat = float(self.model.predict(row.to_frame().T)[0])
            preds.append(yhat)
            hist.loc[ts] = yhat
        return np.asarray(preds, dtype=float)


class TwoStageRainForecaster:
    """
    Rain as two decisions: will the hour be wet, then how much?

    Stage 1 classifies wet hours (>= 0.1 mm). Stage 2 regresses amount on wet
    hours only. Dry hours are forced to 0 mm so the model stops drifting to a
    mid-range drizzle.
    """

    def __init__(
        self,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        rolling: tuple[int, ...] = DEFAULT_ROLLING,
        wet_threshold_mm: float = WET_THRESHOLD_MM,
        wet_prob_threshold: float = 0.45,
        random_state: int = 42,
    ) -> None:
        self.lags = lags
        self.rolling = rolling
        self.wet_threshold_mm = wet_threshold_mm
        self.wet_prob_threshold = wet_prob_threshold
        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            random_state=random_state,
        )
        self.amount_model = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            random_state=random_state,
        )
        self.feature_names_: list[str] = []
        self._exogenous: pd.DataFrame | None = None
        self._wet_fallback = 0.0
        self._amount_fallback = wet_threshold_mm

    def fit(
        self,
        hourly: pd.DataFrame,
        exogenous: pd.DataFrame | None = None,
    ) -> TwoStageRainForecaster:
        featured = add_features(
            hourly[["y"]], lags=self.lags, rolling=self.rolling, exogenous=exogenous
        )
        featured = _fill_exo_nans(featured).dropna()
        if featured.empty:
            raise ValueError("Not enough training rows after feature engineering")
        self.feature_names_ = feature_columns(featured)
        x = featured[self.feature_names_]
        y = featured["y"].to_numpy(dtype=float)
        wet = (y >= self.wet_threshold_mm).astype(int)
        self._wet_fallback = float(wet.mean()) if len(wet) else 0.0
        if len(np.unique(wet)) < 2:
            # One-class window: fall back to amount-only regression with a wet gate.
            self.classifier.fit(x, np.ones(len(x), dtype=int) if wet.any() else np.zeros(len(x), dtype=int))
        else:
            self.classifier.fit(x, wet)

        wet_rows = featured.loc[wet.astype(bool)]
        if len(wet_rows) >= 8:
            self.amount_model.fit(wet_rows[self.feature_names_], wet_rows["y"])
            self._amount_fallback = float(np.median(wet_rows["y"]))
        else:
            # Too few wet hours: fall back to a constant wet-hour amount.
            self._amount_fallback = float(np.median(y[y >= self.wet_threshold_mm])) if wet.any() else self.wet_threshold_mm
            self.amount_model.fit(x, np.clip(y, 0.0, None))

        self._exogenous = exogenous
        return self

    def _predict_one(self, row: pd.Series) -> float:
        if row.isna().any():
            return 0.0 if self._wet_fallback < self.wet_prob_threshold else self._amount_fallback
        frame = row.to_frame().T
        proba = float(self.classifier.predict_proba(frame)[0][1])
        if proba < self.wet_prob_threshold:
            return 0.0
        amount = float(self.amount_model.predict(frame)[0])
        return float(max(self.wet_threshold_mm, amount))

    def predict_recursive(
        self,
        history: pd.Series,
        forecast_index: pd.DatetimeIndex,
        exogenous: pd.DataFrame | None = None,
    ) -> np.ndarray:
        exo = exogenous if exogenous is not None else self._exogenous
        hist = history.dropna().astype(float).copy()
        if hist.empty:
            return np.zeros(len(forecast_index), dtype=float)

        preds: list[float] = []
        for ts in forecast_index:
            temp = hist.copy()
            temp.loc[ts] = np.nan
            featured = _fill_exo_nans(
                add_features(
                    temp.to_frame(name="y"), lags=self.lags, rolling=self.rolling, exogenous=exo
                )
            )
            if ts not in featured.index:
                yhat = 0.0
            else:
                yhat = self._predict_one(featured.loc[ts, self.feature_names_])
            preds.append(yhat)
            hist.loc[ts] = yhat
        return np.asarray(preds, dtype=float)


def allocate_to_sum(shape: np.ndarray, target_sum: float) -> np.ndarray:
    """Scale a non-negative hourly shape so its sum matches target_sum."""
    values = np.clip(np.asarray(shape, dtype=float), 0.0, None)
    target = max(0.0, float(target_sum))
    if values.size == 0:
        return values
    total = float(values.sum())
    if total <= 1e-9:
        return np.full(values.size, target / values.size, dtype=float)
    return values * (target / total)


class DirectHorizonRainModel:
    """
    Predict the rain *total* over the next H hours in one shot.

    Uses lag/calendar features plus optional NWP/reference fields, including the
    model rain sum over the same horizon when that series is available.
    """

    def __init__(
        self,
        horizon_hours: int = 24,
        lags: tuple[int, ...] = DEFAULT_LAGS,
        rolling: tuple[int, ...] = DEFAULT_ROLLING,
        random_state: int = 42,
    ) -> None:
        self.horizon_hours = int(horizon_hours)
        self.lags = lags
        self.rolling = rolling
        self.model = GradientBoostingRegressor(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.07,
            subsample=0.9,
            random_state=random_state,
        )
        self.feature_names_: list[str] = []
        self._exogenous: pd.DataFrame | None = None
        self._fallback = 0.0

    def _row_features(
        self,
        history: pd.Series,
        origin: pd.Timestamp,
        exogenous: pd.DataFrame | None,
    ) -> pd.Series | None:
        hist = history.dropna().astype(float).copy()
        if origin not in hist.index:
            hist.loc[origin] = np.nan
        featured = add_features(
            hist.to_frame(name="y"),
            lags=self.lags,
            rolling=self.rolling,
            exogenous=exogenous,
        )
        featured = _fill_exo_nans(featured)
        if origin not in featured.index:
            return None
        row = featured.loc[origin].copy()
        if exogenous is not None and not exogenous.empty:
            if "nwp_rain_mean" in exogenous.columns:
                exo_rain = pd.to_numeric(exogenous["nwp_rain_mean"], errors="coerce")
            elif "rainfall_mm" in exogenous.columns:
                exo_rain = pd.to_numeric(exogenous["rainfall_mm"], errors="coerce")
            else:
                exo_rain = None
            if exo_rain is not None:
                window = pd.date_range(origin, periods=self.horizon_hours, freq="h", tz="UTC")
                row["exo_rain_sum_h"] = float(exo_rain.reindex(window).fillna(0.0).sum())
            else:
                row["exo_rain_sum_h"] = 0.0
        else:
            row["exo_rain_sum_h"] = 0.0
        return row

    def fit(
        self,
        hourly: pd.DataFrame,
        exogenous: pd.DataFrame | None = None,
    ) -> DirectHorizonRainModel:
        y = hourly["y"].dropna().astype(float)
        if len(y) <= self.horizon_hours + max(self.lags):
            raise ValueError("Not enough history for direct-horizon rain training")

        rows: list[pd.Series] = []
        targets: list[float] = []
        index = y.index.sort_values()
        last_start = len(index) - self.horizon_hours
        for i in range(max(self.lags) + 1, last_start):
            origin = index[i]
            target_idx = index[i : i + self.horizon_hours]
            if len(target_idx) < self.horizon_hours:
                continue
            hist = y.loc[:origin]
            row = self._row_features(hist, origin, exogenous)
            if row is None:
                continue
            target = float(y.loc[target_idx].sum())
            rows.append(row)
            targets.append(target)

        if len(rows) < 24:
            raise ValueError("Not enough direct-horizon training rows")

        frame = pd.DataFrame(rows)
        feature_frame = frame.drop(columns=["y"], errors="ignore")
        self.feature_names_ = [
            col for col in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[col])
        ]
        x = feature_frame[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        mask = x.notna().all(axis=1)
        x = x.loc[mask]
        y_train = np.asarray(targets, dtype=float)[mask.to_numpy()]
        if len(x) < 24:
            raise ValueError("Not enough clean rows for direct-horizon rain training")
        self.model.fit(x, np.log1p(np.clip(y_train, 0.0, None)))
        self._fallback = float(np.median(y_train))
        self._exogenous = exogenous
        return self

    def predict_sum(
        self,
        history: pd.Series,
        origin: pd.Timestamp,
        exogenous: pd.DataFrame | None = None,
    ) -> float:
        exo = exogenous if exogenous is not None else self._exogenous
        row = self._row_features(history, origin, exo)
        if row is None:
            return self._fallback
        values = row.reindex(self.feature_names_)
        if values.isna().any():
            return self._fallback
        raw = float(self.model.predict(values.to_frame().T)[0])
        return float(max(0.0, np.expm1(raw)))

    def predict_block_profile(
        self,
        history: pd.Series,
        forecast_index: pd.DatetimeIndex,
        shape: np.ndarray,
        exogenous: pd.DataFrame | None = None,
        block_hours: int = 24,
    ) -> np.ndarray:
        """Predict each block total and allocate onto an hourly timing shape."""
        exo = exogenous if exogenous is not None else self._exogenous
        hist = history.dropna().astype(float).copy()
        out = np.zeros(len(forecast_index), dtype=float)
        if forecast_index.empty:
            return out

        block = max(1, int(block_hours))
        shape_arr = np.asarray(shape, dtype=float)
        if shape_arr.shape[0] != len(forecast_index):
            shape_arr = np.ones(len(forecast_index), dtype=float)

        for start in range(0, len(forecast_index), block):
            end = min(start + block, len(forecast_index))
            origin = pd.Timestamp(forecast_index[start])
            if origin.tzinfo is None:
                origin = origin.tz_localize("UTC")
            old_h = self.horizon_hours
            self.horizon_hours = end - start
            try:
                total = self.predict_sum(hist, origin, exogenous=exo)
            finally:
                self.horizon_hours = old_h
            out[start:end] = allocate_to_sum(shape_arr[start:end], total)
        return out

