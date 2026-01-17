"""High-level machine-learning forecasting helpers with covariate support.

This module provides a suite of forecasting utilities that automatically treat
all non-date, non-target columns as covariates, infer the sampling frequency,
and iteratively extend the series when only a univariate history is provided.

Each function exposes an "advanced error correction" step that smooths recent
residuals using an exponential window and removes the learned bias from the
raw forecast. The helpers also synthesise reasonable future covariate values
when none are supplied by forward filling the most recent observation, keeping
forecasts stable even when only partial inputs are available.

At least one function wraps Chronos-2 so users can leverage foundation-model
probabilistic forecasts while still benefiting from the package's covariate and
error-correction logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .forecasting import chronos2_forecast

try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception:  # pragma: no cover - optional dependency
    BayesianRidge = None  # type: ignore
    GradientBoostingRegressor = None  # type: ignore
    HistGradientBoostingRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    ElasticNet = None  # type: ignore
    HuberRegressor = None  # type: ignore
    PLSRegression = None  # type: ignore
    Ridge = None  # type: ignore
    mean_squared_error = None  # type: ignore
    MultiOutputRegressor = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore
    SVR = None  # type: ignore


@dataclass
class MLForecastResult:
    """Standardised return object for the ML forecast helpers."""

    forecasts: pd.Series
    model_used: str
    metadata: Dict[str, object]


@dataclass
class FeatureEngineeringResult:
    """Container for generated features and aligned targets."""

    features: pd.DataFrame
    target: pd.Series
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_sklearn() -> None:
    if GradientBoostingRegressor is None or Ridge is None:
        raise ImportError(
            "scikit-learn is required for these machine-learning forecasts. "
            "Install via `pip install analysis3054[ml]` to enable them."
        )


def _infer_frequency(dates: pd.Series, freq: Optional[str]) -> Tuple[str, pd.DatetimeIndex]:
    dt = pd.to_datetime(dates)
    inferred = freq or pd.infer_freq(dt)
    if inferred is None:
        diffs = dt.diff().dropna()
        delta = diffs.mode()[0] if not diffs.empty else pd.Timedelta(days=1)
        inferred = pd.tseries.frequencies.to_offset(delta).freqstr
    future_index = pd.date_range(start=dt.iloc[-1], periods=2, freq=inferred)[1:]
    return inferred, future_index


def _future_index(dates: pd.Series, periods: int, freq: Optional[str]) -> pd.DatetimeIndex:
    inferred, first_step = _infer_frequency(dates, freq)
    return pd.date_range(start=first_step[0], periods=periods, freq=inferred)


def _apply_error_correction(
    true_values: pd.Series, in_sample_preds: pd.Series, forecasts: pd.Series
) -> pd.Series:
    residuals = (true_values - in_sample_preds).dropna()
    if residuals.empty:
        return forecasts
    bias = residuals.ewm(span=min(24, len(residuals)), adjust=False).mean().iloc[-1]
    volatility = residuals.ewm(span=min(24, len(residuals)), adjust=False).std().iloc[-1]
    corrected = forecasts - bias
    if volatility and np.isfinite(volatility):
        corrected = corrected.clip(
            lower=forecasts.median() - 3 * volatility,
            upper=forecasts.median() + 3 * volatility,
        )
    return corrected


def _select_covariates(df: pd.DataFrame, date_col: str, target_col: str, covariate_cols: Optional[Sequence[str]]) -> List[str]:
    if covariate_cols is not None:
        return list(covariate_cols)
    return [c for c in df.columns if c not in {date_col, target_col}]


def auto_generate_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    lags: int = 3,
    rolling_windows: Sequence[int] = (3,),
    fourier_order: int = 2,
    include_calendar: bool = True,
    freq: Optional[str] = None,
) -> FeatureEngineeringResult:
    """Create supervised-learning features with leakage-safe transformations.

    The helper validates inputs, infers frequency when not supplied, and returns a
    :class:`FeatureEngineeringResult` containing an aligned feature matrix,
    target vector, and metadata describing the engineered columns.
    """

    if not len(df):
        raise ValueError("Input dataframe must contain observations.")
    if date_col not in df.columns or target_col not in df.columns:
        raise KeyError("Both date_col and target_col must exist in the dataframe.")
    if lags < 1:
        raise ValueError("lags must be at least 1.")
    if any(w < 1 for w in rolling_windows):
        raise ValueError("All rolling windows must be positive integers.")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    inferred_freq, _ = _infer_frequency(ordered[date_col], freq)

    base = ordered[[date_col, target_col]].copy()
    features = pd.DataFrame(index=base.index)
    lag_cols: List[str] = []
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        features[col] = base[target_col].shift(lag)
        lag_cols.append(col)

    rolling_cols: List[str] = []
    for window in rolling_windows:
        col = f"roll_mean_{window}"
        features[col] = base[target_col].shift(1).rolling(window=window, min_periods=window).mean()
        rolling_cols.append(col)

    calendar_cols: List[str] = []
    if include_calendar:
        dt_index = pd.to_datetime(base[date_col])
        calendar_cols = ["dayofweek", "month", "is_month_end"]
        features["dayofweek"] = dt_index.dt.dayofweek
        features["month"] = dt_index.dt.month
        features["is_month_end"] = dt_index.dt.is_month_end.astype(int)

    fourier_cols: List[str] = []
    if fourier_order > 0:
        t = np.arange(len(base))
        period = max(2, int(round(len(base) / 4)))
        for k in range(1, fourier_order + 1):
            sine_col = f"fourier_sin_{k}"
            cos_col = f"fourier_cos_{k}"
            features[sine_col] = np.sin(2 * np.pi * k * t / period)
            features[cos_col] = np.cos(2 * np.pi * k * t / period)
            fourier_cols.extend([sine_col, cos_col])

    combined = pd.concat([features, base[target_col]], axis=1).dropna()
    if combined.empty:
        raise ValueError("No usable rows after feature generation; check lags/rolling windows.")

    target = combined[target_col]
    engineered = combined.drop(columns=[target_col])
    metadata = {
        "frequency": inferred_freq,
        "lag_features": lag_cols,
        "rolling_features": rolling_cols,
        "calendar_features": calendar_cols,
        "fourier_features": fourier_cols,
    }
    return FeatureEngineeringResult(features=engineered, target=target, metadata=metadata)


def _split_features_targets(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    lags: int,
    covariate_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if date_col in df.columns:
        df = df.drop(columns=[date_col])
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    feature_cols = [c for c in df.columns if c not in {target_col, "idx"}]
    df = df.dropna().reset_index(drop=True)
    X = df[feature_cols]
    y = df[target_col]
    if covariate_cols:
        X = X.assign(**{c: df[c] for c in covariate_cols if c in df.columns})
    return X, y


def _iterative_predict(
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    covariate_cols: List[str],
    lags: int,
    freq: Optional[str],
) -> Tuple[pd.Series, Dict[str, object]]:
    _require_sklearn()
    X, y = _split_features_targets(
        df[[date_col, target_col] + covariate_cols], date_col, target_col, lags, covariate_cols
    )
    feature_order = list(X.columns)
    model = model_factory()
    model.fit(X, y)
    in_sample_preds = pd.Series(model.predict(X), index=y.index)

    history = df[[date_col, target_col] + covariate_cols].copy()
    future_vals: List[float] = []
    future_index = _future_index(history[date_col], horizon, freq)

    for step in range(horizon):
        window = history.iloc[-lags:]
        features = {f"lag_{i+1}": window[target_col].iloc[-(i + 1)] for i in range(lags)}
        for cov in covariate_cols:
            if cov in history.columns:
                features[cov] = history[cov].iloc[-1]
        ordered_values = [features.get(col) for col in feature_order]
        feature_frame = pd.DataFrame([ordered_values], columns=feature_order)
        pred = model.predict(feature_frame)[0]
        future_vals.append(pred)
        new_row = {date_col: future_index[step], target_col: pred}
        for cov in covariate_cols:
            new_row[cov] = history[cov].iloc[-1]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    forecasts = pd.Series(future_vals, index=future_index, name=target_col)
    corrected = _apply_error_correction(y, in_sample_preds, forecasts)
    metadata = {"model": model}
    if mean_squared_error is not None:
        metadata["in_sample_rmse"] = float(np.sqrt(mean_squared_error(y, in_sample_preds)))
    return corrected, metadata


def _append_fourier_features(
    df: pd.DataFrame, date_col: str, *, max_order: int = 3
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach Fourier seasonal terms to a dataframe based on index ordering."""

    enhanced = df.copy()
    t = np.arange(len(enhanced))
    period = max(2, int(round(len(enhanced) / 4)))
    fourier_cols: List[str] = []
    for k in range(1, max_order + 1):
        sine_col = f"fourier_sin_{k}"
        cos_col = f"fourier_cos_{k}"
        enhanced[sine_col] = np.sin(2 * np.pi * k * t / period)
        enhanced[cos_col] = np.cos(2 * np.pi * k * t / period)
        fourier_cols.extend([sine_col, cos_col])
    return enhanced, fourier_cols


def _build_direct_training_matrix(
    df: pd.DataFrame,
    *,
    target_col: str,
    covariate_cols: List[str],
    horizon: int,
    lags: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a direct multi-step training matrix for tree boosters."""

    rows: List[Dict[str, float]] = []
    targets: List[Sequence[float]] = []
    usable = df.reset_index(drop=True)

    for start in range(lags, len(usable) - horizon):
        window = usable.iloc[start - lags : start]
        feature_row = {f"lag_{i+1}": window[target_col].iloc[-(i + 1)] for i in range(lags)}
        for cov in covariate_cols:
            feature_row[cov] = usable[cov].iloc[start - 1]
        rows.append(feature_row)
        targets.append(usable[target_col].iloc[start : start + horizon].values)

    return pd.DataFrame(rows), pd.DataFrame(targets)


# ---------------------------------------------------------------------------
# Public forecast helpers
# ---------------------------------------------------------------------------

def chronos2_auto_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    future_covariates: Optional[pd.DataFrame] = None,
    quantile_levels: Optional[List[float]] = None,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Use Chronos‑2 with automatic covariate detection and bias correction."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    res = chronos2_forecast(
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        covariate_cols=covars,
        future_cov_df=future_covariates,
        prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    forecasts = res.forecasts[target_col]
    inferred_freq, _ = _infer_frequency(df[date_col], freq)
    forecasts.index = pd.date_range(start=df[date_col].iloc[-1], periods=horizon + 1, freq=inferred_freq)[1:]
    diffs = df[target_col].diff().dropna()
    naive_preds = df[target_col].iloc[-1] + diffs.mean() * np.arange(1, horizon + 1)
    corrected = _apply_error_correction(df[target_col].iloc[-len(naive_preds) :], pd.Series(naive_preds.values), forecasts)
    metadata = res.__dict__ | {"applied_frequency": inferred_freq}
    return MLForecastResult(forecasts=corrected, model_used="chronos2_auto", metadata=metadata)


def gradient_boosting_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Gradient Boosting regressor with lag features and covariate repeats."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: GradientBoostingRegressor(random_state=42),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="gradient_boosting", metadata=metadata)


def random_forest_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 8,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Random Forest regressor with residual-based bias removal."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: RandomForestRegressor(n_estimators=300, random_state=7),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="random_forest", metadata=metadata)


def ridge_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 6,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Ridge regression with standardised features and error trimming."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=0.5)),
            ]
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="ridge", metadata=metadata)


def elastic_net_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 6,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Elastic Net regression with covariates treated as exogenous signals."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(alpha=0.2, l1_ratio=0.3)),
            ]
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="elastic_net", metadata=metadata)


def svr_high_frequency_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 48,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """RBF-kernel SVR tuned for high-frequency intraday patterns."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(C=10.0, gamma="scale", epsilon=0.05)),
            ]
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="svr", metadata=metadata)


def xgboost_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """XGBoost regressor with iterative covariate-aware forecasting."""

    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "xgboost is required for xgboost_covariate_forecast. Install via `pip install analysis3054[ml]`."
        ) from e

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=11,
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="xgboost", metadata=metadata)


def lightgbm_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """LightGBM regressor with lagged targets and forward-filled covariates."""

    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "lightgbm is required for lightgbm_covariate_forecast. Install via `pip install analysis3054[ml]`."
        ) from e

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=13,
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="lightgbm", metadata=metadata)


def catboost_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """CatBoost regressor that natively handles categorical covariates."""

    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "catboost is required for catboost_covariate_forecast. Install via `pip install analysis3054[ml]`."
        ) from e

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    categorical_indices = [i for i, c in enumerate(covars) if pd.api.types.is_object_dtype(df[c])]

    def factory() -> object:
        return CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=21,
            cat_features=categorical_indices or None,
            verbose=False,
        )

    forecasts, metadata = _iterative_predict(
        model_factory=factory,
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    metadata["categorical_covariates"] = covars
    return MLForecastResult(forecasts=forecasts, model_used="catboost", metadata=metadata)


def bayesian_ridge_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 10,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Bayesian Ridge regression with automatic frequency inference and bias removal."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline([("scaler", StandardScaler()), ("bayes", BayesianRidge())]),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    return MLForecastResult(forecasts=forecasts, model_used="bayesian_ridge", metadata=metadata)


def huber_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 8,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Robust Huber regression to dampen the influence of outliers."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline([("scaler", StandardScaler()), ("huber", HuberRegressor())]),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    metadata["robust_loss"] = "huber"
    return MLForecastResult(forecasts=forecasts, model_used="huber", metadata=metadata)


def pls_covariate_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 6,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Partial least squares regression that projects covariates into latent factors."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    n_components = max(2, min(5, lags + len(covars)))
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=n_components)),
            ]
        ),
        df=df[[date_col, target_col] + covars],
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars,
        lags=lags,
        freq=freq,
    )
    metadata["latent_components"] = n_components
    return MLForecastResult(forecasts=forecasts, model_used="pls", metadata=metadata)


def fourier_ridge_seasonal_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 10,
    freq: Optional[str] = None,
    fourier_order: int = 3,
) -> MLForecastResult:
    """Ridge regression enriched with Fourier seasonal signatures."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    augmented, fourier_cols = _append_fourier_features(df[[date_col, target_col] + covars], date_col, max_order=fourier_order)
    forecasts, metadata = _iterative_predict(
        model_factory=lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.3))]),
        df=augmented,
        date_col=date_col,
        target_col=target_col,
        horizon=horizon,
        covariate_cols=covars + fourier_cols,
        lags=lags,
        freq=freq,
    )
    metadata["fourier_terms"] = fourier_cols
    return MLForecastResult(forecasts=forecasts, model_used="fourier_ridge", metadata=metadata)


def hist_gradient_direct_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 12,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Direct multi-horizon forecast using HistGradientBoostingRegressor."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)
    training_df = df[[date_col, target_col] + covars].copy()
    X_direct, Y_direct = _build_direct_training_matrix(
        training_df, target_col=target_col, covariate_cols=covars, horizon=horizon, lags=lags
    )
    if X_direct.empty or Y_direct.empty:
        raise ValueError("Not enough history to train direct multioutput forecaster.")
    _require_sklearn()
    if MultiOutputRegressor is None:
        raise ImportError("scikit-learn multioutput support is required for direct forecasting.")
    base_estimator = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400)
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_direct, Y_direct)
    direct_preds = pd.DataFrame(model.predict(X_direct), index=Y_direct.index, columns=Y_direct.columns)

    history = training_df[[date_col, target_col] + covars].copy()
    future_index = _future_index(history[date_col], horizon, freq)
    last_window = history.iloc[-lags:]
    feature_row = {f"lag_{i+1}": last_window[target_col].iloc[-(i + 1)] for i in range(lags)}
    for cov in covars:
        feature_row[cov] = history[cov].iloc[-1]
    forecast_array = model.predict(pd.DataFrame([feature_row]))[0]
    forecast_series = pd.Series(forecast_array, index=future_index, name=target_col)

    corrected = _apply_error_correction(
        Y_direct.iloc[:, 0],
        direct_preds.iloc[:, 0],
        forecast_series,
    )
    metadata = {
        "strategy": "direct_multioutput",
        "in_sample_rmse_step1": float(np.sqrt(mean_squared_error(Y_direct.iloc[:, 0], direct_preds.iloc[:, 0])))
        if mean_squared_error is not None
        else None,
    }
    return MLForecastResult(forecasts=corrected, model_used="hist_gradient_direct", metadata=metadata)


def stacked_meta_ensemble_forecast(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    horizon: int = 24,
    covariate_cols: Optional[Sequence[str]] = None,
    lags: int = 8,
    freq: Optional[str] = None,
) -> MLForecastResult:
    """Stacked ensemble that blends ridge, random forest, and boosting predictions."""

    covars = _select_covariates(df, date_col, target_col, covariate_cols)

    def base_models_factory() -> List[Tuple[str, object]]:
        _require_sklearn()
        return [
            ("ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.4))])),
            ("forest", RandomForestRegressor(n_estimators=200, random_state=17)),
            ("gbr", GradientBoostingRegressor(random_state=5)),
        ]

    X, y = _split_features_targets(df[[date_col, target_col] + covars], date_col, target_col, lags, covars)
    base_models = base_models_factory()
    base_preds = []
    for name, model in base_models:
        model.fit(X, y)
        base_preds.append(pd.Series(model.predict(X), index=y.index, name=name))
    stack_frame = pd.concat(base_preds, axis=1)
    meta = Ridge(alpha=0.2)
    meta.fit(stack_frame, y)
    in_sample_preds = pd.Series(meta.predict(stack_frame), index=y.index)

    history = df[[date_col, target_col] + covars].copy()
    future_index = _future_index(history[date_col], horizon, freq)
    forecasts: List[float] = []

    for step in range(horizon):
        window = history.iloc[-lags:]
        lag_features = {f"lag_{i+1}": window[target_col].iloc[-(i + 1)] for i in range(lags)}
        covar_features = {c: history[c].iloc[-1] for c in covars}
        base_features = {}
        for name, model in base_models:
            temp_input = pd.DataFrame([{**lag_features, **covar_features}])
            base_features[name] = model.predict(temp_input)[0]
        meta_input = pd.DataFrame([base_features])
        pred = meta.predict(meta_input)[0]
        forecasts.append(pred)
        new_row = {date_col: future_index[step], target_col: pred, **covar_features}
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    forecast_series = pd.Series(forecasts, index=future_index, name=target_col)
    corrected = _apply_error_correction(y, in_sample_preds, forecast_series)
    metadata = {"base_models": [name for name, _ in base_models], "meta_model": meta}
    return MLForecastResult(forecasts=corrected, model_used="stacked_meta", metadata=metadata)


__all__ = [
    "FeatureEngineeringResult",
    "MLForecastResult",
    "auto_generate_features",
    "chronos2_auto_covariate_forecast",
    "gradient_boosting_covariate_forecast",
    "random_forest_covariate_forecast",
    "ridge_covariate_forecast",
    "elastic_net_covariate_forecast",
    "bayesian_ridge_covariate_forecast",
    "huber_covariate_forecast",
    "pls_covariate_forecast",
    "fourier_ridge_seasonal_forecast",
    "hist_gradient_direct_forecast",
    "svr_high_frequency_forecast",
    "xgboost_covariate_forecast",
    "lightgbm_covariate_forecast",
    "catboost_covariate_forecast",
    "stacked_meta_ensemble_forecast",
]
