"""Unified forecasting interface for analysis3054.

This module introduces :class:`ForecastEngine`, a lightweight orchestration
layer that exposes a consistent API across the package's diverse forecasting
routines. It standardises input handling (date column, target columns and
optional covariates) and allows users to register additional model backends
while retaining access to advanced utilities such as Chronos‑2.

The engine emphasises composability:

* Built‑in adapters wrap existing helpers like :func:`harmonic_forecast` and
  :func:`chronos2_forecast` so they can be triggered with a single
  :meth:`ForecastEngine.forecast` call.
* Covariates can be passed directly and forwarded to models that support
  them (e.g., Chronos‑2).
* Results include the consolidated forecast DataFrame and any model‑specific
  artefacts for downstream analysis.

In addition to the main engine, convenience helpers produce a default engine
with sensible registrations and a small result dataclass to capture metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from .advanced import harmonic_forecast
from .forecasting import chronos2_forecast, intraday_sarimax_forecast


@dataclass
class EngineForecastResult:
    """Standardised forecast result for :class:`ForecastEngine`.

    Attributes
    ----------
    forecasts : pd.DataFrame
        DataFrame of point forecasts produced by the selected model.
    model_used : str
        Identifier of the backend model.
    metadata : Dict[str, Any]
        Optional extras returned by the backend (e.g., raw model objects,
        quantile intervals or plots).
    """

    forecasts: pd.DataFrame
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


ModelHandler = Callable[..., EngineForecastResult]


class ForecastEngine:
    """Registry and dispatcher for forecasting backends.

    Parameters
    ----------
    default_model : str or None, default None
        Name of the model to use when ``model`` is not specified in
        :meth:`forecast`.
    """

    def __init__(self, default_model: Optional[str] = None) -> None:
        self._models: Dict[str, ModelHandler] = {}
        self.default_model = default_model

    def register_model(self, name: str, handler: ModelHandler) -> None:
        """Register a model handler.

        Parameters
        ----------
        name : str
            Unique name for the model.
        handler : callable
            Function that returns :class:`EngineForecastResult`.
        """

        self._models[name] = handler

    def available_models(self) -> List[str]:
        """Return the list of registered model names."""

        return sorted(self._models.keys())

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        date_col: str,
        target_cols: Optional[Sequence[str]] = None,
        horizon: int = 24,
        model: Optional[str] = None,
        covariate_cols: Optional[List[str]] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> EngineForecastResult:
        """Run a forecast using a registered backend.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data containing the date and target columns.
        date_col : str
            Name of the datetime column.
        target_cols : sequence of str or None, default None
            Target columns to forecast.  If None, all numeric columns other
            than ``date_col`` are used.
        horizon : int, default 24
            Forecast horizon to pass to the backend.
        model : str or None, default None
            Registered model name.  If None, ``default_model`` is used.
        covariate_cols : list of str or None, default None
            Columns to treat as covariates for models that support them.
        future_covariates : pd.DataFrame or None, default None
            Future covariate values aligned on ``date_col``.
        **kwargs : Any
            Additional model‑specific keyword arguments.
        """

        model_name = model or self.default_model
        if model_name is None:
            raise ValueError("No model specified and no default_model configured")
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' is not registered with the engine")

        targets: List[str]
        if target_cols is None:
            targets = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        else:
            targets = list(target_cols)
        if not targets:
            raise ValueError("No target columns supplied for forecasting")

        handler = self._models[model_name]
        return handler(
            df=df,
            date_col=date_col,
            target_cols=targets,
            horizon=horizon,
            covariate_cols=covariate_cols,
            future_covariates=future_covariates,
            **kwargs,
        )


def _harmonic_handler(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_cols: List[str],
    horizon: int,
    **_: Any,
) -> EngineForecastResult:
    subset = df[[date_col] + target_cols].copy()
    result = harmonic_forecast(date=date_col, df=subset, periods=horizon)
    return EngineForecastResult(forecasts=result.forecasts[target_cols], model_used="harmonic", metadata={"raw": result})


def _chronos2_handler(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_cols: List[str],
    horizon: int,
    covariate_cols: Optional[List[str]] = None,
    future_covariates: Optional[pd.DataFrame] = None,
    quantile_levels: Optional[List[float]] = None,
    **kwargs: Any,
) -> EngineForecastResult:
    forecasts: Dict[str, pd.Series] = {}
    metadata: Dict[str, Any] = {"raw_results": {}}

    for target in target_cols:
        result = chronos2_forecast(
            df=df[[date_col] + ([target] if target_cols else []) + (covariate_cols or [])],
            date_col=date_col,
            target_col=target,
            covariate_cols=covariate_cols,
            future_cov_df=future_covariates,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
            **kwargs,
        )
        forecasts[target] = result.forecasts[target]
        metadata["raw_results"][target] = result
        if result.lower_conf_int is not None:
            metadata.setdefault("lower_conf_int", {})[target] = result.lower_conf_int[target]
        if result.upper_conf_int is not None:
            metadata.setdefault("upper_conf_int", {})[target] = result.upper_conf_int[target]

    forecast_df = pd.DataFrame(forecasts)
    return EngineForecastResult(forecasts=forecast_df, model_used="chronos2", metadata=metadata)


def _intraday_sarimax_handler(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_cols: List[str],
    horizon: int,
    load_df: pd.DataFrame,
    **kwargs: Any,
) -> EngineForecastResult:
    burn_subset = df[[date_col] + target_cols]
    res = intraday_sarimax_forecast(
        date=date_col,
        df=burn_subset,
        load_df=load_df,
        periods=horizon,
        **kwargs,
    )
    metadata: Dict[str, Any] = {"raw": res}
    if res.lower_conf_int is not None:
        metadata["lower_conf_int"] = res.lower_conf_int
    if res.upper_conf_int is not None:
        metadata["upper_conf_int"] = res.upper_conf_int
    return EngineForecastResult(forecasts=res.forecasts[target_cols], model_used="intraday_sarimax", metadata=metadata)


def build_default_engine() -> ForecastEngine:
    """Create a :class:`ForecastEngine` with common backends registered."""

    engine = ForecastEngine(default_model="harmonic")
    engine.register_model("harmonic", _harmonic_handler)
    engine.register_model("chronos2", _chronos2_handler)
    engine.register_model("intraday_sarimax", _intraday_sarimax_handler)
    return engine


__all__ = [
    "EngineForecastResult",
    "ForecastEngine",
    "build_default_engine",
]
