"""Lightweight model leaderboard and ensemble utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _align_frames(actual: pd.DataFrame, forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = actual.index.intersection(forecast.index)
    common_cols = [c for c in actual.columns if c in forecast.columns]
    return actual.loc[common_index, common_cols], forecast.loc[common_index, common_cols]


def model_leaderboard(
    actual: pd.DataFrame,
    forecasts: Dict[str, pd.DataFrame],
    *,
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute simple error metrics for a collection of forecast DataFrames."""

    metrics = list(metrics) if metrics is not None else ["mae", "rmse", "mape"]
    rows = []
    for name, forecast_df in forecasts.items():
        aligned_actual, aligned_forecast = _align_frames(actual, forecast_df)
        if aligned_actual.empty or aligned_forecast.empty:
            continue
        errors = aligned_forecast - aligned_actual
        summary: Dict[str, float] = {"model": name}
        if "mae" in metrics:
            summary["mae"] = errors.abs().mean().mean()
        if "rmse" in metrics:
            summary["rmse"] = np.sqrt((errors ** 2).mean().mean())
        if "mape" in metrics:
            denom = aligned_actual.replace(0, np.nan).abs()
            summary["mape"] = (errors.abs() / denom).mean().mean() * 100
        rows.append(summary)

    leaderboard_df = pd.DataFrame(rows)
    if not leaderboard_df.empty:
        leaderboard_df = leaderboard_df.sort_values(by=[c for c in ["mae", "rmse"] if c in leaderboard_df.columns])
    return leaderboard_df.reset_index(drop=True)


def simple_ensemble(
    forecasts: Dict[str, pd.DataFrame],
    *,
    weights: Optional[Dict[str, float]] = None,
    strategy: str = "mean",
) -> pd.DataFrame:
    """Combine multiple forecast DataFrames using a simple strategy."""

    if not forecasts:
        raise ValueError("No forecasts provided for ensembling")

    first_df = next(iter(forecasts.values()))
    index = first_df.index
    columns = first_df.columns

    stacked = []
    for name, df in forecasts.items():
        aligned = df.reindex(index=index, columns=columns)
        stacked.append(aligned.astype(float))

    data = np.stack([df.values for df in stacked], axis=0)
    if weights is not None:
        weight_arr = np.array([weights.get(name, 0.0) for name in forecasts.keys()], dtype=float)
        if weight_arr.sum() == 0:
            raise ValueError("Provided weights sum to zero")
        norm_weights = weight_arr / weight_arr.sum()
        combined = np.tensordot(norm_weights, data, axes=1)
    elif strategy == "median":
        combined = np.nanmedian(data, axis=0)
    else:
        combined = np.nanmean(data, axis=0)

    return pd.DataFrame(combined, index=index, columns=columns)


__all__ = ["model_leaderboard", "simple_ensemble"]
