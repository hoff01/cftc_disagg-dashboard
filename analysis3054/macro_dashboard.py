"""Macro indicator focus datasets and dashboard builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .auto_ml_forecasting import gradient_boosting_covariate_forecast
from .forecasting import chronos2_quantile_forecast
from .eia_distillate import fetch_distillate_product_supplied, EIADataError
from .diesel_indicators import DIESEL_DEMAND_IDS


@dataclass(frozen=True)
class MacroOutputs:
    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    distillate: pd.DataFrame
    distillate_forecast: pd.DataFrame


MACRO_META: Dict[str, Dict[str, str]] = {
    "credit_card_debt": {
        "label": "Credit Card Debt (All Commercial Banks)",
        "unit": "USD (billions)",
        "description": "Balances on credit card and revolving plans at commercial banks.",
    },
    "revolving_credit": {
        "label": "Revolving Consumer Credit",
        "unit": "USD (billions)",
        "description": "Total revolving credit outstanding.",
    },
    "personal_savings_rate": {
        "label": "Personal Savings Rate",
        "unit": "Percent",
        "description": "Personal saving as a percentage of disposable income.",
    },
    "inflation_cpi": {
        "label": "CPI (All Items)",
        "unit": "Index",
        "description": "Consumer Price Index for All Urban Consumers.",
    },
    "inflation_yoy": {
        "label": "CPI YoY Inflation",
        "unit": "Percent",
        "description": "Year-over-year CPI percent change.",
    },
    "real_gdp": {
        "label": "Real GDP",
        "unit": "USD (billions, chained)",
        "description": "Inflation-adjusted GDP.",
    },
    "gdp_yoy": {
        "label": "Real GDP YoY",
        "unit": "Percent",
        "description": "Year-over-year real GDP growth.",
    },
    "fed_funds": {
        "label": "Fed Funds Rate",
        "unit": "Percent",
        "description": "Effective federal funds rate.",
    },
    "treasury_10y": {
        "label": "Treasury 10Y",
        "unit": "Percent",
        "description": "10-year Treasury constant maturity rate.",
    },
    "treasury_2y": {
        "label": "Treasury 2Y",
        "unit": "Percent",
        "description": "2-year Treasury constant maturity rate.",
    },
    "treasury_3mo": {
        "label": "Treasury 3M",
        "unit": "Percent",
        "description": "3-month Treasury bill rate.",
    },
    "yield_spread_10y_2y": {
        "label": "Yield Spread 10Y-2Y",
        "unit": "Percent",
        "description": "10-year minus 2-year Treasury yield spread.",
    },
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet",
        "unit": "USD (billions)",
        "description": "Total assets held by the Federal Reserve.",
    },
    "unemp_to_job_openings": {
        "label": "Unemployed / Job Openings",
        "unit": "Ratio",
        "description": "Unemployed persons divided by job openings.",
    },
    "pmi": {
        "label": "PMI / ISM",
        "unit": "Index",
        "description": "Manufacturing PMI/ISM indicator (if available).",
    },
    "distillate_product_supplied": {
        "label": "Distillate Fuel Oil Product Supplied",
        "unit": "Thousand barrels/day",
        "description": "EIA product supplied for distillate fuel oil.",
    },
}


def _safe_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for cand in candidates:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def build_macro_focus_frames(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = {}
    
    data["credit_card_debt"] = _safe_series(monthly_df, ["CCLACBW027SBOG"])
    data["revolving_credit"] = _safe_series(monthly_df, ["REVOLSL"])
    data["personal_savings_rate"] = _safe_series(monthly_df, ["PSAVERT"])
    data["inflation_cpi"] = _safe_series(monthly_df, ["CPIAUCSL"])
    data["inflation_yoy"] = data["inflation_cpi"].pct_change(12, fill_method=None) * 100.0

    real_gdp = _safe_series(monthly_df, ["GDPC1", "GDP"])
    if real_gdp.isna().all() and "GDPC1" in quarterly_df.columns:
        real_gdp = quarterly_df["GDPC1"].reindex(monthly_df.index, method="ffill")
    data["real_gdp"] = real_gdp
    data["gdp_yoy"] = real_gdp.pct_change(12, fill_method=None) * 100.0

    data["fed_funds"] = _safe_series(monthly_df, ["FEDFUNDS"])
    data["treasury_10y"] = _safe_series(monthly_df, ["DGS10", "GS10"])
    data["treasury_2y"] = _safe_series(monthly_df, ["DGS2", "GS2"])
    data["treasury_3mo"] = _safe_series(monthly_df, ["DGS3MO", "TB3MS"])
    
    if "T10Y2Y" in monthly_df.columns:
        data["yield_spread_10y_2y"] = monthly_df["T10Y2Y"]
    else:
        data["yield_spread_10y_2y"] = data["treasury_10y"] - data["treasury_2y"]

    data["fed_balance_sheet"] = _safe_series(monthly_df, ["WALCL"])

    unemployed = _safe_series(monthly_df, ["UNEMPLOY"])
    job_openings = _safe_series(monthly_df, ["JTSJOL"])
    data["unemp_to_job_openings"] = unemployed / job_openings.replace(0.0, np.nan)

    data["pmi"] = _safe_series(monthly_df, ["NAPM", "PMI"])

    monthly = pd.DataFrame(data, index=monthly_df.index).dropna(how="all")
    quarterly = monthly.resample("QE").mean()
    return monthly, quarterly


def build_diesel_indicator_frames(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    indicator_ids: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(indicator_ids or DIESEL_DEMAND_IDS)
    if monthly_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    data = {}
    for series_id in ids:
        if series_id in monthly_df.columns:
            data[series_id] = pd.to_numeric(monthly_df[series_id], errors="coerce")
        elif series_id in quarterly_df.columns:
            data[series_id] = (
                pd.to_numeric(quarterly_df[series_id], errors="coerce")
                .reindex(monthly_df.index, method="ffill")
            )
        else:
            data[series_id] = np.nan
            
    monthly = pd.DataFrame(data, index=monthly_df.index)
    quarterly = monthly.resample("QE").mean()
    return monthly, quarterly


def fetch_distillate_monthly(
    *,
    api_key: Optional[str] = None,
    series_id: Optional[str] = None,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    try:
        series = fetch_distillate_product_supplied(api_key=api_key, series_id=series_id)
    except EIADataError:
        return pd.DataFrame()
    dist = series.data.copy()
    dist = dist.set_index("date").sort_index()
    dist_monthly = dist.resample("ME").mean()
    if start_date:
        dist_monthly = dist_monthly.loc[pd.to_datetime(start_date) :]
    dist_monthly.columns = ["distillate_product_supplied"]
    return dist_monthly


def build_distillate_forecast(
    distillate_monthly: pd.DataFrame,
    macro_monthly: pd.DataFrame,
    *,
    horizon: int = 12,
    use_chronos2: bool = False,
    target_col: Optional[str] = None,
    freq: str = "ME",
) -> pd.DataFrame:
    if distillate_monthly.empty:
        return pd.DataFrame()

    if target_col is None:
        if "distillate_product_supplied" in distillate_monthly.columns:
            target_col = "distillate_product_supplied"
        else:
            target_col = distillate_monthly.columns[0]

    # Join with outer to preserve future covariates
    full_df = distillate_monthly[[target_col]].join(macro_monthly, how="outer")
    
    # Identify last actual target date
    valid_target = full_df[target_col].dropna()
    if valid_target.empty:
        return pd.DataFrame()
    last_date = valid_target.index.max()
    
    # Split history
    df = full_df.loc[:last_date].copy().dropna(subset=[target_col])
    df = df.reset_index().rename(columns={"index": "date", target_col: "target"})
    
    # Determine future window
    freq_str = freq or pd.infer_freq(valid_target.index) or "ME"
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_str)[1:]
    
    # Ensure full_df covers the future window
    full_df = full_df.reindex(full_df.index.union(future_dates))
    
    # Extract future covariates
    future_rows = full_df.loc[future_dates].copy() if not future_dates.empty else pd.DataFrame()
    
    # Identify known vs past covariates
    potential_covariates = [c for c in macro_monthly.columns if c in full_df.columns]
    known_covariates = []
    past_covariates = []
    
    for col in potential_covariates:
        # Check history presence
        if df[col].isna().all():
            continue
            
        # Check future presence
        if not future_rows.empty and future_rows[col].notna().all():
            known_covariates.append(col)
        else:
            past_covariates.append(col)
            
    all_covariates = known_covariates + past_covariates
    
    if use_chronos2:
        try:
            future_cov_df = None
            if all_covariates and not future_rows.empty:
                future_cov_df = future_rows[all_covariates].reset_index().rename(columns={"index": "date"})
            
            res = chronos2_quantile_forecast(
                df,
                date_col="date",
                target_col="target",
                covariate_cols=all_covariates or None,
                future_cov_df=future_cov_df,
                prediction_length=horizon,
                quantile_levels=[0.2, 0.5, 0.8],
                device_map="cpu",
            )
            forecast = res.forecasts
            if isinstance(forecast.columns, pd.MultiIndex):
                forecast = forecast.droplevel(0, axis=1)
            forecast.columns = [float(c) for c in forecast.columns]
            out = pd.DataFrame({
                "date": forecast.index,
                "p20": forecast.get(0.2, np.nan),
                "p50": forecast.get(0.5, np.nan),
                "p80": forecast.get(0.8, np.nan),
            }).reset_index(drop=True)
            out["model"] = "chronos2"
            return out
        except Exception:
            pass

    # Fallback to GBM (Gradient Boosting) which handles covariates via lag features
    try:
        result = gradient_boosting_covariate_forecast(
            df,
            date_col="date",
            target_col="target",
            horizon=horizon,
            covariate_cols=all_covariates or None,
            freq=freq,
        )
        forecast_series = result.forecasts
        model_used = result.model_used
    except ImportError:
        last_value = df["target"].iloc[-1]
        forecast_series = pd.Series([last_value] * horizon, index=future_dates)
        model_used = "naive"

    residuals = df["target"].diff().dropna()
    spread = residuals.std() if not residuals.empty else df["target"].std()
    if pd.isna(spread):
        spread = 0.0
    offset = 0.84 * spread
    out = pd.DataFrame({
        "date": forecast_series.index,
        "p20": (forecast_series - offset).clip(lower=0.0),
        "p50": forecast_series,
        "p80": (forecast_series + offset).clip(lower=0.0),
    }).reset_index(drop=True)
    out["model"] = model_used
    return out


def build_macro_outputs(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    *,
    distillate_api_key: Optional[str] = None,
    distillate_series_id: Optional[str] = None,
    use_chronos2: bool = False,
) -> MacroOutputs:
    macro_monthly, macro_quarterly = build_macro_focus_frames(monthly_df, quarterly_df)
    distillate_monthly = fetch_distillate_monthly(
        api_key=distillate_api_key, series_id=distillate_series_id
    )
    forecast = build_distillate_forecast(
        distillate_monthly,
        macro_monthly,
        horizon=12,
        use_chronos2=use_chronos2,
    )
    return MacroOutputs(
        monthly=macro_monthly,
        quarterly=macro_quarterly,
        distillate=distillate_monthly,
        distillate_forecast=forecast,
    )


__all__ = [
    "MACRO_META",
    "build_diesel_indicator_frames",
    "build_macro_focus_frames",
    "build_macro_outputs",
    "build_distillate_forecast",
    "fetch_distillate_monthly",
    "MacroOutputs",
]
