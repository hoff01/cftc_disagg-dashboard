"""
Finance‑oriented functions for time‑series analysis.

This module introduces additional utilities tailored to financial
analysis, particularly useful for commodity traders seeking to
understand risk and relationships among price series.  The functions
operate on pandas DataFrames and employ rolling statistics and
adjustments for trading volume.

Included are:

* :func:`liquidity_adjusted_volatility` – Compute volatility adjusted
  for liquidity by dividing returns by the square root of trading
  volume before applying a rolling standard deviation.  This
  normalises price movements by liquidity, highlighting periods of
  heightened illiquidity.

* :func:`rolling_beta` – Calculate the rolling beta of a target
  series relative to a benchmark series using a specified window.
  Beta measures the sensitivity of the target to the benchmark and is
  widely used to assess hedge ratios and exposure.

Each function returns a structured result object containing the
relevant series and intermediate data used in the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class LiquidityAdjustedVolatilityResult:
    """Result container for liquidity‑adjusted volatility.

    Attributes
    ----------
    volatility : pandas.DataFrame
        DataFrame of adjusted volatility for each price column.
    returns : pandas.DataFrame
        DataFrame of raw returns for each price column.
    adjusted_returns : pandas.DataFrame
        DataFrame of liquidity‑adjusted returns (returns divided by
        ``sqrt(volume)``) used in volatility calculation.
    """
    volatility: pd.DataFrame
    returns: pd.DataFrame
    adjusted_returns: pd.DataFrame


def liquidity_adjusted_volatility(
    date: Union[str, pd.Series, Iterable],
    df: pd.DataFrame,
    *,
    price_cols: Optional[List[str]] = None,
    volume_col: str = 'volume',
    window: int = 20,
    return_type: str = 'log',
    annualize_factor: Optional[float] = None,
) -> LiquidityAdjustedVolatilityResult:
    """Compute liquidity‑adjusted volatility for one or more price series.

    This function normalises returns by dividing by the square root of
    trading volume before computing a rolling standard deviation.
    Illiquid periods, characterised by low volume, thus yield higher
    adjusted volatility.  The adjustment resembles the Amihud
    illiquidity measure but uses a square‑root scaling for symmetry.

    Parameters
    ----------
    date : str or iterable
        Column in ``df`` containing dates, or an iterable of dates.
    df : pandas.DataFrame
        DataFrame with price and volume data.  If ``price_cols`` is
        ``None``, all numeric columns except ``volume_col`` and
        ``date`` are treated as price series.
    price_cols : list of str or None, default None
        Names of price columns.  If ``None``, use all numeric columns
        other than ``volume_col`` and ``date``.
    volume_col : str, default ``'volume'``
        Name of the column containing trading volume.
    window : int, default 20
        Rolling window size for volatility calculation.
    return_type : {'log','pct'}, default 'log'
        Type of return: log returns or percentage change.
    annualize_factor : float or None, default None
        Optional factor for annualising volatility (multiplying by
        ``sqrt(annualize_factor)``).

    Returns
    -------
    LiquidityAdjustedVolatilityResult
        Object containing volatility, raw returns, and adjusted returns.

    Notes
    -----
    The square‑root of volume scaling assumes that price impact is
    inversely proportional to the square root of liquidity.  Users
    should ensure that volume data are strictly positive; zeros are
    replaced by ``NaN`` during adjustment.
    """
    # Determine date index
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        dt_index = pd.to_datetime(df[date])
    else:
        dt_index = pd.to_datetime(pd.Series(date))
    # Identify price columns
    if price_cols is None:
        price_cols = [c for c in df.columns if c not in {date, volume_col} and pd.api.types.is_numeric_dtype(df[c])]
    if not price_cols:
        raise ValueError("No price columns found for liquidity‑adjusted volatility")
    # Extract volume and replace zeros to avoid division by zero
    vol = pd.to_numeric(df[volume_col], errors='coerce').replace(0, np.nan)
    sqrt_vol = np.sqrt(vol)
    returns_data = {}
    adj_returns_data = {}
    vol_data = {}
    for col in price_cols:
        series = pd.to_numeric(df[col], errors='coerce')
        # Compute returns
        if return_type == 'log':
            ret = np.log(series).diff()
        elif return_type == 'pct':
            ret = series.pct_change()
        else:
            raise ValueError("return_type must be 'log' or 'pct'")
        # Adjust returns by liquidity
        adj_ret = ret / sqrt_vol
        # Rolling volatility of adjusted returns
        vol_series = adj_ret.rolling(window).std()
        if annualize_factor is not None:
            vol_series = vol_series * np.sqrt(annualize_factor)
        returns_data[col] = ret.values
        adj_returns_data[col] = adj_ret.values
        vol_data[col] = vol_series.values
    returns_df = pd.DataFrame(returns_data, index=dt_index)
    adj_returns_df = pd.DataFrame(adj_returns_data, index=dt_index)
    volatility_df = pd.DataFrame(vol_data, index=dt_index)
    return LiquidityAdjustedVolatilityResult(volatility=volatility_df, returns=returns_df, adjusted_returns=adj_returns_df)


@dataclass
class RollingBetaResult:
    """Result container for rolling beta analysis.

    Attributes
    ----------
    beta : pandas.Series
        Rolling beta of the target series relative to the benchmark.
    intercept : pandas.Series
        Rolling intercept of the regression.
    r_squared : pandas.Series
        Rolling R^2 values of the regression.
    returns : pandas.DataFrame
        DataFrame of target and benchmark returns used in the rolling
        regression.
    """
    beta: pd.Series
    intercept: pd.Series
    r_squared: pd.Series
    returns: pd.DataFrame


def rolling_beta(
    date: Union[str, pd.Series, Iterable],
    df: pd.DataFrame,
    *,
    target: str,
    benchmark: str,
    window: int = 52,
    return_type: str = 'log',
    min_periods: Optional[int] = None,
) -> RollingBetaResult:
    """Compute the rolling beta of ``target`` relative to ``benchmark``.

    Rolling beta is estimated using linear regression of the target
    returns on the benchmark returns within a moving window.  This
    function calculates beta, intercept and R^2 for each window, using
    the specified return type.  Missing data are automatically
    forward‑ and backward‑filled within each window.

    Parameters
    ----------
    date : str or iterable
        Date column in ``df`` or iterable of dates.
    df : pandas.DataFrame
        DataFrame containing the price series for the target and
        benchmark.  Numeric columns are required.
    target : str
        Column name of the target series.
    benchmark : str
        Column name of the benchmark series.
    window : int, default 52
        Size of the rolling window (number of observations).
    return_type : {'log','pct'}, default 'log'
        Return calculation type: log returns or percentage change.
    min_periods : int or None, default None
        Minimum number of observations in each window required to
        compute regression statistics.  If ``None``, defaults to
        ``window``.

    Returns
    -------
    RollingBetaResult
        Object containing rolling beta, intercept, R^2 and returns.

    Notes
    -----
    The regression is computed analytically using the formulas for
    covariance and variance: ``beta = cov(target,benchmark)/var(benchmark)``.
    Intercept and R^2 are derived from the means and variances of the
    returns within each window.
    """
    if target not in df.columns or benchmark not in df.columns:
        raise KeyError("Target or benchmark column not found in DataFrame")
    # Determine date index
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        dt_index = pd.to_datetime(df[date])
    else:
        dt_index = pd.to_datetime(pd.Series(date))
    # Compute returns
    series_target = pd.to_numeric(df[target], errors='coerce').ffill().bfill()
    series_benchmark = pd.to_numeric(df[benchmark], errors='coerce').ffill().bfill()
    if return_type == 'log':
        ret_target = np.log(series_target).diff()
        ret_benchmark = np.log(series_benchmark).diff()
    elif return_type == 'pct':
        ret_target = series_target.pct_change()
        ret_benchmark = series_benchmark.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'pct'")
    returns_df = pd.DataFrame({target: ret_target, benchmark: ret_benchmark}, index=dt_index)
    # Rolling calculations
    m = window if min_periods is None else min_periods
    # Rolling means
    mean_target = ret_target.rolling(window, min_periods=m).mean()
    mean_benchmark = ret_benchmark.rolling(window, min_periods=m).mean()
    # Deviations
    dev_target = ret_target - mean_target
    dev_benchmark = ret_benchmark - mean_benchmark
    # Rolling covariance and variance
    cov = (dev_target * dev_benchmark).rolling(window, min_periods=m).mean()
    var_benchmark = (dev_benchmark ** 2).rolling(window, min_periods=m).mean()
    # Beta
    beta = cov / var_benchmark
    # Intercept
    intercept = mean_target - beta * mean_benchmark
    # R^2
    var_target = (dev_target ** 2).rolling(window, min_periods=m).mean()
    r_squared = (cov ** 2) / (var_target * var_benchmark)
    return RollingBetaResult(beta=beta, intercept=intercept, r_squared=r_squared, returns=returns_df)
