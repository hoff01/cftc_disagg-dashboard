"""
Advanced statistical functions for time series.

This module includes utilities for estimating long‑memory and
scaling properties (Hurst exponent, detrended fluctuation analysis)
and for evaluating risk‑adjusted performance (rolling Sharpe ratio).

The functions here are designed for commodity traders and analysts
seeking deeper insights into the stochastic behaviour of price
series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling statistics and trend tests
# ---------------------------------------------------------------------------

@dataclass
class RollingZScoreResult:
    """Container for rolling z‑score calculations.

    Attributes
    ----------
    zscores : pandas.DataFrame
        DataFrame of rolling z‑score values for each series.  Indices
        correspond to the date column of the input DataFrame.
    flags : pandas.DataFrame
        Boolean DataFrame indicating where the absolute z‑score
        exceeds the specified threshold (potential anomalies).
    """

    zscores: pd.DataFrame
    flags: pd.DataFrame


def rolling_zscore(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    window: int = 20,
    threshold: float = 3.0,
    columns: Optional[List[str]] = None,
) -> RollingZScoreResult:
    """Compute rolling z‑scores to detect outliers.

    For each numeric column, this function calculates the rolling
    mean and standard deviation over a window of length ``window`` and
    normalises the series by these estimates.  Points where the
    absolute z‑score exceeds ``threshold`` are flagged as potential
    anomalies.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates for alignment.  Used
        to index the output DataFrames.
    df : pandas.DataFrame
        DataFrame with the data.
    window : int, default 20
        Size of the rolling window used to compute mean and standard
        deviation.
    threshold : float, default 3.0
        Absolute z‑score threshold for flagging anomalies.
    columns : list of str or None, default None
        Specific columns to analyse.  If ``None``, all numeric
        columns except the date column are used.

    Returns
    -------
    RollingZScoreResult
        Dataclass containing z‑scores and anomaly flags.
    """
    # Determine date index
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        dt_index = pd.to_datetime(df[date])
    else:
        dt_index = pd.to_datetime(date)
    # Select columns
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        raise ValueError("No numeric columns selected for rolling z‑score calculation")
    zscore_data = {}
    flag_data = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        roll_mean = series.rolling(window=window, min_periods=1).mean()
        roll_std = series.rolling(window=window, min_periods=1).std(ddof=0)
        z = (series - roll_mean) / roll_std.replace(0, np.nan)
        zscore_data[col] = z.values
        flag_data[col] = (z.abs() > threshold).astype(bool).values
    zscores_df = pd.DataFrame(zscore_data, index=dt_index, columns=columns)
    flags_df = pd.DataFrame(flag_data, index=dt_index, columns=columns)
    return RollingZScoreResult(zscores=zscores_df, flags=flags_df)


@dataclass
class MannKendallResult:
    """Container for the Mann–Kendall trend test results.

    Attributes
    ----------
    S : float
        Mann–Kendall test statistic.
    Var : float
        Variance of the test statistic.
    z : float
        Normalised test statistic (z‑score).
    p : float
        Two‑sided p‑value for the trend.
    trend : str
        Interpretation of the trend: 'increasing', 'decreasing' or
        'no trend'.
    """
    S: float
    Var: float
    z: float
    p: float
    trend: str


def mann_kendall_test(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> dict:
    """Perform the Mann–Kendall trend test on one or more series.

    The Mann–Kendall test assesses whether a monotonic upward or
    downward trend exists in a time series without assuming any
    particular distribution.  For each series, this function
    computes the S statistic, its variance, the normalised z‑score,
    two‑sided p‑value and classifies the trend.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates for alignment (unused
        in calculations).
    df : pandas.DataFrame
        DataFrame containing the series to test.
    columns : list of str or None, default None
        Specific columns to test.  If ``None``, all numeric columns
        except the date column are used.
    alpha : float, default 0.05
        Significance level used to classify the trend.

    Returns
    -------
    dict
        Mapping from column name to :class:`MannKendallResult`.
    """
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    results = {}
    for col in columns:
        x = pd.to_numeric(df[col], errors='coerce').dropna().values
        n = len(x)
        if n < 2:
            results[col] = MannKendallResult(S=np.nan, Var=np.nan, z=np.nan, p=np.nan, trend='insufficient data')
            continue
        # Compute S statistic
        S = 0
        for k in range(n - 1):
            S += np.sum(np.sign(x[k+1:] - x[k]))
        # Ties count for variance
        # Identify unique values and their counts
        unique, counts = np.unique(x, return_counts=True)
        # Compute variance with ties adjustment
        Var_S = (n * (n - 1) * (2 * n + 5)) / 18.0
        tie_sum = 0.0
        for c in counts:
            if c > 1:
                tie_sum += c * (c - 1) * (2 * c + 5)
        Var_S -= tie_sum / 18.0
        # Compute z statistic
        if S > 0:
            z = (S - 1) / np.sqrt(Var_S)
        elif S < 0:
            z = (S + 1) / np.sqrt(Var_S)
        else:
            z = 0.0
        # Two‑sided p‑value from standard normal
        p_val = 2 * (1.0 - 0.5 * (1 + np.math.erf(abs(z) / np.sqrt(2))))
        # Determine trend
        if p_val < alpha:
            trend = 'increasing' if z > 0 else 'decreasing'
        else:
            trend = 'no trend'
        results[col] = MannKendallResult(S=float(S), Var=float(Var_S), z=float(z), p=float(p_val), trend=trend)
    return results


@dataclass
class BollingerBandsResult:
    """Container for Bollinger Bands computation.

    Attributes
    ----------
    bands : Dict[str, pandas.DataFrame]
        Mapping from each series name to a DataFrame with columns
        'mean', 'upper', and 'lower', indexed by the date column.
    parameters : dict
        Dictionary of the parameters used (window and n_std).
    """
    bands: dict
    parameters: dict


def bollinger_bands(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    window: int = 20,
    n_std: float = 2.0,
) -> BollingerBandsResult:
    """Compute Bollinger Bands for one or more series.

    Bollinger Bands consist of a rolling mean and bands placed a
    specified number of standard deviations above and below the mean.
    They are commonly used to identify overbought or oversold
    conditions and to gauge volatility.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates.  Used as the index of
        the returned DataFrames.
    df : pandas.DataFrame
        DataFrame with the data.
    columns : list of str or None, default None
        Specific columns to compute bands for.  If ``None``, all
        numeric columns except the date column are used.
    window : int, default 20
        Size of the rolling window for the moving average and
        standard deviation.
    n_std : float, default 2.0
        Number of standard deviations to define the upper and lower
        bands.

    Returns
    -------
    BollingerBandsResult
        Dataclass containing the bands for each series and the
        parameters used.
    """
    # Determine date index
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        dt_index = pd.to_datetime(df[date])
    else:
        dt_index = pd.to_datetime(date)
    # Select columns
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        raise ValueError("No numeric columns selected for Bollinger Bands calculation")
    bands_dict: dict = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std(ddof=0)
        upper = mean + n_std * std
        lower = mean - n_std * std
        bands_dict[col] = pd.DataFrame({'mean': mean, 'upper': upper, 'lower': lower}, index=dt_index)
    return BollingerBandsResult(bands=bands_dict, parameters={'window': window, 'n_std': n_std})


def hurst_exponent(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    return_type: str = 'log',
    max_lag: int = 20,
) -> pd.Series:
    """Estimate the Hurst exponent for one or more series.

    The Hurst exponent quantifies the long‑term memory of a time
    series.  Values greater than 0.5 indicate persistence (trend
    reinforcement), values less than 0.5 indicate anti‑persistence
    (mean reversion) and a value of 0.5 corresponds to a random walk.
    A simple R/S–based estimator is used here.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the series.
    columns : list of str or None, default None
        Columns to analyse.  If ``None``, all numeric columns except
        the date column are used.
    return_type : {'log','pct'}, default 'log'
        Type of transformation applied before Hurst calculation: log
        returns or percentage changes.  When returns are used, the
        cumulative sum of returns is analysed.
    max_lag : int, default 20
        Maximum lag to use in the rescaled range method.  Smaller
        values yield quicker estimates at the cost of precision.

    Returns
    -------
    pandas.Series
        Hurst exponent for each column.
    """
    # Identify columns
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    hurst_vals = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        # Convert to returns if requested
        if return_type == 'log':
            x = np.log(series).diff().dropna()
        elif return_type == 'pct':
            x = series.pct_change().dropna()
        else:
            raise ValueError("return_type must be 'log' or 'pct'")
        # Cumulative sum of demeaned returns
        y = np.cumsum(x - x.mean())
        # Compute R/S for different lags
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            diff_series = y[lag:] - y[:-lag]
            tau.append(np.sqrt(np.std(diff_series)))
        # Fit linear regression in log–log space
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        hurst_vals[col] = slope * 2.0
    return pd.Series(hurst_vals)


def sample_entropy(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    m: int = 2,
    r: float = 0.2,
    normalize: bool = True,
) -> pd.Series:
    """Compute the sample entropy of one or more series.

    Sample entropy (SampEn) is a measure of complexity that quantifies
    the regularity and unpredictability of fluctuations in a time
    series.  Lower values correspond to more self‑similar (predictable)
    data, while higher values indicate greater irregularity.  The
    implementation here follows the definition by Richman and Moorman
    (2000) and supports scaling the tolerance ``r`` by the standard
    deviation of the series.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to analyse.  If ``None``, all numeric columns except
        the date column are used.
    m : int, default 2
        Length of sequences to compare.  Typical values are 2 or 3.
    r : float, default 0.2
        Tolerance for accepting matches.  If ``normalize`` is True,
        this value is multiplied by the standard deviation of the
        series; otherwise it is used as an absolute threshold.
    normalize : bool, default True
        Whether to scale ``r`` by the standard deviation of the
        series.  This makes the tolerance relative to the variability
        of the data.

    Returns
    -------
    pandas.Series
        Sample entropy value for each column.

    Notes
    -----
    Sample entropy is defined as the negative natural logarithm of the
    ratio of the number of matching subsequences of length ``m+1`` to
    those of length ``m``.  A match occurs when the maximum absolute
    difference between two subsequences does not exceed ``r``.
    """
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    sampen_vals = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna().values
        N = len(series)
        if N <= m + 1:
            sampen_vals[col] = np.nan
            continue
        # Normalise tolerance if requested
        tol = r * np.std(series) if normalize else r
        # Build embedding vectors
        # Sequences of length m and m+1
        x_m = np.array([series[i : i + m] for i in range(N - m + 1)])
        x_m1 = np.array([series[i : i + m + 1] for i in range(N - m)])
        # Compute Chebyshev distances between all pairs (except self) for m and m+1
        def _count_matches(X: np.ndarray, tol: float) -> int:
            count = 0
            K = X.shape[0]
            for i in range(K - 1):
                # Difference with subsequent vectors
                diff = np.max(np.abs(X[i + 1:] - X[i]), axis=1)
                count += np.sum(diff <= tol)
            return count
        B = _count_matches(x_m, tol)
        A = _count_matches(x_m1, tol)
        # Avoid division by zero
        if B == 0 or A == 0:
            sampen_vals[col] = np.nan
        else:
            sampen_vals[col] = -np.log(A / B)
    return pd.Series(sampen_vals)


def higuchi_fractal_dimension(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    kmax: int = 10,
) -> pd.Series:
    """Estimate the fractal dimension of a time series using Higuchi's method.

    Higuchi's algorithm estimates the fractal dimension by measuring
    the length of the curve at different scales and examining how it
    scales with the scale factor.  This method captures the geometric
    complexity of the series and is applicable to financial time
    series that exhibit fractal‑like properties.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to analyse.  If ``None``, all numeric columns except
        the date column are used.
    kmax : int, default 10
        Maximum number of intervals ``k`` to consider.  Higher values
        may yield more precise estimates but increase computation time.

    Returns
    -------
    pandas.Series
        Higuchi fractal dimension for each column.

    Notes
    -----
    The fractal dimension estimated by Higuchi's method lies between
    1 and 2 for most time series.  Values closer to 1 indicate a
    smoother curve, while values closer to 2 suggest a more complex
    (rough) series.
    """
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    fd_vals = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna().values
        N = len(series)
        if N < 2:
            fd_vals[col] = np.nan
            continue
        L = []
        k_values = range(1, kmax + 1)
        for k in k_values:
            Lk = []
            for m in range(k):
                # Build the time series with stride k starting at m
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                # Sum absolute differences
                diffs = np.abs(np.diff(series[idx]))
                length = np.sum(diffs)
                # Normalise by k to the appropriate factor
                # Multiply by (N - 1) / (floor((N - m - 1) / k) * k) to adjust for number of segments
                n_segments = len(idx) - 1
                if n_segments > 0:
                    length = (length * (N - 1) / (n_segments * k))
                    Lk.append(length)
            # Compute average length for this k
            if Lk:
                L.append(np.mean(Lk))
        # Fit line in log-log space: log(L(k)) vs log(1/k)
        if not L:
            fd_vals[col] = np.nan
            continue
        log_L = np.log(L)
        log_inv_k = np.log(1.0 / np.array(list(k_values)[: len(L)]))
        # Linear regression: slope = D, fractal dimension estimate
        slope, _ = np.polyfit(log_inv_k, log_L, 1)
        fd_vals[col] = slope
    return pd.Series(fd_vals)


def dfa_exponent(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    scales: Optional[List[int]] = None,
    return_type: str = 'log',
) -> pd.Series:
    """Estimate the scaling exponent via detrended fluctuation analysis.

    DFA is a method for detecting long‑range correlations in noisy
    non‑stationary time series.  It computes the RMS fluctuation of a
    detrended cumulative series at various scales and fits a line in
    log–log space.  The slope of this line is the DFA exponent.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to analyse.  If ``None``, all numeric columns except
        the date column are used.
    scales : list of int or None, default None
        Window sizes to use for DFA.  If ``None``, a default set of
        exponentially increasing scales is used.
    return_type : {'log','pct'}, default 'log'
        Type of return transformation before analysis.

    Returns
    -------
    pandas.Series
        DFA exponent for each column.
    """
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if scales is None:
        scales = [4, 8, 16, 32, 64]
    dfa_vals = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        if return_type == 'log':
            x = np.log(series).diff().dropna()
        elif return_type == 'pct':
            x = series.pct_change().dropna()
        else:
            raise ValueError("return_type must be 'log' or 'pct'")
        # Integrate demeaned returns
        y = np.cumsum(x - x.mean())
        n = len(y)
        F = []
        for s in scales:
            if s >= n:
                continue
            # Number of segments
            m = n // s
            rms = []
            for i in range(m):
                segment = y[i * s:(i + 1) * s]
                # Fit linear trend
                t = np.arange(s)
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)
                rms.append(np.sqrt(np.mean((segment - trend) ** 2)))
            if rms:
                F.append((s, np.mean(rms)))
        if not F:
            dfa_vals[col] = np.nan
            continue
        s_vals, f_vals = zip(*F)
        log_s = np.log(s_vals)
        log_f = np.log(f_vals)
        slope, _ = np.polyfit(log_s, log_f, 1)
        dfa_vals[col] = slope
    return pd.Series(dfa_vals)


@dataclass
class RollingSharpeResult:
    """Container for rolling Sharpe ratio analysis.

    Attributes
    ----------
    sharpe : pandas.Series
        Rolling Sharpe ratio of the target series.
    returns : pandas.Series
        Returns used in the calculation.
    """
    sharpe: pd.Series
    returns: pd.Series


def rolling_sharpe_ratio(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    column: str,
    window: int = 52,
    risk_free_rate: float = 0.0,
    return_type: str = 'log',
    annualize: bool = True,
    periods_per_year: int = 52,
) -> RollingSharpeResult:
    """Compute the rolling Sharpe ratio for a single series.

    The Sharpe ratio measures the risk‑adjusted return by dividing the
    excess return (over a risk‑free rate) by the standard deviation of
    returns.  This function computes the Sharpe ratio within a moving
    window.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    column : str
        Name of the price series to analyse.
    window : int, default 52
        Window size for computing the rolling mean and standard
        deviation of returns.
    risk_free_rate : float, default 0.0
        Constant risk‑free rate per period.  Subtracted from returns.
    return_type : {'log','pct'}, default 'log'
        Type of returns to compute.
    annualize : bool, default True
        Whether to annualise the Sharpe ratio by multiplying by
        ``sqrt(periods_per_year)``.
    periods_per_year : int, default 52
        Number of periods per year (e.g. 252 for daily, 52 for weekly).

    Returns
    -------
    RollingSharpeResult
        Dataclass containing the rolling Sharpe ratio and the returns.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    series = pd.to_numeric(df[column], errors='coerce').ffill().bfill()
    if return_type == 'log':
        ret = np.log(series).diff()
    elif return_type == 'pct':
        ret = series.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'pct'")
    excess = ret - risk_free_rate
    mean_ret = excess.rolling(window).mean()
    std_ret = excess.rolling(window).std()
    sharpe = mean_ret / std_ret
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    return RollingSharpeResult(sharpe=sharpe, returns=ret)

# ---------------------------------------------------------------------------
# Transformations and decomposition
# ---------------------------------------------------------------------------

@dataclass
class BoxCoxTransformResult:
    """Container for Box–Cox transformation results.

    Attributes
    ----------
    transformed : pandas.DataFrame
        Transformed data.  Each specified column has been Box–Cox
        transformed and shifted to ensure positivity.
    lambdas : dict
        Mapping from column name to the estimated Box–Cox lambda
        parameter for that series.
    shifts : dict
        Mapping from column name to the additive shift applied to
        ensure strictly positive values.  The original series can be
        recovered by subtracting this shift after applying the
        inverse Box–Cox transformation.
    """
    transformed: pd.DataFrame
    lambdas: dict
    shifts: dict


def box_cox_transform(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    lmbdas: Optional[dict] = None,
    inverse: bool = False,
) -> BoxCoxTransformResult:
    """Apply the Box–Cox transformation (or its inverse) to series.

    The Box–Cox transformation is often used to stabilize variance and
    make data more Gaussian.  It requires input values to be strictly
    positive, so this function automatically shifts each series by a
    constant equal to ``1 - min(series)`` if necessary.  When
    ``inverse`` is True, the inverse Box–Cox transformation is
    applied using provided lambda parameters.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to transform.  If ``None``, all numeric columns
        except the date column are used.
    lmbdas : dict or None, default None
        Dictionary of lambda parameters for the inverse
        transformation.  Required when ``inverse=True``.
    inverse : bool, default False
        If True, apply the inverse Box–Cox transformation.  In this
        case, ``lmbdas`` must be provided and the function returns
        values in the original scale.

    Returns
    -------
    BoxCoxTransformResult
        Dataclass with the transformed data, estimated lambdas and
        shifts.  When ``inverse=True``, the lambdas and shifts are
        those supplied in ``lmbdas`` and the values correspond to
        the original scale.

    Raises
    ------
    ImportError
        If ``scipy`` is not installed.
    ValueError
        If lambda parameters are missing when performing the
        inverse transformation.
    """
    try:
        from scipy.stats import boxcox
        from scipy.special import boxcox as boxcox_forward  # alias
        from scipy.special import inv_boxcox as inv_boxcox
    except Exception as e:
        raise ImportError(
            "scipy is required for Box–Cox transformations. "
            "Please install scipy to use this function."
        ) from e
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        raise ValueError("No numeric columns selected for Box–Cox transformation")
    transformed_data = {}
    lambda_params = {} if not inverse else (lmbdas or {})
    shifts = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').astype(float)
        if not inverse:
            # Shift to ensure positivity
            min_val = np.nanmin(series)
            shift = 1.0 - min_val if min_val <= 0 else 0.0
            shifts[col] = shift
            positive_series = series + shift
            # Estimate lambda via MLE
            with np.errstate(invalid='ignore'):
                transformed, lam = boxcox(positive_series.dropna())
            # Reindex to original length
            full_transformed = pd.Series(index=series.index, dtype=float)
            full_transformed.loc[positive_series.dropna().index] = transformed
            # Store
            transformed_data[col] = full_transformed
            lambda_params[col] = lam
        else:
            if lmbdas is None or col not in lmbdas:
                raise ValueError(
                    f"Missing lambda parameter for column '{col}' in inverse Box–Cox transformation"
                )
            lam = lmbdas[col]
            # Determine shift; if none provided, assume zero
            shift = 0.0
            if col in df.columns and not df[col].isnull().all():
                # Attempt to infer shift by assuming original min > 0
                min_val = np.nanmin(df[col])
                shift = 1.0 - min_val if min_val <= 0 else 0.0
            shifts[col] = shift
            transformed_series = pd.to_numeric(df[col], errors='coerce').astype(float)
            # Apply inverse transform
            inv_values = inv_boxcox(transformed_series.dropna(), lam)
            original = inv_values - shift
            full_orig = pd.Series(index=series.index, dtype=float)
            full_orig.loc[transformed_series.dropna().index] = original
            transformed_data[col] = full_orig
    return BoxCoxTransformResult(
        transformed=pd.DataFrame(transformed_data, index=df.index),
        lambdas=lambda_params,
        shifts=shifts,
    )


@dataclass
class SeasonalAdjustmentResult:
    """Container for seasonal adjustment results.

    Attributes
    ----------
    adjusted : pandas.DataFrame
        Seasonally adjusted series.  Each specified column has had its
        seasonal component removed.
    seasonal : pandas.DataFrame
        The extracted seasonal component for each series.
    seasonal_period : int
        The seasonal period used in the decomposition.
    """
    adjusted: pd.DataFrame
    seasonal: pd.DataFrame
    seasonal_period: int


def seasonal_adjust(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    seasonal_period: Optional[int] = None,
) -> SeasonalAdjustmentResult:
    """Remove seasonal components from time series via STL.

    The function decomposes each series using STL and subtracts the
    seasonal component, returning the seasonally adjusted series and
    the extracted seasonal component.  If ``seasonal_period`` is not
    specified, it is inferred from the data frequency where possible.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to adjust.  If ``None``, all numeric columns except
        the date column are used.
    seasonal_period : int or None, default None
        Length of the seasonal cycle.  If ``None``, a period is
        inferred from the data frequency (e.g. 12 for monthly, 52 for
        weekly).  When no reasonable inference can be made, a period
        of 2 is used.

    Returns
    -------
    SeasonalAdjustmentResult
        Dataclass containing the seasonally adjusted series, the
        seasonal component and the period used.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as e:
        raise ImportError(
            "statsmodels is required for seasonal adjustment. "
            "Please install statsmodels to use this function."
        ) from e
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        raise ValueError("No numeric columns selected for seasonal adjustment")
    # Infer seasonal period if none provided
    sp = seasonal_period
    if sp is None:
        if isinstance(date, str) and date in df.columns:
            dt = pd.to_datetime(df[date])
        else:
            dt = pd.to_datetime(date)
        try:
            freq = pd.infer_freq(dt)
        except Exception:
            freq = None
        if freq is not None:
            if freq.startswith('M'):
                sp = 12
            elif freq.startswith('Q'):
                sp = 4
            elif freq.startswith('W'):
                sp = 52
            elif freq.startswith('A') or freq.startswith('Y'):
                sp = 1
            else:
                sp = 2
        else:
            sp = 2
    adjusted_data = {}
    seasonal_data = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        if series.dropna().shape[0] < 4:
            adjusted_data[col] = pd.Series(np.nan, index=df.index)
            seasonal_data[col] = pd.Series(np.nan, index=df.index)
            continue
        stl = STL(series, period=sp, robust=True)
        res = stl.fit()
        adjusted_data[col] = series - res.seasonal
        seasonal_data[col] = res.seasonal
    return SeasonalAdjustmentResult(
        adjusted=pd.DataFrame(adjusted_data, index=df.index),
        seasonal=pd.DataFrame(seasonal_data, index=df.index),
        seasonal_period=sp,
    )


# ---------------------------------------------------------------------------
# Stationarity and decomposition utilities
# ---------------------------------------------------------------------------

@dataclass
class StationarityTestResult:
    """Container for combined stationarity test results.

    Attributes
    ----------
    adf_stat : float
        Test statistic from the augmented Dickey–Fuller (ADF) test.
    adf_p : float
        p‑value from the ADF test.  Small values suggest rejecting the
        null hypothesis of a unit root (non‑stationarity).
    adf_crit : dict
        Dictionary of critical values for the ADF test.
    kpss_stat : float
        Test statistic from the KPSS test.  Large values suggest
        rejecting the null hypothesis of stationarity.
    kpss_p : float
        p‑value from the KPSS test.
    kpss_crit : dict
        Dictionary of critical values for the KPSS test.
    stationary_adf : bool
        True if the ADF test rejects the null at the given
        significance level.
    stationary_kpss : bool
        True if the KPSS test fails to reject the null (i.e. the
        series appears stationary).
    """
    adf_stat: float
    adf_p: float
    adf_crit: dict
    kpss_stat: float
    kpss_p: float
    kpss_crit: dict
    stationary_adf: bool
    stationary_kpss: bool


def stationarity_tests(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> dict:
    """Run a battery of stationarity tests on one or more series.

    This utility performs both the Augmented Dickey–Fuller (ADF) test
    and the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test on each
    specified column.  The ADF test has a null hypothesis that the
    series possesses a unit root (is non‑stationary), while the KPSS
    test has a null hypothesis of stationarity.  By examining both
    tests together, one can better classify a series as stationary or
    non‑stationary.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Specific columns to test.  If ``None``, all numeric columns
        except the date column are used.
    alpha : float, default 0.05
        Significance level for interpreting the tests.  The ADF
        null hypothesis is rejected when the p‑value is below
        ``alpha``.  The KPSS null hypothesis is rejected when the
        p‑value is below ``alpha``.

    Returns
    -------
    dict
        Mapping from column name to :class:`StationarityTestResult`.

    Notes
    -----
    The KPSS implementation in ``statsmodels`` may issue warnings
    when the test statistic lies outside the range of tabulated
    critical values.  This function suppresses those warnings
    internally.  If ``statsmodels`` is not installed, an
    :class:`ImportError` is raised.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception as e:
        raise ImportError(
            "statsmodels is required for stationarity tests. "
            "Please install statsmodels to use this function."
        ) from e
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    results: dict = {}
    for col in columns:
        x = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        if x.empty:
            results[col] = StationarityTestResult(
                adf_stat=np.nan,
                adf_p=np.nan,
                adf_crit={},
                kpss_stat=np.nan,
                kpss_p=np.nan,
                kpss_crit={},
                stationary_adf=False,
                stationary_kpss=False,
            )
            continue
        # ADF test
        try:
            adf_res = adfuller(x, autolag='AIC')
            adf_stat = float(adf_res[0])
            adf_p = float(adf_res[1])
            adf_crit = adf_res[4]
            stationary_adf = adf_p < alpha
        except Exception:
            adf_stat = np.nan
            adf_p = np.nan
            adf_crit = {}
            stationary_adf = False
        # KPSS test (level stationarity)
        try:
            # Some versions of statsmodels raise warnings when the
            # statistic is outside the range; suppress them.
            with np.errstate(invalid='ignore'):
                kpss_res = kpss(x, regression='c', nlags='auto')
            kpss_stat = float(kpss_res[0])
            kpss_p = float(kpss_res[1])
            kpss_crit = kpss_res[3]
            stationary_kpss = kpss_p >= alpha
        except Exception:
            kpss_stat = np.nan
            kpss_p = np.nan
            kpss_crit = {}
            stationary_kpss = False
        results[col] = StationarityTestResult(
            adf_stat=adf_stat,
            adf_p=adf_p,
            adf_crit=adf_crit,
            kpss_stat=kpss_stat,
            kpss_p=kpss_p,
            kpss_crit=kpss_crit,
            stationary_adf=stationary_adf,
            stationary_kpss=stationary_kpss,
        )
    return results


@dataclass
class TrendSeasonalityStrengthResult:
    """Container for trend and seasonal strength measures.

    Attributes
    ----------
    strengths : pandas.DataFrame
        DataFrame indexed by series names with two columns:
        ``trend_strength`` and ``seasonal_strength``.  Values lie
        between 0 and 1, where higher numbers indicate a stronger
        trend or seasonal component relative to the remainder.
    seasonal_period : int
        The seasonal period used in the calculation.
    """
    strengths: pd.DataFrame
    seasonal_period: int


def trend_seasonality_strength(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    seasonal_period: Optional[int] = None,
) -> TrendSeasonalityStrengthResult:
    """Measure the strength of trend and seasonality in time series.

    This function decomposes each series using STL (seasonal
    decomposition via Loess) and computes the proportion of
    variability explained by the trend and seasonal components.  The
    strength of trend is defined as ``max(0, 1 - var(remainder) / var(trend + remainder))``
    and the strength of seasonality as ``max(0, 1 - var(remainder) / var(seasonal + remainder))``
    (see Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*).

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the series.
    columns : list of str or None, default None
        Columns to analyse.  If ``None``, all numeric columns except
        the date column are used.
    seasonal_period : int or None, default None
        Length of the seasonal cycle.  If ``None``, a period is
        inferred from the data frequency when possible (e.g. 12 for
        monthly, 52 for weekly).  When the frequency is irregular or
        cannot be inferred, no seasonal component is included and
        only trend strength is computed.

    Returns
    -------
    TrendSeasonalityStrengthResult
        Dataclass containing the strength measures and the seasonal
        period used.

    Raises
    ------
    ImportError
        If the ``statsmodels`` package is not installed.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as e:
        raise ImportError(
            "statsmodels is required for STL decomposition. "
            "Please install statsmodels to use this function."
        ) from e
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    strengths = {}
    # Infer seasonal period if needed
    if seasonal_period is None:
        # Use date frequency to guess a seasonal period
        if isinstance(date, str) and date in df.columns:
            dt = pd.to_datetime(df[date])
        else:
            dt = pd.to_datetime(date)
        try:
            freq = pd.infer_freq(dt)
        except Exception:
            freq = None
        if freq is not None:
            if freq.startswith('M'):
                seasonal_period = 12
            elif freq.startswith('Q'):
                seasonal_period = 4
            elif freq.startswith('W'):
                seasonal_period = 52
            elif freq.startswith('A') or freq.startswith('Y'):
                seasonal_period = 1
            else:
                seasonal_period = None
        else:
            seasonal_period = None
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        # STL requires a minimum number of observations
        if series.dropna().shape[0] < 4:
            strengths[col] = {'trend_strength': np.nan, 'seasonal_strength': np.nan}
            continue
        if seasonal_period is not None and seasonal_period > 1:
            stl = STL(series, period=seasonal_period, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
            var_r = np.nanvar(resid)
            # Strength of trend
            var_trend = np.nanvar(trend + resid)
            trend_strength = 0.0
            seasonal_strength = 0.0
            if var_trend > 0 and var_r >= 0:
                trend_strength = max(0.0, 1.0 - var_r / var_trend)
            # Strength of seasonality
            var_seas = np.nanvar(seasonal + resid)
            if var_seas > 0 and var_r >= 0:
                seasonal_strength = max(0.0, 1.0 - var_r / var_seas)
            strengths[col] = {
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength,
            }
        else:
            # When no seasonal component, estimate only trend strength
            stl = STL(series, period=2, robust=True)
            res = stl.fit()
            trend = res.trend
            resid = res.resid
            var_r = np.nanvar(resid)
            var_trend = np.nanvar(trend + resid)
            trend_strength = 0.0
            if var_trend > 0 and var_r >= 0:
                trend_strength = max(0.0, 1.0 - var_r / var_trend)
            strengths[col] = {
                'trend_strength': trend_strength,
                'seasonal_strength': np.nan,
            }
    strengths_df = pd.DataFrame.from_dict(strengths, orient='index')
    return TrendSeasonalityStrengthResult(strengths=strengths_df, seasonal_period=seasonal_period or 1)
