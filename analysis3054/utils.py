"""
Utility functions for data manipulation and merging.

This module provides several helpers for combining and transforming
pandas DataFrames.  These utilities are designed to support typical
workflows encountered in commodity analysis, where disparate data
sources must be merged on a variety of keys or aligned in time.  In
addition to straightforward column and row merges, more advanced
matching strategies are implemented, such as nearest‑key joins and
coalescing overlaps.  All functions expect pandas DataFrames and
preserve the index and data types wherever possible.

Included are:

* :func:`conditional_column_merge` – Merge one or more columns from an
  auxiliary DataFrame into a primary DataFrame based on matching key
  values.  Supports matching on multiple values by splitting a
  delimiter‑separated key column.

* :func:`conditional_row_merge` – Append rows from an auxiliary
  DataFrame to a primary DataFrame when a specified column contains
  values found in a list of keys.  Useful for selectively augmenting
  a dataset with additional observations.

* :func:`nearest_key_merge` – Join two DataFrames by matching on the
  closest value of a numeric key within a specified tolerance.  This
  is helpful when exact matches are rare or when values represent
  continuous measurements (e.g. timestamps or price levels).

* :func:`coalesce_merge` – Merge two DataFrames on a set of keys and
  coalesce overlapping columns, preferring non‑null values from the
  left DataFrame.  This can be used to combine cleaned and raw
  datasets while retaining the most reliable data.

* :func:`rolling_fill` – Fill missing values in numeric columns using
  a rolling window statistic (mean, median, etc.).  This provides a
  simple yet effective way to impute gaps without distorting trends.

* :func:`add_time_features` – Expand a datetime column into calendar
  features (year, month, week, etc.) for modeling or visualization.

* :func:`resample_time_series` – Resample a DataFrame by time,
  applying aggregation and optional forward/back fills.

* :func:`winsorize_columns` – Clip numeric columns to quantile
  bounds to reduce the influence of extreme outliers.

* :func:`add_lag_features` – Create lagged versions of numeric
  columns for supervised forecasting models.

* :func:`scale_columns` – Standardize or normalize numeric columns
  with optional centering and scaling.

* :func:`rolling_window_features` – Compute rolling statistics
  (mean, std, min, max, median) for numeric columns.

Each function is accompanied by a docstring explaining its usage and
parameters.  See the examples in the package documentation for
guidance on how to apply these utilities in your analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from getpass import getpass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import os

import numpy as np
import pandas as pd


def conditional_column_merge(
    df: pd.DataFrame,
    other: pd.DataFrame,
    *,
    df_key: Union[str, Sequence[str]],
    other_key: Union[str, Sequence[str]],
    columns: Union[str, List[str]],
    delimiter: str = '|',
    multiple: bool = False,
    suffix: str = '_y',
) -> pd.DataFrame:
    """Merge selected columns from ``other`` into ``df`` based on key values.

    This function performs a left join of ``df`` with ``other``, adding
    one or more columns from ``other`` when values in the key column(s)
    match.  When ``multiple`` is ``True``, the key column in ``df`` can
    contain delimiter‑separated lists of values; the merge will occur if
    **any** of the values matches the corresponding key in ``other``.

    Parameters
    ----------
    df : pandas.DataFrame
        Primary DataFrame to which columns will be added.
    other : pandas.DataFrame
        Auxiliary DataFrame containing the columns to merge.
    df_key : str or list of str
        Column(s) in ``df`` on which to match.  If ``multiple`` is
        ``True`` and a single string is provided, values in this column
        may be delimiter‑separated lists of keys.
    other_key : str or list of str
        Column(s) in ``other`` on which to match.  Must align with
        ``df_key``.  Composite keys are supported by providing lists.
    columns : str or list of str
        Name(s) of the column(s) in ``other`` to merge into ``df``.
    delimiter : str, default ``'|'``
        Delimiter used to split multi‑value keys when ``multiple`` is
        ``True``.
    multiple : bool, default ``False``
        Whether to treat the key column in ``df`` as containing
        delimiter‑separated lists of values.  If ``False``, an exact
        match on the key(s) is required.
    suffix : str, default ``'_y'``
        Suffix to append to merged column names if there is a
        collision with existing column names in ``df``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the selected columns from ``other``
        merged in.  Rows in ``df`` that do not match any key in
        ``other`` will contain ``NaN`` for the merged columns.
    """
    df_keys = [df_key] if isinstance(df_key, str) else list(df_key)
    other_keys = [other_key] if isinstance(other_key, str) else list(other_key)
    if len(df_keys) != len(other_keys):
        raise ValueError("df_key and other_key must have the same number of elements")
    columns_to_add = [columns] if isinstance(columns, str) else list(columns)
    other_merge = other[other_keys + columns_to_add].copy()
    if multiple and len(df_keys) == 1:
        key = df_keys[0]
        # Create a unique row identifier to handle duplicate/unsorted indices
        row_id_col = "_row_id_temp"
        exploded = df[[key]].copy()
        exploded[row_id_col] = range(len(exploded))

        exploded[key] = exploded[key].astype(str).str.split(delimiter)
        exploded = exploded.explode(key)

        merged = exploded.merge(other_merge, left_on=key, right_on=other_keys[0], how='left')
        agg_dict = {col: 'first' for col in columns_to_add}
        aggregated = merged.groupby(row_id_col)[columns_to_add].agg(agg_dict)

        result = df.copy()
        for col in columns_to_add:
            new_name = col if col not in result.columns else f"{col}{suffix}"
            # Align aggregated results back to the original dataframe using the row ID
            result[new_name] = aggregated[col].reindex(range(len(result))).values
        return result
    else:
        result = df.merge(other_merge, left_on=df_keys, right_on=other_keys, how='left', suffixes=('', suffix))
        for k in other_keys:
            if k in result.columns and k not in df_keys:
                result = result.drop(columns=k)
        return result


def conditional_row_merge(
    df: pd.DataFrame,
    other: pd.DataFrame,
    *,
    key_col: str,
    values: Iterable,
    how: str = 'append',
) -> pd.DataFrame:
    """Append or replace rows in ``df`` based on matches in ``other``.

    Parameters
    ----------
    df : pandas.DataFrame
        Primary DataFrame.
    other : pandas.DataFrame
        Secondary DataFrame from which rows are selected.
    key_col : str
        Column name in both DataFrames on which to match.
    values : iterable
        Values to match in ``other[key_col]``.
    how : {'append','replace'}, default 'append'
        If ``'append'``, matching rows from ``other`` are appended to
        ``df``.  If ``'replace'``, matching rows in ``df`` are removed
        before appending.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows appended or replaced.
    """
    sel = other[other[key_col].isin(values)]
    result = df.copy()
    if how not in ['append', 'replace']:
        raise ValueError("how must be 'append' or 'replace'")
    if how == 'replace':
        result = result[~result[key_col].isin(values)]
    combined = pd.concat([result, sel], axis=0, ignore_index=True)
    return combined


def nearest_key_merge(
    df: pd.DataFrame,
    other: pd.DataFrame,
    *,
    df_key: str,
    other_key: str,
    tolerance: Optional[float] = None,
    direction: str = 'nearest',
    suffix: str = '_y',
) -> pd.DataFrame:
    """Merge ``df`` and ``other`` on the nearest numeric key within a tolerance.

    See module documentation for details.
    """
    if df_key not in df.columns or other_key not in other.columns:
        raise KeyError("df_key or other_key column not found in the provided DataFrames")
    # Ensure both keys are numeric and comparable
    # Use astype(float) to align dtypes; invalid parsing will raise
    # Convert key columns to numeric once for efficient comparison
    # Avoid modifying original DataFrames; operate on shallow copies
    df_sorted = df.copy()
    other_sorted = other.copy()
    df_sorted[df_key] = pd.to_numeric(df_sorted[df_key], errors='coerce').astype(float)
    other_sorted[other_key] = pd.to_numeric(other_sorted[other_key], errors='coerce').astype(float)
    df_sorted = df_sorted.sort_values(df_key, kind='mergesort').reset_index(drop=True)
    other_sorted = other_sorted.sort_values(other_key, kind='mergesort').reset_index(drop=True)
    merged = pd.merge_asof(
        df_sorted,
        other_sorted,
        left_on=df_key,
        right_on=other_key,
        direction=direction,
        tolerance=tolerance,
        suffixes=('', suffix),
    )
    merged = merged.sort_index()
    return merged


def coalesce_merge(
    df: pd.DataFrame,
    other: pd.DataFrame,
    *,
    on: Union[str, List[str]],
    prefer: str = 'df',
    suffix: str = '_other',
) -> pd.DataFrame:
    """Merge and coalesce overlapping columns, preferring non‑null values.

    See module documentation for details.
    """
    on_cols = [on] if isinstance(on, str) else list(on)
    merged = df.merge(other, on=on_cols, how='outer', suffixes=('', suffix))
    result = merged.copy()
    overlap = set(df.columns).intersection(other.columns) - set(on_cols)
    for col in overlap:
        other_col = f"{col}{suffix}"
        if prefer == 'df':
            result[col] = merged[col].combine_first(merged[other_col])
        else:
            result[col] = merged[other_col].combine_first(merged[col])
        result = result.drop(columns=other_col)
    return result


def rolling_fill(
    df: pd.DataFrame,
    *,
    window: int = 3,
    method: str = 'mean',
    min_periods: Optional[int] = 1,
) -> pd.DataFrame:
    """Fill missing numeric values using a rolling window statistic.

    See module documentation for details.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    result = df.copy()
    for col in numeric_cols:
        series = df[col]
        if method == 'mean':
            stat = series.rolling(window, min_periods=min_periods).mean()
        elif method == 'median':
            stat = series.rolling(window, min_periods=min_periods).median()
        elif method == 'min':
            stat = series.rolling(window, min_periods=min_periods).min()
        elif method == 'max':
            stat = series.rolling(window, min_periods=min_periods).max()
        else:
            raise ValueError("method must be 'mean', 'median', 'min' or 'max'")
        filled = series.fillna(stat)
        result[col] = filled
    return result


def add_time_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    prefix: Optional[str] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Expand a datetime column into calendar-derived features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str
        Column name holding datetimes.
    prefix : str, optional
        Prefix for generated feature columns. Defaults to ``date_col``.
    drop_original : bool, default False
        Whether to drop the original datetime column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with additional calendar features.
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in DataFrame")
    result = df.copy()
    dt = pd.to_datetime(result[date_col], errors="coerce")
    prefix = f"{date_col}_" if prefix is None else prefix
    result[f"{prefix}year"] = dt.dt.year
    result[f"{prefix}quarter"] = dt.dt.quarter
    result[f"{prefix}month"] = dt.dt.month
    result[f"{prefix}week"] = dt.dt.isocalendar().week.astype("Int64")
    result[f"{prefix}day"] = dt.dt.day
    result[f"{prefix}dayofweek"] = dt.dt.dayofweek
    result[f"{prefix}dayofyear"] = dt.dt.dayofyear
    result[f"{prefix}is_month_start"] = dt.dt.is_month_start
    result[f"{prefix}is_month_end"] = dt.dt.is_month_end
    result[f"{prefix}is_quarter_start"] = dt.dt.is_quarter_start
    result[f"{prefix}is_quarter_end"] = dt.dt.is_quarter_end
    result[f"{prefix}is_year_start"] = dt.dt.is_year_start
    result[f"{prefix}is_year_end"] = dt.dt.is_year_end
    result[f"{prefix}is_weekend"] = dt.dt.dayofweek >= 5
    if drop_original:
        result = result.drop(columns=[date_col])
    return result


def resample_time_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    freq: str,
    agg: Union[str, Callable, Mapping[str, Union[str, Callable]]] = "mean",
    fill_method: Optional[str] = None,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Resample a DataFrame by time and aggregate values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    date_col : str
        Column name containing datetimes.
    freq : str
        Pandas offset alias (e.g., ``'D'``, ``'W'``, ``'M'``).
    agg : str, callable, or mapping, default ``"mean"``
        Aggregation strategy for resampling. Use a mapping to specify
        different aggregations per column.
    fill_method : {'ffill', 'bfill', None}, optional
        Optional fill method to apply after resampling.
    numeric_only : bool, default True
        When ``agg`` is not a mapping, restrict resampling to numeric columns.

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with ``date_col`` restored as a column.
    """
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in DataFrame")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    if isinstance(agg, Mapping):
        resampled = data.set_index(date_col).resample(freq).agg(agg)
    else:
        if numeric_only:
            value_cols = [c for c in data.columns if c != date_col and pd.api.types.is_numeric_dtype(data[c])]
            resampled = data.set_index(date_col)[value_cols].resample(freq).agg(agg)
        else:
            resampled = data.set_index(date_col).resample(freq).agg(agg)
    if fill_method in {"ffill", "bfill"}:
        resampled = resampled.fillna(method=fill_method)
    elif fill_method is not None:
        raise ValueError("fill_method must be 'ffill', 'bfill', or None")
    return resampled.reset_index()


def winsorize_columns(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    limits: Tuple[float, float] = (0.01, 0.01),
) -> pd.DataFrame:
    """Clip numeric columns to quantile bounds to reduce outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : sequence of str, optional
        Columns to winsorize. Defaults to all numeric columns.
    limits : tuple of float, default ``(0.01, 0.01)``
        Lower and upper quantile limits. For example, ``(0.01, 0.01)``
        clips values below the 1st percentile and above the 99th percentile.

    Returns
    -------
    pandas.DataFrame
        DataFrame with winsorized values.
    """
    if len(limits) != 2:
        raise ValueError("limits must be a 2-tuple of (lower, upper)")
    lower, upper = limits
    if not (0 <= lower < 1 and 0 <= upper < 1):
        raise ValueError("limits must be within [0, 1)")
    result = df.copy()
    if columns is None:
        columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    for col in columns:
        series = pd.to_numeric(result[col], errors="coerce")
        lower_q = series.quantile(lower)
        upper_q = series.quantile(1 - upper)
        result[col] = series.clip(lower=lower_q, upper=upper_q)
    return result


def add_lag_features(
    df: pd.DataFrame,
    *,
    lags: Sequence[int],
    columns: Optional[Sequence[str]] = None,
    prefix: str = "lag",
) -> pd.DataFrame:
    """Add lagged versions of numeric columns to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    lags : sequence of int
        Positive integer lags to apply (e.g., ``[1, 7, 14]``).
    columns : sequence of str, optional
        Columns to lag. Defaults to all numeric columns.
    prefix : str, default ``"lag"``
        Prefix for lag feature column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional lag feature columns.
    """
    if not lags:
        raise ValueError("lags must contain at least one integer")
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be positive integers")
    result = df.copy()
    if columns is None:
        columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    for col in columns:
        for lag in lags:
            result[f"{prefix}_{col}_{lag}"] = result[col].shift(lag)
    return result


def scale_columns(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    method: str = "standard",
    center: bool = True,
    scale: bool = True,
) -> pd.DataFrame:
    """Scale numeric columns using standardization or min-max normalization.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : sequence of str, optional
        Columns to scale. Defaults to all numeric columns.
    method : {'standard', 'minmax'}, default ``'standard'``
        Scaling strategy. ``'standard'`` uses (x - mean) / std, while
        ``'minmax'`` uses (x - min) / (max - min).
    center : bool, default True
        Whether to subtract the mean (standard) or min (minmax).
    scale : bool, default True
        Whether to divide by std (standard) or range (minmax).

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled numeric columns.
    """
    result = df.copy()
    if columns is None:
        columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    for col in columns:
        series = pd.to_numeric(result[col], errors="coerce")
        if method == "standard":
            mean = series.mean()
            std = series.std(ddof=0)
            values = series
            if center:
                values = values - mean
            if scale:
                values = values / std if std != 0 else values * 0
        elif method == "minmax":
            min_val = series.min()
            max_val = series.max()
            denom = max_val - min_val
            values = series
            if center:
                values = values - min_val
            if scale:
                values = values / denom if denom != 0 else values * 0
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        result[col] = values
    return result


def rolling_window_features(
    df: pd.DataFrame,
    *,
    window: int,
    columns: Optional[Sequence[str]] = None,
    stats: Sequence[str] = ("mean", "std", "min", "max", "median"),
    prefix: str = "roll",
    min_periods: Optional[int] = 1,
) -> pd.DataFrame:
    """Compute rolling window statistics for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    window : int
        Rolling window size.
    columns : sequence of str, optional
        Columns to compute rolling features for. Defaults to all numeric columns.
    stats : sequence of str, default (mean, std, min, max, median)
        Rolling statistics to compute.
    prefix : str, default ``"roll"``
        Prefix for new feature columns.
    min_periods : int, optional
        Minimum observations required per window.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rolling feature columns appended.
    """
    if window <= 0:
        raise ValueError("window must be a positive integer")
    result = df.copy()
    if columns is None:
        columns = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    for col in columns:
        series = pd.to_numeric(result[col], errors="coerce")
        rolling = series.rolling(window=window, min_periods=min_periods)
        for stat in stats:
            if stat == "mean":
                values = rolling.mean()
            elif stat == "std":
                values = rolling.std(ddof=0)
            elif stat == "min":
                values = rolling.min()
            elif stat == "max":
                values = rolling.max()
            elif stat == "median":
                values = rolling.median()
            else:
                raise ValueError("stats must contain only mean, std, min, max, or median")
            result[f"{prefix}_{col}_{stat}_{window}"] = values
    return result


def get_padd(series: pd.Series, *, specific: bool = True) -> pd.Series:
    """Map state identifiers to Petroleum Administration for Defense Districts.

    This helper normalizes a pandas Series of state identifiers—including
    full names (``"Texas"``), postal abbreviations (``"TX"``), or numeric
    Federal Information Processing Standards (FIPS) codes (``48``)—and
    returns the corresponding Petroleum Administration for Defense District
    (PADD) label.  The mapping is case‑insensitive, trims extraneous
    whitespace, and pads single‑digit numeric inputs to two characters so
    FIPS codes such as ``1`` resolve correctly to ``"01"``.

    Parameters
    ----------
    series : pandas.Series
        Series containing state identifiers.  Values may be strings or
        numbers; all inputs are coerced to strings for matching.
    specific : bool, default True
        When ``True``, PADD 1 is split into "PADD 1-Northeast" (1A and
        1B) and "PADD 1-Southeast" (1C).  When ``False``, all PADD 1
        subdivisions are collapsed into the single label "PADD 1".

    Returns
    -------
    pandas.Series
        Series of human‑readable PADD labels.  Inputs that cannot be
        mapped return ``numpy.nan``.

    Examples
    --------
    >>> identifiers = pd.Series(["Texas", "NY", 12, "northern mariana islands"])
    >>> get_padd(identifiers)
    0               PADD 3
    1    PADD 1-Northeast
    2     PADD 1-Southeast
    3               PADD 7
    dtype: object
    >>> get_padd(identifiers, specific=False)
    0    PADD 3
    1    PADD 1
    2    PADD 1
    3    PADD 7
    dtype: object
    """

    padd_map = {
        # PADD 1A (New England)
        "ct": "1A",
        "me": "1A",
        "ma": "1A",
        "nh": "1A",
        "ri": "1A",
        "vt": "1A",
        "connecticut": "1A",
        "maine": "1A",
        "massachusetts": "1A",
        "new hampshire": "1A",
        "rhode island": "1A",
        "vermont": "1A",
        "09": "1A",
        "23": "1A",
        "25": "1A",
        "33": "1A",
        "44": "1A",
        "50": "1A",

        # PADD 1B (Central Atlantic)
        "de": "1B",
        "dc": "1B",
        "md": "1B",
        "nj": "1B",
        "ny": "1B",
        "pa": "1B",
        "delaware": "1B",
        "district of columbia": "1B",
        "maryland": "1B",
        "new jersey": "1B",
        "new york": "1B",
        "pennsylvania": "1B",
        "10": "1B",
        "11": "1B",
        "24": "1B",
        "34": "1B",
        "36": "1B",
        "42": "1B",

        # PADD 1C (Lower Atlantic)
        "fl": "1C",
        "ga": "1C",
        "nc": "1C",
        "sc": "1C",
        "va": "1C",
        "wv": "1C",
        "florida": "1C",
        "georgia": "1C",
        "north carolina": "1C",
        "south carolina": "1C",
        "virginia": "1C",
        "west virginia": "1C",
        "12": "1C",
        "13": "1C",
        "37": "1C",
        "45": "1C",
        "51": "1C",
        "54": "1C",

        # PADD 2 (Midwest)
        "il": "2",
        "in": "2",
        "ia": "2",
        "ks": "2",
        "ky": "2",
        "mi": "2",
        "mn": "2",
        "mo": "2",
        "ne": "2",
        "nd": "2",
        "oh": "2",
        "ok": "2",
        "sd": "2",
        "tn": "2",
        "wi": "2",
        "illinois": "2",
        "indiana": "2",
        "iowa": "2",
        "kansas": "2",
        "kentucky": "2",
        "michigan": "2",
        "minnesota": "2",
        "missouri": "2",
        "nebraska": "2",
        "north dakota": "2",
        "ohio": "2",
        "oklahoma": "2",
        "south dakota": "2",
        "tennessee": "2",
        "wisconsin": "2",
        "17": "2",
        "18": "2",
        "19": "2",
        "20": "2",
        "21": "2",
        "26": "2",
        "27": "2",
        "29": "2",
        "31": "2",
        "38": "2",
        "39": "2",
        "40": "2",
        "46": "2",
        "47": "2",
        "55": "2",

        # PADD 3 (Gulf Coast)
        "al": "3",
        "ar": "3",
        "la": "3",
        "ms": "3",
        "nm": "3",
        "tx": "3",
        "alabama": "3",
        "arkansas": "3",
        "louisiana": "3",
        "mississippi": "3",
        "new mexico": "3",
        "texas": "3",
        "01": "3",
        "05": "3",
        "22": "3",
        "28": "3",
        "35": "3",
        "48": "3",

        # PADD 4 (Rocky Mountain)
        "co": "4",
        "id": "4",
        "mt": "4",
        "ut": "4",
        "wy": "4",
        "colorado": "4",
        "idaho": "4",
        "montana": "4",
        "utah": "4",
        "wyoming": "4",
        "08": "4",
        "16": "4",
        "30": "4",
        "49": "4",
        "56": "4",

        # PADD 5 (West Coast)
        "ak": "5",
        "az": "5",
        "ca": "5",
        "hi": "5",
        "nv": "5",
        "or": "5",
        "wa": "5",
        "alaska": "5",
        "arizona": "5",
        "california": "5",
        "hawaii": "5",
        "nevada": "5",
        "oregon": "5",
        "washington": "5",
        "02": "5",
        "04": "5",
        "06": "5",
        "15": "5",
        "32": "5",
        "41": "5",
        "53": "5",

        # PADD 6 (Territories)
        "pr": "6",
        "vi": "6",
        "puerto rico": "6",
        "virgin islands": "6",
        "72": "6",
        "78": "6",

        # PADD 7 (Territories)
        "gu": "7",
        "as": "7",
        "mp": "7",
        "guam": "7",
        "american samoa": "7",
        "northern mariana islands": "7",
        "66": "7",
        "60": "7",
        "69": "7",
    }

    label_map = {
        "1A": "PADD 1-Northeast" if specific else "PADD 1",
        "1B": "PADD 1-Northeast" if specific else "PADD 1",
        "1C": "PADD 1-Southeast" if specific else "PADD 1",
        "2": "PADD 2",
        "3": "PADD 3",
        "4": "PADD 4",
        "5": "PADD 5",
        "6": "PADD 6",
        "7": "PADD 7",
    }

    normalized = series.astype(str).str.strip().str.lower()
    normalized = normalized.replace(regex={r"^(\d)$": r"0\1"})

    return normalized.map(padd_map).map(label_map)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a basic data quality report for a pandas DataFrame.

    This utility inspects each column of ``df`` and computes common
    diagnostics, including data type, number of missing values,
    percentage missing, number of unique values and example values.
    The resulting report is useful for quickly assessing the quality
    and characteristics of a dataset before proceeding with analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to profile.

    Returns
    -------
    pandas.DataFrame
        A report with one row per column.  Columns include:

        ``'dtype'`` – the pandas dtype;
        ``'missing_count'`` – number of missing entries;
        ``'missing_pct'`` – percentage of missing entries;
        ``'unique_count'`` – number of unique values (up to a limit);
        ``'example'`` – a representative non‑missing value or None.

    Notes
    -----
    For columns with many unique values, counting all uniques may be
    expensive.  A heuristic is used to approximate the unique count
    when the number exceeds 10000.
    """
    report_data = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        missing_count = series.isna().sum()
        missing_pct = missing_count / n_rows * 100 if n_rows > 0 else np.nan
        # Unique count with heuristic for large cardinality
        if series.nunique(dropna=True) > 10000:
            unique_count = '>10000'
        else:
            unique_count = series.nunique(dropna=True)
        # Example value
        example = None
        non_na = series.dropna()
        if not non_na.empty:
            example = non_na.iloc[0]
        report_data.append({
            'column': col,
            'dtype': str(dtype),
            'missing_count': int(missing_count),
            'missing_pct': float(missing_pct),
            'unique_count': unique_count,
            'example': example,
        })
    return pd.DataFrame(report_data).set_index('column')


def df_split(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    *,
    split_date: Optional[Union[str, pd.Timestamp]] = None,
    split_index: Optional[int] = None,
    include_split: bool = False,
    dropna_target: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series DataFrame into historical and future parts.

    This utility partitions a DataFrame containing a datetime column and a
    target column into two subsets: one containing the historical data
    (training context) and the other containing the future horizon.  It
    provides several ways to define the split point.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least a datetime column and a
        target column.  The DataFrame should be sorted by the datetime
        column in ascending order.
    date_col : str
        Name of the datetime column in ``df``.  This column is used to
        determine the split point when ``split_date`` is provided.
    target_col : str
        Name of the target column in ``df``.  When neither
        ``split_date`` nor ``split_index`` is specified, the function
        finds the last non‑missing value in this column and splits
        immediately after that observation.
    split_date : str or pandas.Timestamp, optional
        Explicit date at which to split.  All rows with
        ``df[date_col]`` strictly less than ``split_date`` (or
        ``<=`` when ``include_split`` is True) are placed in the
        historical DataFrame; the remainder belong to the future DataFrame.
        ``split_date`` is converted to ``datetime64`` using
        ``pandas.to_datetime``.
    split_index : int, optional
        Integer position at which to split the DataFrame.  If
        non‑negative, the first ``split_index`` rows go to the
        historical DataFrame.  Negative indices are interpreted as
        positions from the end.  If both ``split_date`` and
        ``split_index`` are provided, ``split_date`` takes precedence.
    include_split : bool, default False
        Whether to include the row that matches the split point in
        the historical DataFrame.  If ``True`` and ``split_date`` is
        provided, rows with ``df[date_col] == split_date`` are kept in
        the historical DataFrame; otherwise they are included in the
        future DataFrame.  Likewise, when splitting by index,
        ``include_split`` controls whether the row at ``split_index``
        belongs to the historical part.
    dropna_target : bool, default False
        If ``True``, drop rows where the target column is missing
        before determining the last non‑missing value.  When
        ``False``, the last non‑null value is searched in the original
        order including NaNs.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A tuple ``(historical, future)`` where ``historical``
        contains the portion of the data before the split point and
        ``future`` contains the portion at or after the split point.

    Notes
    -----
    Use this helper to prepare datasets for forecasting functions.
    For example, you can train models on the historical DataFrame and
    evaluate forecasts against the future DataFrame.  You can also
    specify ``split_date`` to align the context with a known cut‑off
    point, such as the last observed value in a training set.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in DataFrame")
    # Ensure datetime conversion
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Determine split index
    if split_date is not None:
        # Convert and split by date
        split_ts = pd.to_datetime(split_date)
        if include_split:
            hist_mask = df[date_col] <= split_ts
        else:
            hist_mask = df[date_col] < split_ts
        historical = df.loc[hist_mask].copy()
        future = df.loc[~hist_mask].copy()
    elif split_index is not None:
        # Split by positional index
        if split_index < 0:
            split_index = len(df) + split_index
        if include_split:
            cut = split_index + 1
        else:
            cut = split_index
        historical = df.iloc[:cut].copy()
        future = df.iloc[cut:].copy()
    else:
        # Split after last non‑missing target
        series = df[target_col]
        if dropna_target:
            non_na_idx = series.dropna().index
        else:
            non_na_idx = series[series.notna()].index
        if non_na_idx.size == 0:
            # If no non‑missing values, return empty historical
            historical = df.iloc[[]].copy()
            future = df.copy()
        else:
            last_idx = non_na_idx.max()
            # Determine cut based on last non‑missing row
            cut = df.index.get_loc(last_idx)
            if include_split:
                cut += 1
            historical = df.iloc[:cut].copy()
            future = df.iloc[cut:].copy()
    return historical.reset_index(drop=True), future.reset_index(drop=True)


PromptFunction = Callable[[str, bool], str]


@dataclass
class EnvVariableRequest:
    """Specification for an environment variable prompt.

    Parameters
    ----------
    name : str
        The environment variable name to check or create.
    prompt : str
        The message shown to the user when prompting for the value.
    secret : bool, default False
        Whether the value should be entered without echoing to the
        console (uses :func:`getpass.getpass`).  GUI prompts will mask
        the value when supported by the runtime.
    """

    name: str
    prompt: str
    secret: bool = False


def _default_prompt(message: str, secret: bool = False) -> str:
    """Prompt for a value using a GUI dialog when available, otherwise the console."""

    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        value = simpledialog.askstring(
            title="Input required", prompt=message, show="*" if secret else None
        )
        root.destroy()
        if value is not None:
            return value
    except Exception:
        # Fall back to console prompting below
        pass

    if secret:
        return getpass(f"{message}: ")
    return input(f"{message}: ")


def _normalize_requests(
    variables: Sequence[Union[EnvVariableRequest, Mapping[str, object], str]]
) -> List[EnvVariableRequest]:
    normalized: List[EnvVariableRequest] = []
    for item in variables:
        if isinstance(item, EnvVariableRequest):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(EnvVariableRequest(name=item, prompt=f"Enter a value for {item}"))
        elif isinstance(item, Mapping):
            if "name" not in item:
                raise KeyError("Mapping requests must include a 'name' key")
            normalized.append(
                EnvVariableRequest(
                    name=str(item["name"]),
                    prompt=str(item.get("prompt", f"Enter a value for {item['name']}")),
                    secret=bool(item.get("secret", False)),
                )
            )
        else:
            raise TypeError(
                "variables must contain EnvVariableRequest, mapping objects, or strings"
            )
    return normalized


def ensure_env_variables(
    variables: Sequence[Union[EnvVariableRequest, Mapping[str, object], str]],
    *,
    env: Optional[MutableMapping[str, str]] = None,
    prompt_fn: Optional[PromptFunction] = None,
) -> Dict[str, str]:
    """Ensure required environment variables exist by interactively prompting the user.

    The function checks whether each requested environment variable is already set. If a
    value is missing, the user is prompted to provide one (using a GUI dialog when
    available, otherwise the console). The collected values are written to the provided
    ``env`` mapping (defaults to :data:`os.environ`), making them accessible to any code
    that runs afterward.

    Parameters
    ----------
    variables : sequence
        A list of variable descriptions. Each entry can be an
        :class:`EnvVariableRequest`, a mapping with ``name``, optional ``prompt`` and
        ``secret`` keys, or simply a string containing the environment variable name.
    env : mutable mapping, optional
        Environment mapping to update. Defaults to :data:`os.environ`.
    prompt_fn : callable, optional
        Custom prompting function accepting ``(message, secret)`` and returning a string
        value. When omitted, a GUI dialog is attempted with a console fallback.

    Returns
    -------
    dict
        Dictionary of resolved environment variables (including ones that already
        existed) keyed by name.
    """

    env = os.environ if env is None else env
    prompt = _default_prompt if prompt_fn is None else prompt_fn
    normalized = _normalize_requests(variables)

    resolved: Dict[str, str] = {}
    for request in normalized:
        if request.name not in env or env[request.name] == "":
            env[request.name] = prompt(request.prompt, request.secret)
        resolved[request.name] = env[request.name]
    return resolved


def configure_snowflake_connector(
    *,
    variables: Optional[Sequence[Union[EnvVariableRequest, Mapping[str, object], str]]] = None,
    env: Optional[MutableMapping[str, str]] = None,
    prompt_fn: Optional[PromptFunction] = None,
) -> Dict[str, str]:
    """Interactively collect Snowflake credentials and return connector kwargs.

    This helper walks the user through configuring a Snowflake connection in a script.
    It prompts for common Snowflake environment variables (username, password, account,
    warehouse, database, schema, and role), setting them in ``env`` if they are not
    already defined. Extra variables can be supplied via ``variables`` to cover custom
    settings. The returned dictionary can be passed directly to
    ``snowflake.connector.connect``.

    Examples
    --------
    >>> kwargs = configure_snowflake_connector()
    >>> import snowflake.connector
    >>> conn = snowflake.connector.connect(**kwargs)

    Parameters
    ----------
    variables : sequence, optional
        Additional variable requests to prompt for. Entries follow the same format as
        :func:`ensure_env_variables`.
    env : mutable mapping, optional
        Environment mapping to update. Defaults to :data:`os.environ`.
    prompt_fn : callable, optional
        Custom prompting function accepting ``(message, secret)``.

    Returns
    -------
    dict
        Keyword arguments compatible with :func:`snowflake.connector.connect`.
    """

    default_requests: List[EnvVariableRequest] = [
        EnvVariableRequest("SNOWFLAKE_USER", "Enter your Snowflake username"),
        EnvVariableRequest("SNOWFLAKE_PASSWORD", "Enter your Snowflake password", secret=True),
        EnvVariableRequest(
            "SNOWFLAKE_ACCOUNT", "Enter your Snowflake account identifier (e.g. xy12345.us-east-2)"
        ),
        EnvVariableRequest("SNOWFLAKE_WAREHOUSE", "Enter the default Snowflake warehouse"),
        EnvVariableRequest("SNOWFLAKE_DATABASE", "Enter the default Snowflake database"),
        EnvVariableRequest("SNOWFLAKE_SCHEMA", "Enter the default Snowflake schema"),
        EnvVariableRequest("SNOWFLAKE_ROLE", "Enter the default Snowflake role"),
    ]

    merged_requests = default_requests + list(variables or [])
    resolved = ensure_env_variables(merged_requests, env=env, prompt_fn=prompt_fn)

    connector_kwargs: Dict[str, str] = {
        "user": resolved["SNOWFLAKE_USER"],
        "password": resolved["SNOWFLAKE_PASSWORD"],
        "account": resolved["SNOWFLAKE_ACCOUNT"],
    }

    optional_fields = {
        "SNOWFLAKE_WAREHOUSE": "warehouse",
        "SNOWFLAKE_DATABASE": "database",
        "SNOWFLAKE_SCHEMA": "schema",
        "SNOWFLAKE_ROLE": "role",
    }

    for env_key, connector_key in optional_fields.items():
        value = resolved.get(env_key)
        if value:
            connector_kwargs[connector_key] = value

    return connector_kwargs
