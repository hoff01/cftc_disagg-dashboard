"""
Machine learning forecasting utilities using AutoGluon.

The functions in this module build upon AutoGluon‐TimeSeries to provide
simple wrappers for forecasting multiple independent time series stored
within a single pandas DataFrame.  Each numeric column (other than the
date column) is treated as its own time series.  A separate model is
trained for each series using AutoGluon's high‑level presets, and the
resulting forecasts are returned in a single DataFrame.  Optionally,
prediction intervals (confidence intervals) can also be returned.

The primary function is :func:`ml_forecast`.

Note
----
AutoGluon is a heavy dependency and is **not** included by default when
installing this package.  To use :func:`ml_forecast`, install the
``autogluon-timeseries`` package version >= 1.0, for example via

```
pip install autogluon.timeseries
```

or follow the installation instructions on the AutoGluon website.  If
AutoGluon is not available, :func:`ml_forecast` will raise an
ImportError.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Dict

import pandas as pd
import numpy as np

@dataclass
class ForecastResult:
    """Container for forecast outputs.

    Attributes
    ----------
    forecasts : pd.DataFrame
        DataFrame containing the point forecasts.  The first column is
        ``'date'`` followed by one column per target series.  The length
        of this DataFrame equals the specified forecast horizon.
    conf_intervals : Optional[pd.DataFrame]
        DataFrame containing the lower and upper prediction bounds
        concatenated side‑by‑side.  This DataFrame has the same
        ``'date'`` column as ``forecasts``.  For each target series
        ``col``, two columns are included: ``f"{col}_lower"`` and
        ``f"{col}_upper"``.  This attribute is ``None`` if
        ``return_conf_int`` is set to ``False`` when calling
        :func:`ml_forecast`.
    """

    forecasts: pd.DataFrame
    conf_intervals: Optional[pd.DataFrame] = None


def ml_forecast(
    date: Union[str, pd.Series, Iterable],
    df: pd.DataFrame,
    *,
    periods: int = 12,
    freq: Optional[str] = None,
    presets: str = "best_quality",
    model: Optional[str] = None,
    return_conf_int: bool = True,
    quantile_levels: Tuple[float, float] = (0.05, 0.95),
    time_limit: Optional[int] = None,
    random_seed: Optional[int] = 0,
) -> ForecastResult:
    """Forecast future values for each numeric column using AutoGluon.

    This function trains an individual AutoGluon time series model for
    each numeric column in ``df`` (excluding the specified date column)
    and generates forecasts for the next ``periods`` time steps.  The
    resulting predictions are returned as a DataFrame with a common
    ``'date'`` column and one column per original series.  Optionally,
    prediction intervals can also be returned in a second DataFrame.

    Parameters
    ----------
    date : Union[str, pd.Series, Iterable]
        Column name, pandas Series, or iterable containing the date for
        each observation.  These will be converted to ``datetime64``.
    df : pd.DataFrame
        DataFrame containing the time series.  Numeric columns other
        than the date column are interpreted as separate series.
    periods : int, default 12
        Number of periods to forecast into the future.
    freq : Optional[str], default ``None``
        String representing the frequency of the data (e.g., ``"D"`` for
        daily, ``"W"`` for weekly).  If ``None``, the frequency will be
        inferred from the ``date`` series using :func:`pandas.infer_freq`.
        If inference fails, an exception is raised.
    presets : str, default ``"best_quality"``
        Preset string passed to AutoGluon specifying the trade‑off
        between accuracy and training speed.  See the AutoGluon
        documentation for details.  Common options include
        ``"fast_training"``, ``"medium_quality"`` and ``"best_quality"``.
    model : Optional[str], default ``None``
        Name of a specific model trained by AutoGluon to use for
        prediction.  When ``None`` (the default), AutoGluon will select
        the best model based on validation performance.
    return_conf_int : bool, default ``True``
        Whether to return prediction intervals.  If ``True``, the
        returned :class:`ForecastResult` will contain a second
        DataFrame ``conf_intervals`` with lower and upper bounds.
    quantile_levels : Tuple[float, float], default ``(0.05, 0.95)``
        Quantile levels to use for the lower and upper prediction
        bounds.  Only used if ``return_conf_int`` is ``True``.
    time_limit : Optional[int], default ``None``
        Maximum training time in seconds to allocate per series.  If
        ``None``, AutoGluon will train until convergence.
    random_seed : Optional[int], default ``0``
        Random seed passed to AutoGluon for reproducible results.

    Returns
    -------
    ForecastResult
        An object containing the point forecasts and, optionally,
        prediction intervals.

    Raises
    ------
    ImportError
        If AutoGluon is not installed.  Install with
        ``pip install autogluon.timeseries`` to enable this feature.
    ValueError
        If no numeric columns are found or the frequency cannot be
        inferred.
    KeyError
        If the specified date column name does not exist in ``df``.

    Examples
    --------
    >>> from analysis3054 import ml_forecast
    >>> result = ml_forecast(date='date', df=my_dataframe, periods=12)
    >>> forecasts = result.forecasts
    >>> conf_ints = result.conf_intervals
    """
    try:
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    except Exception as e:
        raise ImportError(
            "AutoGluon is required for ml_forecast. Install via 'pip install autogluon.timeseries'"
        ) from e

    # Extract the date series
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        date_series = df[date]
    else:
        date_series = pd.Series(date)

    # Infer frequency if not provided
    if freq is None:
        inferred = pd.infer_freq(pd.to_datetime(date_series).sort_values())
        if inferred is None:
            raise ValueError(
                "Unable to infer frequency from the date series. Please specify the 'freq' parameter."
            )
        freq = inferred

    # Identify numeric columns to forecast
    cols_to_forecast: List[str] = []
    for col in df.columns:
        if col == date:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols_to_forecast.append(col)
    if not cols_to_forecast:
        raise ValueError("No numeric data columns found to forecast")

    # Prepare outputs
    forecast_values: Dict[str, np.ndarray] = {}
    lower_bounds: Dict[str, np.ndarray] = {}
    upper_bounds: Dict[str, np.ndarray] = {}

    # Convert dates to datetime once
    dt = pd.to_datetime(date_series)
    last_date = dt.max()

    for col in cols_to_forecast:
        # Extract series and drop missing values in alignment with date
        series = df[col].astype(float)
        # Build a DataFrame for AutoGluon with item_id, timestamp and target
        ts_df = pd.DataFrame({
            'item_id': [col] * len(dt),
            'timestamp': dt,
            'target': series.values,
        })
        # Drop rows with missing target values
        ts_df = ts_df.dropna(subset=['target'])
        if ts_df.empty:
            raise ValueError(f"Series '{col}' contains no valid observations after removing missing values")
        # Convert to TimeSeriesDataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(
            ts_df,
            id_column='item_id',
            timestamp_column='timestamp',
        )
        # Initialize predictor
        predictor = TimeSeriesPredictor(
            prediction_length=periods,
            freq=freq,
            target='target',
            random_seed=random_seed,
        )
        # Train model
        predictor.fit(
            train_data=train_data,
            presets=presets,
            time_limit=time_limit,
        )
        # Generate point forecasts
        forecast_df = predictor.predict(
            train_data=train_data,
            model=model,
        )
        # Convert to pandas and extract the target column for this series
        # forecast_df has a MultiIndex (item_id, timestamp) and a single 'target' column
        forecast_series = forecast_df.loc[col]
        forecast_values[col] = forecast_series['target'].values
        # Generate prediction intervals if requested
        if return_conf_int:
            lower_q, upper_q = quantile_levels
            ci_df = predictor.predict(
                train_data=train_data,
                model=model,
                quantile_levels=[lower_q, upper_q],
            )
            ci_series = ci_df.loc[col]
            lower_bounds[col] = ci_series[str(lower_q)].values
            upper_bounds[col] = ci_series[str(upper_q)].values

    # Build the forecast dates
    # Use the inferred frequency to generate the forecast horizon
    # Add offset of one period after the last date
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception:
        offset = pd.tseries.frequencies.to_offset(pd.tseries.frequencies.to_offset(freq))
    start_date = last_date + offset
    forecast_dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Assemble point forecast DataFrame
    forecasts_df = pd.DataFrame(forecast_values)
    forecasts_df.insert(0, 'date', forecast_dates)

    # Assemble confidence interval DataFrame if needed
    if return_conf_int:
        conf_df = pd.DataFrame()
        conf_df['date'] = forecast_dates
        for col in cols_to_forecast:
            conf_df[f"{col}_lower"] = lower_bounds[col]
            conf_df[f"{col}_upper"] = upper_bounds[col]
        return ForecastResult(forecasts=forecasts_df, conf_intervals=conf_df)
    else:
        return ForecastResult(forecasts=forecasts_df, conf_intervals=None)
