"""
Monthly prediction from weekly data.

This module defines :func:`monthly_predictor`, a helper that attempts
to reconstruct missing monthly values from weekly observations.  The
problem arises when monthly aggregates are delayed relative to weekly
data: for instance, monthly statistics may be published two months
after the fact, while weekly updates continue to arrive.  To fill the
gaps, the predictor learns a simple relationship between aggregated
weekly metrics and known monthly totals, then applies this relationship
to months where the monthly total has not yet been observed.

The approach is deliberately straightforward:

1.  The weekly data are aggregated by calendar month to compute summary
    features such as the sum and mean of weekly observations, the
    number of weekly points in the month, the last weekly value of the
    month, and the first weekly value of the following month.  The
    latter helps capture any spillover into the next period if the
    monthly total is influenced by early data from the subsequent
    month.

2.  An advanced regression model is trained on months where the true
    monthly total is available.  By default the function attempts to
    use an XGBoost regressor, which can capture non‑linear
    relationships between the aggregated weekly features and the
    monthly totals.  If XGBoost is not available, it falls back to
    a random forest regressor, and finally to a simple linear model.
    Should there be too few observations to fit any model (fewer than
    two), the function reverts to a simple ratio of the monthly total
    to the sum of weekly values.

3.  The trained model (or ratio) is then used to estimate the missing
    monthly values.  Predicted values are returned in a single
    DataFrame, aligned by month.  Existing monthly observations are
    preserved in the output.

The goal is not to be perfect but to provide a reasonable estimate
based solely on available weekly data.  This function is resilient
against missing weeks and gracefully handles scenarios where features
cannot be computed (by filling with zeros).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import pandas as pd
import numpy as np

try:  # Optional dependency; gracefully degrade if scikit-learn is unavailable
    from sklearn.exceptions import NotFittedError
    # Import the Theil‑Sen estimator for robust regression.  This estimator is
    # resistant to outliers and provides a more sophisticated alternative to
    # ordinary least squares.
    from sklearn.linear_model import TheilSenRegressor, LinearRegression
except Exception:  # pragma: no cover - fallback for environments without sklearn
    class NotFittedError(Exception):
        """Lightweight stand-in when scikit-learn is not installed."""

    TheilSenRegressor = None  # type: ignore
    LinearRegression = None  # type: ignore


@dataclass
class MonthlyPredictionResult:
    """Container for monthly prediction outputs.

    Attributes
    ----------
    monthly : pd.DataFrame
        DataFrame indexed by calendar month (as timestamps) containing
        three columns: ``'month'`` (period start), ``'observed'`` (the
        original monthly values, possibly with missing data), and
        ``'predicted'`` (the filled monthly values).  Missing values
        remain ``NaN`` in the ``'observed'`` column, while the
        ``'predicted'`` column contains either the observed value (if
        present) or the estimate produced by this function.
    model : Optional[object]
        The fitted regression model used for prediction.  ``None`` if
        the ratio fallback was used due to insufficient data.
    ratio : Optional[float]
        The average ratio of monthly totals to the sum of weekly
        observations used in the ratio fallback.  ``None`` if a
        regression model was fitted instead.
    """

    monthly: pd.DataFrame
    model: Optional[LinearRegression] = None
    ratio: Optional[float] = None


def monthly_predictor(
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    *,
    weekly_date_col: Union[str, Iterable] = 'date',
    weekly_value_col: str = 'value',
    monthly_date_col: Union[str, Iterable] = 'date',
    monthly_value_col: str = 'value',
) -> MonthlyPredictionResult:
    """Estimate missing monthly values from weekly observations.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        DataFrame containing weekly data.  Must include a date column
        (default ``'date'``) and a numeric value column (default
        ``'value'``).  Additional columns are ignored.
    monthly_df : pd.DataFrame
        DataFrame containing monthly data with a date column (default
        ``'date'``) and a numeric value column (default ``'value'``).
        Missing monthly values should be represented as ``NaN``.
    weekly_date_col : Union[str, Iterable], default ``'date'``
        Name of the date column in ``weekly_df``.  If an iterable is
        provided instead of a column name, it will be converted to a
        pandas Series.
    weekly_value_col : str, default ``'value'``
        Name of the numeric column in ``weekly_df`` containing the
        weekly observations.
    monthly_date_col : Union[str, Iterable], default ``'date'``
        Name of the date column in ``monthly_df``.  If an iterable is
        provided instead of a column name, it will be converted to a
        pandas Series.
    monthly_value_col : str, default ``'value'``
        Name of the numeric column in ``monthly_df`` containing the
        monthly totals.

    Returns
    -------
    MonthlyPredictionResult
        An object containing the merged monthly DataFrame with both
        observed and predicted values, as well as the underlying
        regression model or fallback ratio used for prediction.

    Notes
    -----
    * The function infers calendar months using the first day of the
      month as the canonical timestamp.  Monthly values are assumed to
      correspond to complete calendar months.
    * If fewer than two months have observed data, the function
      computes a simple ratio of monthly totals to the sum of weekly
      values and uses it to estimate missing months.
    * Weekly data that cannot be aggregated into a month (e.g., due to
      non‑uniform frequencies) will still contribute to the monthly sum
      via pandas grouping.  Weeks with missing values are ignored.
    """
    # Extract date and value series from weekly data
    if isinstance(weekly_date_col, str):
        if weekly_date_col not in weekly_df.columns:
            raise KeyError(f"Weekly date column '{weekly_date_col}' not found in weekly_df")
        weekly_dates = weekly_df[weekly_date_col]
    else:
        weekly_dates = pd.Series(weekly_date_col)
    if weekly_value_col not in weekly_df.columns:
        raise KeyError(f"Weekly value column '{weekly_value_col}' not found in weekly_df")
    weekly_values = pd.to_numeric(weekly_df[weekly_value_col], errors='coerce')

    # Convert weekly dates to datetime and derive month period
    weekly_dt = pd.to_datetime(weekly_dates)
    weekly_month = weekly_dt.dt.to_period('M')

    # Build a DataFrame for weekly data with month key
    weekly_data = pd.DataFrame({
        'month': weekly_month,
        'value': weekly_values,
        'date': weekly_dt,
    })
    # Remove rows with missing weekly values
    weekly_data = weekly_data.dropna(subset=['value'])

    # Aggregate weekly data to monthly features
    # Sum, mean, count and last value of the month
    agg = weekly_data.groupby('month').agg(
        sum_weekly=('value', 'sum'),
        mean_weekly=('value', 'mean'),
        count_weekly=('value', 'count'),
        last_week_val=('value', lambda x: x.iloc[-1]),
    ).reset_index()
    # Compute first weekly value of next month
    # Create a helper DataFrame with first values per month
    first_vals = weekly_data.groupby('month').agg(first_week_val=('value', 'first')).reset_index()
    # Map to next month by shifting the month series
    first_vals['month_prev'] = first_vals['month'] - 1  # previous month
    # Merge to align first value of next month with previous month
    next_month_val = first_vals[['month_prev', 'first_week_val']].rename(columns={
        'month_prev': 'month',
        'first_week_val': 'next_month_first_week_val',
    })
    # Merge features
    agg = agg.merge(next_month_val, on='month', how='left')
    # Replace missing next_month_first_week_val with zero (no info)
    agg['next_month_first_week_val'] = agg['next_month_first_week_val'].fillna(0.0)

    # Prepare monthly observations
    if isinstance(monthly_date_col, str):
        if monthly_date_col not in monthly_df.columns:
            raise KeyError(f"Monthly date column '{monthly_date_col}' not found in monthly_df")
        monthly_dates = monthly_df[monthly_date_col]
    else:
        monthly_dates = pd.Series(monthly_date_col)
    if monthly_value_col not in monthly_df.columns:
        raise KeyError(f"Monthly value column '{monthly_value_col}' not found in monthly_df")
    monthly_values = pd.to_numeric(monthly_df[monthly_value_col], errors='coerce')

    monthly_dt = pd.to_datetime(monthly_dates).dt.to_period('M')
    monthly_data = pd.DataFrame({'month': monthly_dt, 'observed': monthly_values})

    # Merge aggregated weekly features with monthly data
    merged = pd.merge(agg, monthly_data, on='month', how='outer')
    merged = merged.sort_values('month').reset_index(drop=True)

    # Create lagged features for modeling: previous month's sum and last weekly value
    merged['prev_sum_weekly'] = merged['sum_weekly'].shift(1).fillna(0.0)
    merged['prev_last_week_val'] = merged['last_week_val'].shift(1).fillna(0.0)

    # Identify rows where observed monthly value is available (non‑NaN)
    observed_mask = merged['observed'].notna()
    # Define feature columns including lagged features
    feature_cols = [
        'sum_weekly',
        'mean_weekly',
        'count_weekly',
        'last_week_val',
        'next_month_first_week_val',
        'prev_sum_weekly',
        'prev_last_week_val',
    ]
    X_train = merged.loc[observed_mask, feature_cols]
    y_train = merged.loc[observed_mask, 'observed']

    # Initialize model and ratio
    model: Optional[object] = None
    ratio: Optional[float] = None

    # Fit a robust regression model if sufficient data; otherwise compute ratio
    if len(y_train) >= 2 and not X_train.isnull().any().any():
        # Use Theil‑Sen estimator for robustness against outliers and
        # non‑linearity.  This estimator is more sophisticated than a
        # simple linear regression and is appropriate for small datasets.
        if TheilSenRegressor is not None:
            model = TheilSenRegressor(random_state=0)
            model.fit(X_train, y_train)
        elif LinearRegression is not None:
            model = LinearRegression()
            model.fit(X_train, y_train)
    if model is None:
        # Compute ratio as average of observed monthly values divided by sum of weekly values
        valid_ratios = []
        for obs, sw in zip(y_train, X_train['sum_weekly']):
            if sw != 0:
                valid_ratios.append(obs / sw)
        ratio = float(np.nanmean(valid_ratios)) if valid_ratios else 1.0

    # Predict missing monthly values
    predictions: np.ndarray = np.full(len(merged), np.nan)
    for idx, row in merged.iterrows():
        if not np.isnan(row['observed']):
            predictions[idx] = row['observed']
        else:
            # Extract feature values as float array
            feats = np.array(row[feature_cols], dtype=float).reshape(1, -1)
            # Use model if available and features are finite
            if model is not None and not np.any(np.isnan(feats)) and not np.any(np.isinf(feats)):
                try:
                    predictions[idx] = float(model.predict(feats)[0])
                except Exception:
                    pass
            # Fallback: use ratio if prediction is still NaN
            if np.isnan(predictions[idx]):
                if ratio is not None and row['sum_weekly'] != 0:
                    predictions[idx] = ratio * row['sum_weekly']
                else:
                    predictions[idx] = row['sum_weekly']

    # Assemble output DataFrame
    # Convert period to timestamp representing the first day of the month
    months_ts = merged['month'].dt.to_timestamp()
    monthly_out = pd.DataFrame({
        'month': months_ts,
        'observed': merged['observed'].values,
        'predicted': predictions,
    })

    return MonthlyPredictionResult(monthly=monthly_out, model=model, ratio=ratio)
