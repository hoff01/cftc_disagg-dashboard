"""
Visualization utilities for time‑series analysis.

This module defines additional plotting functions that complement the
core `five_year_plot` by providing common financial visualizations
such as cumulative return charts and drawdown analyses.  All
functions use Plotly to produce interactive figures suitable for
exploratory analysis and presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def cumulative_return_plot(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    return_type: str = 'log',
) -> go.Figure:
    """Plot cumulative returns of one or more series.

    Cumulative returns are computed as the cumulative product of
    ``1 + pct_change`` or the exponentiated cumulative sum of log
    returns.  The resulting figure contains one trace per series.

    Parameters
    ----------
    date : str or pandas.Series
        Column or series containing dates for the x‑axis.
    df : pandas.DataFrame
        DataFrame with numeric series to plot.
    columns : list of str or None, default None
        Specific columns to include.  If ``None``, all numeric
        columns except the date column are used.
    return_type : {'log','pct'}, default 'log'
        Type of return to compute before accumulation.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure showing cumulative returns.

    Notes
    -----
    Cumulative returns are normalised to start at 1.  Missing values
    are forward filled before return calculation.
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
        raise ValueError("No numeric columns selected for cumulative return plot")
    fig = go.Figure()
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        if return_type == 'log':
            returns = np.log(series).diff().fillna(0.0)
            cumulative = np.exp(returns.cumsum())
        elif return_type == 'pct':
            returns = series.pct_change().fillna(0.0)
            cumulative = (1 + returns).cumprod()
        else:
            raise ValueError("return_type must be 'log' or 'pct'")
        fig.add_trace(go.Scatter(x=dt_index, y=cumulative, mode='lines', name=col))
    fig.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
    return fig


@dataclass
class DrawdownResult:
    """Container for drawdown series and summary statistics.

    Attributes
    ----------
    drawdown : pandas.DataFrame
        DataFrame of drawdown values for each series.  Drawdown is
        defined as ``(price / running_max) - 1``.
    max_drawdown : dict
        Mapping of column name to the maximum (most negative) drawdown
        value observed in that series.
    """
    drawdown: pd.DataFrame
    max_drawdown: dict


def max_drawdown(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    return_type: str = 'log',
) -> DrawdownResult:
    """Compute the drawdown series and maximum drawdown for each column.

    Drawdown measures the percentage decline from a running maximum.  It
    helps traders understand the worst losses experienced over a period.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment.  Not used in calculation
        but returned as the index of the drawdown DataFrame.
    df : pandas.DataFrame
        DataFrame containing price data.
    columns : list of str or None, default None
        Specific columns to compute drawdown for.  If ``None``, all
        numeric columns except the date are used.
    return_type : {'log','pct'}, default 'log'
        Unused parameter included for API symmetry with other
        functions.  Future versions may use this to select between
        price or return based drawdowns.

    Returns
    -------
    DrawdownResult
        Dataclass containing the drawdown series and the maximum
        drawdown per column.
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
        raise ValueError("No numeric columns selected for drawdown calculation")
    drawdown_data = {}
    max_dd = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce').ffill().bfill().values
        running_max = np.maximum.accumulate(series)
        dd = (series / running_max) - 1.0
        drawdown_data[col] = dd
        max_dd[col] = float(dd.min())
    drawdown_df = pd.DataFrame(drawdown_data, index=dt_index)
    return DrawdownResult(drawdown=drawdown_df, max_drawdown=max_dd)


def forecast_plot(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    forecast: pd.DataFrame,
    lower: Optional[pd.DataFrame] = None,
    upper: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    test_start: Optional[Union[str, pd.Timestamp]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot historical data against forecasts with optional confidence bands.

    This utility creates an interactive Plotly figure showing the
    original time series alongside forecasted values.  If lower and
    upper confidence interval DataFrames are provided, shaded areas
    representing prediction intervals are drawn.  Multiple series
    are displayed on separate subplots sharing the x‑axis.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates for the x‑axis of
        the historical data.
    df : pandas.DataFrame
        DataFrame with the historical data.  Numeric columns are
        plotted as the actual values.
    forecast : pandas.DataFrame
        DataFrame containing forecasted values.  The index should
        represent future dates and columns correspond to those in
        ``df``.
    lower : pandas.DataFrame or None, default None
        Lower bounds of prediction intervals.  Must have the same
        shape and column names as ``forecast``.  If ``None``, no
        confidence region is plotted.
    upper : pandas.DataFrame or None, default None
        Upper bounds of prediction intervals.  Must have the same
        shape and column names as ``forecast``.  Ignored if
        ``lower`` is ``None``.
    columns : list of str or None, default None
        Specific columns to plot.  If ``None``, all numeric
        columns present in both ``df`` and ``forecast`` are used.
    test_start : str or pandas.Timestamp or None, default None
        Optional boundary indicating where the historical data
        transitions from training to test/hold-out observations.
        When provided (or inferred; see Notes), data at or after
        this timestamp are plotted as a separate "test" trace so
        that a single DataFrame containing both training and test
        observations can be visualised.
    title : str or None, default None
        Title for the overall figure.  A default is generated
        otherwise.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure containing subplots for each series.

    Notes
    -----
    If ``test_start`` is not supplied but the historical DataFrame
    already contains rows whose dates overlap the forecast horizon
    (i.e., ``df`` includes both training and test data), the split
    point is automatically inferred from the first forecast date.
    """
    # Determine historical dates
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        dt_hist = pd.to_datetime(df[date], errors="coerce")
    else:
        dt_hist = pd.to_datetime(date, errors="coerce")
    valid_hist = ~dt_hist.isna()
    if not valid_hist.any():
        raise ValueError("Historical dates must be valid datetimes for plotting")
    if not valid_hist.all():
        df = df.loc[valid_hist].copy()
        dt_hist = dt_hist.loc[valid_hist]
    sort_idx = dt_hist.argsort()
    if not dt_hist.is_monotonic_increasing:
        df = df.iloc[sort_idx].reset_index(drop=True)
        dt_hist = dt_hist.iloc[sort_idx].reset_index(drop=True)
    else:
        dt_hist = dt_hist.reset_index(drop=True)
    # Normalize forecast index to datetime and sort
    forecast_clean = forecast.copy()
    forecast_clean.index = pd.to_datetime(forecast_clean.index, errors="coerce")
    forecast_clean = forecast_clean.loc[~forecast_clean.index.isna()].sort_index()
    if forecast_clean.empty:
        raise ValueError("Forecast data must contain valid datetime index values")
    # Establish a mask indicating test/hold-out rows when available
    test_mask: Optional[pd.Series]
    if test_start is not None:
        test_boundary = pd.to_datetime(test_start)
        test_mask = dt_hist >= test_boundary
    else:
        forecast_start = forecast_clean.index.min()
        if pd.notna(forecast_start) and (dt_hist >= forecast_start).any():
            test_mask = dt_hist >= forecast_start
        else:
            test_mask = None
    if lower is not None:
        lower = lower.copy()
        lower.index = pd.to_datetime(lower.index, errors="coerce")
        lower = lower.loc[~lower.index.isna()].sort_index()
    if upper is not None:
        upper = upper.copy()
        upper.index = pd.to_datetime(upper.index, errors="coerce")
        upper = upper.loc[~upper.index.isna()].sort_index()

    # Determine columns to plot
    if columns is None:
        # Intersection of numeric columns in df and forecast
        cols_hist = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
        cols_fore = list(forecast_clean.columns)
        columns = [c for c in cols_hist if c in cols_fore]
    if not columns:
        raise ValueError("No common numeric columns found for plotting forecasts")
    n_series = len(columns)
    # Create subplots
    fig = make_subplots(rows=n_series, cols=1, shared_xaxes=True, subplot_titles=columns)
    # Default title
    if title is None:
        title = "Forecast vs. Historical"
    # Plot each series
    for i, col in enumerate(columns, start=1):
        # Historical trace
        series_hist = pd.to_numeric(df[col], errors='coerce').ffill().bfill()
        if test_mask is not None and test_mask.any() and (~test_mask).any():
            # Training portion
            fig.add_trace(
                go.Scatter(
                    x=dt_hist[~test_mask],
                    y=series_hist[~test_mask],
                    mode='lines',
                    name=f"{col} (train)",
                    line=dict(color='blue'),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
            # Test portion
            fig.add_trace(
                go.Scatter(
                    x=dt_hist[test_mask],
                    y=series_hist[test_mask],
                    mode='lines',
                    name=f"{col} (test)",
                    line=dict(color='gray', dash='dot'),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=dt_hist,
                    y=series_hist,
                    mode='lines',
                    name=f"{col} (history)",
                    line=dict(color='blue'),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
        # Forecast trace
        forecast_series = pd.to_numeric(forecast_clean[col], errors='coerce').ffill().bfill()
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                mode='lines',
                name=f"{col} (forecast)",
                line=dict(color='red', dash='dash'),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )
        # Confidence intervals
        if lower is not None and upper is not None and col in lower.columns and col in upper.columns:
            lower_series = pd.to_numeric(lower[col], errors='coerce').ffill().bfill()
            upper_series = pd.to_numeric(upper[col], errors='coerce').ffill().bfill()
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=upper_series.index,
                    y=upper_series,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )
            # Lower band with fill to upper
            fig.add_trace(
                go.Scatter(
                    x=lower_series.index,
                    y=lower_series,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f"{col} (interval)" if i == 1 else None,
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
    fig.update_layout(title=title, xaxis_title="Date")
    # Only label y‑axis on the leftmost subplot
    for i, col in enumerate(columns, start=1):
        fig.update_yaxes(title_text=col, row=i, col=1)
    return fig


def acf_pacf_plot(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    column: str,
    nlags: int = 40,
    alpha: float = 0.05,
) -> go.Figure:
    """Plot the autocorrelation and partial autocorrelation functions.

    This function computes the autocorrelation function (ACF) and
    partial autocorrelation function (PACF) of a specified series and
    visualises them in a two‑panel Plotly figure.  Confidence
    intervals are drawn as horizontal bands based on a specified
    significance level.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates.  Used only for
        alignment and is not used directly in ACF/PACF calculations.
    df : pandas.DataFrame
        DataFrame containing the data.
    column : str
        Name of the column to analyse.  Must be present in ``df``.
    nlags : int, default 40
        Number of lags to compute.  Larger values provide more
        information but increase noise.
    alpha : float, default 0.05
        Significance level for confidence intervals.  A typical
        choice is 0.05 (95% confidence).

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with two subplots: ACF (top) and PACF (bottom).
    """
    # Validate column
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    series = pd.to_numeric(df[column], errors='coerce').dropna().astype(float)
    if len(series) < 2:
        raise ValueError("Series is too short to compute ACF/PACF")
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except Exception as e:
        raise ImportError(
            "statsmodels is required for acf_pacf_plot. "
            "Please install statsmodels to use this function."
        ) from e
    # Compute ACF with confidence intervals
    acf_vals, confint_acf = acf(series, nlags=nlags, alpha=alpha, fft=True)
    # Compute PACF with confidence intervals
    # Statsmodels pacf does not support 'efficient' argument in all versions
    # Use Yule‑Walker method for PACF estimation; unbiased variant not supported in all statsmodels versions
    pacf_vals, confint_pacf = pacf(series, nlags=nlags, alpha=alpha, method='yw')
    # Lag indices
    lags = np.arange(len(acf_vals))
    # Build figure with two subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[f"ACF: {column}", f"PACF: {column}"])
    # ACF bars
    fig.add_trace(
        go.Bar(x=lags, y=acf_vals, name='ACF', marker_color='steelblue'),
        row=1, col=1,
    )
    # ACF confidence intervals as shaded area
    upper_acf = confint_acf[:, 1] - acf_vals
    lower_acf = acf_vals - confint_acf[:, 0]
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=upper_acf + acf_vals,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=lower_acf * -1 + acf_vals,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=1, col=1,
    )
    # PACF bars
    fig.add_trace(
        go.Bar(x=lags, y=pacf_vals, name='PACF', marker_color='darkorange'),
        row=2, col=1,
    )
    # PACF confidence intervals
    upper_pacf = confint_pacf[:, 1] - pacf_vals
    lower_pacf = pacf_vals - confint_pacf[:, 0]
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=upper_pacf + pacf_vals,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=lower_pacf * -1 + pacf_vals,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_layout(height=600, title_text=f"ACF and PACF for {column}")
    fig.update_xaxes(title_text='Lag', row=2, col=1)
    fig.update_yaxes(title_text='Correlation', row=1, col=1)
    fig.update_yaxes(title_text='Partial Correlation', row=2, col=1)
    return fig
