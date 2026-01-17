"""
Plotting functions for the EIA 5‑year band plot.

This module exposes a single function, :func:`five_year_plot`, which
generates interactive plots using Plotly to visualize a time series
against its five‑year historical range, average, and recent values.
The resulting figure helps analysts quickly see whether current
observations are high, low, or typical relative to recent history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qcolors

# For smoothing of line data
from datetime import datetime, timedelta
try:
    # SciPy is used for cubic spline smoothing.  If unavailable, smoothing
    # functionality will gracefully fall back to no smoothing.
    from scipy.interpolate import make_interp_spline  # type: ignore
except Exception:
    make_interp_spline = None  # type: ignore


@dataclass
class FiveYearPlotConfig:
    """Configuration options for the 5‑year band plot.

    Attributes
    ----------
    prior_year_lines : int
        Number of individual prior years (immediately preceding the current
        year) to draw as separate lines. The remaining years up to the
        five‑year range contribute only to the shaded band and average.
    current_year_color : str
        Colour for the current year line.  Defaults to ``"blue"``.
    first_prior_year_color : str
        Colour for the most recent prior year line.  Defaults to ``"red"``.
        When ``prior_year_lines`` is greater than 1, subsequent prior years
        will be assigned colours from the Plotly qualitative palette.
    band_color : str
        Colour (including opacity) for the five‑year min/max band.  Defaults
        to a light grey with 50 % opacity.
    mean_color : str
        Colour for the five‑year average line.  Defaults to ``"black"``.
    """

    prior_year_lines: int = 1
    current_year_color: str = "blue"
    first_prior_year_color: str = "red"
    band_color: str = "rgba(128,128,128,0.3)"
    mean_color: str = "black"


def _prepare_time_series(date: Iterable, values: pd.Series) -> pd.DataFrame:
    """Prepare a DataFrame with year and day‑of‑year for grouping.

    Parameters
    ----------
    date : Iterable
        Iterable of datetime‑like objects.  These will be converted to
        ``pd.Timestamp`` via ``pd.to_datetime``.
    values : pd.Series
        The values corresponding to the dates.

    Returns
    -------
    DataFrame
        DataFrame with columns ``'year'``, ``'dayofyear'`` and ``'value'``.
    """
    dt = pd.to_datetime(date)
    series = pd.Series(values).reset_index(drop=True)
    # ensure length match
    if len(dt) != len(series):
        raise ValueError("Length of date and values must match")
    df = pd.DataFrame({
        'year': dt.dt.year,
        'dayofyear': dt.dt.dayofyear,
        'value': series
    })
    return df


def five_year_plot(
    date: Union[str, pd.Series, Iterable],
    df: pd.DataFrame,
    *,
    prior_year_lines: int = 1,
    current_year_color: str = "blue",
    prior_year_color: str = "red",
    band_color: str = "rgba(128,128,128,0.3)",
    mean_color: str = "black",
    height_per_plot: int = 350,
    forecast: Optional[pd.DataFrame] = None,
    smooth: bool = False,
) -> go.Figure:
    """Generate a 5‑year band plot for one or more columns of a DataFrame.

    This function produces an interactive Plotly figure comprised of one
    subplot per data column (excluding the provided date column).  For
    each series it draws a shaded band between the minimum and maximum
    values observed on each day of the year across the previous five
    calendar years, a solid black line for the five‑year average, and
    lines for the current and specified number of prior years.  The x‑axis
    corresponds to the day‑of‑year and is converted to actual dates for
    the current year to improve readability.

    Parameters
    ----------
    date : Union[str, pd.Series, Iterable]
        Column name, pandas Series, or iterable containing the date for
        each observation.  These will be converted to ``datetime64``.
    df : pd.DataFrame
        DataFrame containing the data columns.  If the column name
        specified in ``date`` exists in ``df``, it will be used as the
        date column; otherwise ``date`` must be provided separately.
    prior_year_lines : int, default 1
        Number of individual prior years immediately preceding the current
        year to draw as separate lines.  Set to zero to suppress
        individual prior year lines.
    current_year_color : str, default "blue"
        Colour for the current year line.
    prior_year_color : str, default "red"
        Colour for the most recent prior year line.  When more than one
        prior year line is requested, colours for additional years will be
        selected automatically from the Plotly qualitative palette.
    band_color : str, default ``"rgba(128,128,128,0.3)"``
        Colour (with alpha channel) for the five‑year minimum–maximum band.
    mean_color : str, default "black"
        Colour for the five‑year average line.
    height_per_plot : int, default 350
        Vertical size allocated to each subplot in pixels.  The total
        figure height will be ``height_per_plot * n_cols``.

    forecast : pandas.DataFrame or None, default None
        Optional DataFrame containing forecast values for the same
        columns in ``df`` (excluding the date column).  The date column
        should have the same name as ``date`` if a string is given; it
        will be converted to ``datetime`` and used to align forecasts
        with the plotted time axis.  When provided, two dotted lines are
        drawn: a blue dotted line for the current‐year forecast and a
        yellow dotted line for the next‐year forecast.  Forecasts for
        years beyond ``current_year + 1`` are ignored.  If no forecast
        values exist for either year, no dotted line is plotted.
    smooth : bool, default False
        If ``True``, apply cubic spline smoothing to the mean, current
        year, prior years and forecast lines.  Smoothing uses SciPy’s
        ``make_interp_spline`` function if available; if SciPy cannot
        be imported or the data length is insufficient, the raw data
        will be used instead.  The shaded band is never smoothed.

    Returns
    -------
    plotly.graph_objects.Figure
        The resulting Plotly figure with one subplot per column.

    Examples
    --------
    Create a band plot for all data columns in a DataFrame ``df`` with a
    ``'date'`` column:

    >>> from analysis3054 import five_year_plot
    >>> fig = five_year_plot(date='date', df=df)
    >>> fig.show()
    """
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        date_series = df[date]
    else:
        date_series = pd.Series(date)
    # Identify numeric columns to plot: exclude the date column if present
    # and any non‑numeric columns
    cols_to_plot = []
    for col in df.columns:
        if col == date:
            continue
        # only plot numeric dtypes
        if pd.api.types.is_numeric_dtype(df[col]):
            cols_to_plot.append(col)
    if not cols_to_plot:
        raise ValueError("No numeric data columns found to plot")

    # Determine height based on number of subplots
    n_plots = len(cols_to_plot)
    total_height = height_per_plot * n_plots

    # Determine the palette for additional prior years
    palette = list(qcolors.Dark24)
    # Remove current and first prior year colours if they exist in the palette
    # to prevent duplication; we'll cycle through the remainder
    def _cycle_colors(exclude: List[str], n: int) -> List[str]:
        available = [c for c in palette if c.lower() not in [e.lower() for e in exclude]]
        if not available:
            # fallback palette if all colours are excluded
            available = list(qcolors.Set1)
        # cycle or truncate to n
        repeats = (n + len(available) - 1) // len(available)
        colors = (available * repeats)[:n]
        return colors

    # Set up the figure with subplots
    fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.08)

    # Convert dates to datetime and compute current year
    dt = pd.to_datetime(date_series)
    if dt.empty:
        raise ValueError("Date series is empty")
    current_year = int(dt.dt.year.max())

    # Determine the years available in the dataset
    years_available = np.sort(dt.dt.year.unique())

    # Determine the range of historical years for the band: last five years preceding current_year
    historical_years = [y for y in range(current_year - 5, current_year) if y in years_available]
    if not historical_years:
        raise ValueError(
            "Insufficient historical data: no years found in the five years preceding the current year"
        )

    # Determine which prior years to draw individually
    prior_years = [y for y in range(current_year - 1, current_year - prior_year_lines - 1, -1) if y in years_available]

    # Precompute a mapping from day-of-year to actual dates for the current year
    # We'll use January 1 of current_year and add day-of-year minus 1
    start_of_year = pd.Timestamp(year=current_year, month=1, day=1)
    # We'll compute for days 1..366; but we only need the days present in data
    # We'll create a dictionary for quick lookup
    def day_to_date(dayofyear: int) -> pd.Timestamp:
        return start_of_year + pd.to_timedelta(dayofyear - 1, unit='D')

    # Helper for smoothing line data.  When ``smooth`` is True and SciPy is
    # available, this function returns a higher‑resolution set of x
    # timestamps and corresponding y values using a cubic spline.  If
    # smoothing is disabled, SciPy is unavailable or fewer than four
    # points are provided, the original data are returned unchanged.
    def _smooth_xy(x_dates: List[pd.Timestamp], y_vals: Iterable, n_points: Optional[int] = None):
        # ``smooth`` and ``make_interp_spline`` are closed over from the
        # outer scope.  ``n_points`` allows overriding the number of
        # interpolated points; if None, a heuristic based on the original
        # length is used.
        if not smooth or make_interp_spline is None:
            # return as arrays for consistency
            return x_dates, np.array(list(y_vals), dtype=float)
        # Need at least four data points to perform a cubic spline
        if len(x_dates) < 4:
            return x_dates, np.array(list(y_vals), dtype=float)
        # Convert datetime to ordinal floats for interpolation
        x_numeric = np.array([d.toordinal() for d in x_dates], dtype=float)
        y_array = np.array(list(y_vals), dtype=float)
        # Sort by x to ensure monotonicity
        order = np.argsort(x_numeric)
        x_sorted = x_numeric[order]
        y_sorted = y_array[order]
        # Determine number of points for interpolation
        if n_points is None:
            # At least triple the original number of points, but not
            # fewer than 200 to provide a smooth appearance
            n_points = max(len(x_sorted) * 3, 200)
        # Create new x grid spanning the original range
        x_new = np.linspace(x_sorted[0], x_sorted[-1], n_points)
        try:
            spline = make_interp_spline(x_sorted, y_sorted, k=3)
            y_new = spline(x_new)
        except Exception:
            # On failure (e.g. singular matrix), fall back to original
            return x_dates, y_array
        # Convert back to timestamps, accounting for fractional days
        x_new_dates: List[pd.Timestamp] = []
        for xi in x_new:
            integer_part = int(np.floor(xi))
            fractional_part = xi - integer_part
            dt = datetime.fromordinal(integer_part) + timedelta(days=fractional_part)
            x_new_dates.append(pd.Timestamp(dt))
        return x_new_dates, y_new

    # Build each subplot
    for idx, col in enumerate(cols_to_plot, start=1):
        # Prepare data for this column
        series = df[col]
        prepared = _prepare_time_series(dt, series)

        # Historical data for band and mean
        hist_df = prepared[prepared['year'].isin(historical_years)].copy()
        if hist_df.empty:
            raise ValueError(f"No historical data found for column '{col}' in years {historical_years}")
        grouped = hist_df.groupby('dayofyear')['value']
        band_min = grouped.min()
        band_max = grouped.max()
        band_mean = grouped.mean()
        # Align index for dayofyear
        band_index = band_min.index.astype(int)
        # Convert dayofyear to dates for x axis
        x_dates = [day_to_date(day) for day in band_index]

        # Add band: we need two traces, first for max, then for min with fill to next y
        # We'll show legend only on first plot
        show_legend = idx == 1
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=band_max.values,
                name='5‑year max',
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=band_min.values,
                name='5‑year range',
                fill='tonexty',
                fillcolor=band_color,
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip',
                showlegend=show_legend,
            ),
            row=idx,
            col=1,
        )
        # Add mean line (optionally smoothed)
        mean_x_vals = x_dates
        mean_y_vals = band_mean.values
        # Apply smoothing if requested
        mean_x_smooth, mean_y_smooth = _smooth_xy(mean_x_vals, mean_y_vals)
        fig.add_trace(
            go.Scatter(
                x=mean_x_smooth,
                y=mean_y_smooth,
                name='5‑year average',
                mode='lines',
                line=dict(color=mean_color, width=2, dash='dash'),
                showlegend=show_legend,
            ),
            row=idx,
            col=1,
        )

        # Add current year line (optionally smoothed)
        cur_df = prepared[prepared['year'] == current_year]
        if not cur_df.empty:
            cur_grouped = cur_df.groupby('dayofyear')['value'].mean()
            cur_x = [day_to_date(int(day)) for day in cur_grouped.index]
            cur_y = cur_grouped.values
            # Apply smoothing if requested
            cur_x_smooth, cur_y_smooth = _smooth_xy(cur_x, cur_y)
            fig.add_trace(
                go.Scatter(
                    x=cur_x_smooth,
                    y=cur_y_smooth,
                    name=f"{current_year}",
                    mode='lines',
                    line=dict(color=current_year_color, width=2),
                    showlegend=show_legend,
                ),
                row=idx,
                col=1,
            )

        # Add prior year lines
        # assign colours: first prior year uses prior_year_color; subsequent use palette
        additional_prior_years = prior_years[1:] if len(prior_years) > 1 else []
        additional_colours = _cycle_colors([current_year_color, prior_year_color], len(additional_prior_years))
        colour_map = {}
        if prior_years:
            # first prior year
            year = prior_years[0]
            year_df = prepared[prepared['year'] == year]
            if not year_df.empty:
                year_grouped = year_df.groupby('dayofyear')['value'].mean()
                year_x = [day_to_date(int(day)) for day in year_grouped.index]
                year_y = year_grouped.values
                # Apply smoothing if requested
                year_x_smooth, year_y_smooth = _smooth_xy(year_x, year_y)
                fig.add_trace(
                    go.Scatter(
                        x=year_x_smooth,
                        y=year_y_smooth,
                        name=str(year),
                        mode='lines',
                        line=dict(color=prior_year_color, width=2, dash='dot'),
                        showlegend=show_legend,
                    ),
                    row=idx,
                    col=1,
                )
        # Additional prior years beyond the first
        for j, year in enumerate(additional_prior_years):
            year_df = prepared[prepared['year'] == year]
            if year_df.empty:
                continue
            year_grouped = year_df.groupby('dayofyear')['value'].mean()
            year_x = [day_to_date(int(day)) for day in year_grouped.index]
            year_y = year_grouped.values
            # Apply smoothing if requested
            year_x_smooth, year_y_smooth = _smooth_xy(year_x, year_y)
            fig.add_trace(
                go.Scatter(
                    x=year_x_smooth,
                    y=year_y_smooth,
                    name=str(year),
                    mode='lines',
                    line=dict(color=additional_colours[j], width=2, dash='dot'),
                    showlegend=show_legend,
                ),
                row=idx,
                col=1,
            )

        # Plot forecast lines when provided
        if forecast is not None and isinstance(forecast, pd.DataFrame):
            # Determine the date series in the forecast DataFrame.  Only
            # proceed when the date column matches the input and the
            # forecast contains the current column.
            f_date_series: Optional[pd.Series] = None
            if isinstance(date, str) and date in forecast.columns:
                f_date_series = forecast[date]
            if f_date_series is not None and col in forecast.columns:
                f_dt = pd.to_datetime(f_date_series)
                # We only draw forecasts for the current and next year
                for f_year in [current_year, current_year + 1]:
                    mask = f_dt.dt.year == f_year
                    if mask.any():
                        f_year_values = forecast.loc[mask, col]
                        f_year_dates = f_dt[mask]
                        # Prepare the series for this year
                        f_prepared = _prepare_time_series(f_year_dates, f_year_values)
                        if not f_prepared.empty:
                            f_grouped = f_prepared.groupby('dayofyear')['value'].mean()
                            f_x = [day_to_date(int(day)) for day in f_grouped.index]
                            f_y = f_grouped.values
                            f_x_smooth, f_y_smooth = _smooth_xy(f_x, f_y)
                            # Colour and label selection
                            if f_year == current_year:
                                forecast_color = current_year_color
                            else:
                                forecast_color = "yellow"
                            trace_name = f"{f_year} forecast"
                            fig.add_trace(
                                go.Scatter(
                                    x=f_x_smooth,
                                    y=f_y_smooth,
                                    name=trace_name,
                                    mode='lines',
                                    line=dict(color=forecast_color, width=2, dash='dot'),
                                    showlegend=show_legend,
                                ),
                                row=idx,
                                col=1,
                            )

        # Set subplot title as column name
        fig.update_yaxes(title_text=col, row=idx, col=1)

    # Format x axis across all subplots
    fig.update_xaxes(
        title_text=f"Date ({current_year})",
        tickformat="%b %d",
        dtick="M1",
        row=n_plots,
        col=1,
    )

    # Configure the overall layout
    fig.update_layout(
        height=total_height,
        showlegend=True,
        legend_title_text="Legend",
        hovermode="x unified",
        margin=dict(t=40, b=40, l=60, r=40),
    )
    return fig
