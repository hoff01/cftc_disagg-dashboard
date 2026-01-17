"""
Statistical analysis utilities for time‑series data.

This module introduces functions for deeper statistical exploration of
commodity price series, such as cross‑correlation analysis and
principal component analysis (PCA).  The functions here are intended
to assist analysts in identifying lag relationships and common
underlying factors among multiple series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def cross_correlation_plot(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    series1: str,
    series2: str,
    max_lag: int = 20,
    normalize: bool = True,
) -> go.Figure:
    """Compute and plot the cross‑correlation function between two series.

    Cross‑correlation measures the similarity between two time series as
    a function of the lag of one relative to the other.  This function
    computes the cross‑correlation for lags from ``-max_lag`` to
    ``+max_lag`` and displays the results as a bar chart.  Positive lags
    indicate that ``series2`` is lagged behind ``series1``.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment.  Used solely for length
        and ordering.
    df : pandas.DataFrame
        DataFrame containing the two series.  Both must be numeric.
    series1, series2 : str
        Names of the two series to analyse.
    max_lag : int, default 20
        Maximum lag (in number of observations) to compute.
    normalize : bool, default True
        Whether to normalize the correlation by the product of standard
        deviations.  When ``True``, the result is the cross‑correlation
        function; otherwise, raw cross‑covariances are plotted.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of cross‑correlation values at each lag.

    Notes
    -----
    Missing values are forward‑filled and backward‑filled before
    calculation.  The correlation at lag ``k`` is defined as the
    correlation between ``series1[t]`` and ``series2[t−k]``.
    """
    if series1 not in df.columns or series2 not in df.columns:
        raise KeyError("Specified series not found in DataFrame")
    # Extract and preprocess series
    x = pd.to_numeric(df[series1], errors='coerce').ffill().bfill().values
    y = pd.to_numeric(df[series2], errors='coerce').ffill().bfill().values
    n = len(x)
    if n != len(y):
        raise ValueError("Series must have the same length")
    # Subtract means
    x_cent = x - x.mean()
    y_cent = y - y.mean()
    # Compute full cross correlation via convolution
    corr_full = np.correlate(x_cent, y_cent, mode='full')
    # Define lags
    lags = np.arange(-n + 1, n)
    # Extract desired range
    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr_vals = corr_full[mask]
    lags_sel = lags[mask]
    if normalize:
        denom = np.std(x_cent) * np.std(y_cent) * n
        corr_vals = corr_vals / denom
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags_sel, y=corr_vals, marker=dict(color='blue')))
    fig.update_layout(
        title=f"Cross‑Correlation: {series1} vs {series2}",
        xaxis_title="Lag",
        yaxis_title="Correlation" if normalize else "Cross‑Covariance",
        bargap=0.2,
    )
    return fig


@dataclass
class PCAResult:
    """Result container for principal component analysis.

    Attributes
    ----------
    components : pandas.DataFrame
        DataFrame containing the principal component scores.  Each
        column corresponds to a component.
    explained_variance : np.ndarray
        Array of explained variances (eigenvalues) for each component.
    explained_variance_ratio : np.ndarray
        Proportion of variance explained by each component.
    loadings : pandas.DataFrame
        The PCA loadings (eigenvectors) with variables as rows and
        components as columns.
    """
    components: pd.DataFrame
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    loadings: pd.DataFrame


def pca_decomposition(
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    scale: bool = True,
) -> PCAResult:
    """Perform principal component analysis on selected columns.

    PCA reduces the dimensionality of a dataset by projecting it onto
    orthogonal components that capture the maximum variance.  This
    function standardises the data (zero mean and unit variance) by
    default before applying PCA, which is recommended when variables
    have different scales.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric data to decompose.
    columns : list of str or None, default None
        Columns to include in the PCA.  If ``None``, all numeric
        columns are used.
    n_components : int or None, default None
        Number of principal components to compute.  If ``None``, use
        the minimum of the number of selected columns and the number of
        observations.
    scale : bool, default True
        Whether to standardise the data before PCA.

    Returns
    -------
    PCAResult
        Dataclass containing component scores, explained variances,
        explained variance ratios, and loadings.

    Notes
    -----
    scikit‑learn's :class:`sklearn.decomposition.PCA` is used under the
    hood.  If the package is not installed, an ImportError is raised.
    """
    from sklearn.decomposition import PCA
    # Identify columns
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        raise ValueError("No numeric columns selected for PCA")
    X = df[columns].astype(float)
    # Standardise
    if scale:
        X = (X - X.mean()) / X.std(ddof=0)
    # Determine number of components
    if n_components is None or n_components > min(X.shape):
        n_components = min(X.shape)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    # Loadings: eigenvectors
    loadings = pd.DataFrame(pca.components_.T, index=columns, columns=[f'PC{i+1}' for i in range(n_components)])
    comp_df = pd.DataFrame(components, index=df.index, columns=[f'PC{i+1}' for i in range(n_components)])
    return PCAResult(
        components=comp_df,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        loadings=loadings,
    )


def partial_autocorrelation_plot(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    column: str,
    nlags: int = 20,
    alpha: Optional[float] = None,
    method: str = 'yw',
) -> go.Figure:
    """Compute and plot the partial autocorrelation function (PACF) for a series.

    The PACF measures the correlation between a series and its lagged
    version after removing the influence of intervening lags.  It is
    useful for identifying the order of autoregressive (AR) models and
    understanding the dependence structure of a time series.

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment.  Used solely for length
        and ordering.
    df : pandas.DataFrame
        DataFrame containing the target series.  Must be numeric.
    column : str
        Name of the series to analyse.
    nlags : int, default 20
        Number of lags to compute.
    alpha : float or None, default None
        Significance level for confidence intervals.  If provided,
        shaded regions are added to the plot representing ±1.96/√N
        bands (for large samples).  This parameter is for visual
        reference only; it does not alter the PACF calculation.
    method : {'yw','ols'}, default 'yw'
        Method used to compute PACF when Statsmodels is not
        available.  'yw' employs the Yule–Walker equations to solve
        for PACF coefficients recursively, while 'ols' fits
        successive autoregressions via ordinary least squares.  If
        Statsmodels is installed, its PACF implementation is
        preferred regardless of this setting.

    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart of partial autocorrelations at each lag.

    Notes
    -----
    This function requires the `statsmodels` package for the most
    accurate computation.  If `statsmodels` is not available, a
    fallback implementation is used based on the chosen method.  The
    fallback may differ slightly from the Statsmodels result for
    higher lags or small samples.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    # Extract and preprocess series
    x = pd.to_numeric(df[column], errors='coerce').ffill().bfill().values
    n = len(x)
    # Try Statsmodels PACF
    pacf_vals = None
    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf
        pacf_vals = sm_pacf(x, nlags=nlags, method='yw')  # statsmodels handles method internally
    except Exception:
        pass
    if pacf_vals is None:
        # Fallback implementation
        def _pacf_yw(x: np.ndarray, nlags: int) -> np.ndarray:
            # Yule–Walker recursion
            pacf = np.zeros(nlags + 1)
            pacf[0] = 1.0
            if nlags == 0:
                return pacf
            # Autocorrelation function
            acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')[n - 1:] / n
            # Initial values
            phi = np.zeros(nlags + 1)
            phi[1] = acf[1] / acf[0]
            pacf[1] = phi[1]
            var = acf[0] * (1 - phi[1] ** 2)
            # Recursively compute higher orders
            for k in range(2, nlags + 1):
                sum_phi = np.sum([phi[j] * acf[k - j] for j in range(1, k)])
                phi[k] = (acf[k] - sum_phi) / var
                # Update previous coefficients
                phi_prev = phi[1:k].copy()
                for j in range(1, k):
                    phi[j] = phi_prev[j - 1] - phi[k] * phi_prev[k - j - 1]
                pacf[k] = phi[k]
                var *= (1 - phi[k] ** 2)
            return pacf
        def _pacf_ols(x: np.ndarray, nlags: int) -> np.ndarray:
            # Fit successive AR models via ordinary least squares
            pacf = np.zeros(nlags + 1)
            pacf[0] = 1.0
            for k in range(1, nlags + 1):
                # Build lagged matrix
                Y = x[k:]
                X = np.column_stack([x[k - i - 1: n - i - 1] for i in range(k)])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                pacf[k] = beta[-1]
            return pacf
        if method == 'ols':
            pacf_vals = _pacf_ols(x, nlags)
        else:
            pacf_vals = _pacf_yw(x, nlags)
    # Generate lags and plot
    lags = np.arange(0, nlags + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags[1:], y=pacf_vals[1:], marker=dict(color='green')))
    fig.update_layout(
        title=f"Partial Autocorrelation: {column}",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation",
        bargap=0.2,
    )
    # Add confidence intervals if requested
    if alpha is not None and n > 0:
        # Approximate 95% confidence interval ±1.96/√n or general from alpha
        from scipy import stats as sc_stats  # Local import to avoid global dependency
        z = sc_stats.norm.ppf(1 - alpha / 2.0)
        bound = z / np.sqrt(n)
        fig.add_hrect(y0=-bound, y1=bound, fillcolor='rgba(200,200,200,0.2)', line_width=0)
    return fig

# ---------------------------------------------------------------------------
# Granger causality analysis
# ---------------------------------------------------------------------------

def granger_causality_matrix(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    columns: Optional[List[str]] = None,
    maxlag: int = 4,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute a pairwise Granger causality matrix for multiple series.

    For each ordered pair of series ``(cause, effect)``, this
    function performs Granger causality tests up to ``maxlag`` and
    records the smallest p‑value across all tested lags.  A low
    p‑value suggests that the ``cause`` series contains information
    useful for predicting the ``effect`` series.  The resulting
    DataFrame has rows representing potential causes and columns
    representing potential effects (i.e., the p‑value of column j
    Granger‑causing column i is at position ``pvalues.loc[j, i]``).

    Parameters
    ----------
    date : str or pandas.Series
        Date column or series for alignment (unused in calculation).
    df : pandas.DataFrame
        DataFrame containing the data.
    columns : list of str or None, default None
        Columns to include in the analysis.  If ``None``, all
        numeric columns except the date column are used.
    maxlag : int, default 4
        Maximum number of lags to test for Granger causality.
    verbose : bool, default False
        If True, print detailed output from the underlying
        ``statsmodels`` function.  Otherwise, suppress printing.

    Returns
    -------
    pandas.DataFrame
        DataFrame of p‑values.  Entry ``(i, j)`` contains the
        minimal p‑value for the null hypothesis that series ``i`` does
        not Granger‑cause series ``j``.

    Raises
    ------
    ImportError
        If the ``statsmodels`` package is not installed.
    ValueError
        If fewer than two columns are available for analysis.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        raise ImportError(
            "statsmodels is required for Granger causality analysis. "
            "Please install statsmodels to use this function."
        ) from e
    if columns is None:
        columns = [c for c in df.columns if c != date and pd.api.types.is_numeric_dtype(df[c])]
    if len(columns) < 2:
        raise ValueError("At least two numeric columns are required for Granger causality analysis")
    pvals = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns, dtype=float)
    for cause in columns:
        for effect in columns:
            if cause == effect:
                pvals.loc[cause, effect] = np.nan
                continue
            data_pair = df[[effect, cause]].dropna()
            if data_pair.shape[0] <= maxlag:
                pvals.loc[cause, effect] = np.nan
                continue
            try:
                gc_res = grangercausalitytests(data_pair, maxlag=maxlag, verbose=verbose)
                p_values = []
                for lag in range(1, maxlag + 1):
                    try:
                        p_values.append(gc_res[lag][0]['ssr_ftest'][1])
                    except Exception:
                        pass
                pvals.loc[cause, effect] = np.nanmin(p_values) if p_values else np.nan
            except Exception:
                pvals.loc[cause, effect] = np.nan
    return pvals
