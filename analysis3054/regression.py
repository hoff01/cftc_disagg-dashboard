"""
Regression and correlation utilities.

This module provides functions for linear regression, rolling
correlation analysis and structural break testing, leveraging
`statsmodels` for statistical estimation.  The goal is to simplify
common modelling tasks encountered in commodity analytics and risk
management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "ols_regression",
    "RegressionResult",
    "rolling_correlation",
    "RollingCorrelationResult",
    "cusum_olsresid_test",
    "CusumTestResult",
]


@dataclass
class RegressionResult:
    """Container for ordinary least squares regression output.

    Attributes
    ----------
    coefficients : pandas.Series
        Estimated coefficients (including the intercept if added).
    std_errors : pandas.Series
        Standard errors of the coefficients.
    t_values : pandas.Series
        t‑statistics for the coefficients.
    p_values : pandas.Series
        p‑values associated with the t‑statistics.
    r2 : float
        Coefficient of determination (R‑squared).
    adj_r2 : float
        Adjusted R‑squared.
    predictions : pandas.Series or None
        Fitted values for the dependent variable if
        ``return_predictions`` is True; otherwise None.
    """
    coefficients: pd.Series
    std_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    r2: float
    adj_r2: float
    predictions: Optional[pd.Series] = None


def ols_regression(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    y: str,
    X: List[str],
    add_constant: bool = True,
    return_predictions: bool = False,
) -> RegressionResult:
    """Fit an ordinary least squares (OLS) regression.

    This function wraps ``statsmodels.api.OLS`` to fit a linear
    regression model of the form ``y ~ X``.  It extracts the
    estimated coefficients, standard errors, t‑statistics and
    p‑values, along with R‑squared measures.  Optionally, it returns
    the fitted values of the dependent variable.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates.  Used only for
        alignment when returning fitted values.
    df : pandas.DataFrame
        DataFrame containing the dependent and independent variables.
    y : str
        Name of the dependent variable column.
    X : list of str
        Names of the independent variable columns.  If
        ``add_constant`` is True, a constant will be added
        internally.
    add_constant : bool, default True
        Whether to add an intercept term to the regression.
    return_predictions : bool, default False
        If True, return the fitted values as a pandas Series with
        the same index as the input DataFrame.

    Returns
    -------
    RegressionResult
        Dataclass containing regression statistics and optionally
        predictions.

    Raises
    ------
    ImportError
        If the ``statsmodels`` package is not available.
    KeyError
        If ``y`` or any of the ``X`` columns are not in ``df``.
    """
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise ImportError(
            "statsmodels is required for ols_regression. "
            "Please install statsmodels to use this function."
        ) from e
    # Validate columns
    missing_cols = [c for c in [y] + X if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in DataFrame")
    # Prepare design matrix and response
    y_vec = pd.to_numeric(df[y], errors='coerce').astype(float)
    X_mat = df[X].apply(pd.to_numeric, errors='coerce').astype(float)
    if add_constant:
        X_mat = sm.add_constant(X_mat, has_constant='add')
    # Fit model
    model = sm.OLS(y_vec, X_mat, missing='drop')
    res = model.fit()
    # Extract results
    params = res.params
    std_err = res.bse
    t_vals = res.tvalues
    p_vals = res.pvalues
    r2 = res.rsquared
    adj_r2 = res.rsquared_adj
    predictions = None
    if return_predictions:
        # Align with date index if provided as a column name or series
        if isinstance(date, str) and date in df.columns:
            index = pd.to_datetime(df[date])
        else:
            index = pd.to_datetime(date)
        # Generate in-sample fitted values
        preds = res.predict(X_mat)
        predictions = pd.Series(preds, index=index, name=y)
    return RegressionResult(
        coefficients=params,
        std_errors=std_err,
        t_values=t_vals,
        p_values=p_vals,
        r2=r2,
        adj_r2=adj_r2,
        predictions=predictions,
    )


@dataclass
class RollingCorrelationResult:
    """Container for rolling correlation output.

    Attributes
    ----------
    correlation : pandas.Series
        Rolling correlation values indexed by the date column.
    """
    correlation: pd.Series


def rolling_correlation(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    series1: str,
    series2: str,
    window: int = 20,
) -> RollingCorrelationResult:
    """Compute the rolling Pearson correlation between two series.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates used as the index for
        the output.
    df : pandas.DataFrame
        DataFrame containing the two series.
    series1 : str
        Name of the first series column.
    series2 : str
        Name of the second series column.
    window : int, default 20
        Size of the rolling window used to compute the correlation.

    Returns
    -------
    RollingCorrelationResult
        Dataclass containing the rolling correlation values.
    """
    # Validate columns
    for col in [series1, series2]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
    if isinstance(date, str):
        if date not in df.columns:
            raise KeyError(f"Date column '{date}' not found in DataFrame")
        index = pd.to_datetime(df[date])
    else:
        index = pd.to_datetime(date)
    s1 = pd.to_numeric(df[series1], errors='coerce').ffill().bfill().astype(float)
    s2 = pd.to_numeric(df[series2], errors='coerce').ffill().bfill().astype(float)
    corr = s1.rolling(window=window, min_periods=1).corr(s2)
    corr_series = pd.Series(corr.values, index=index, name=f"corr_{series1}_{series2}")
    return RollingCorrelationResult(correlation=corr_series)


@dataclass
class CusumTestResult:
    """Container for the CUSUM test results.

    Attributes
    ----------
    statistic : float
        Test statistic for the CUSUM test.
    p_value : float
        Two‑sided p‑value indicating the probability of a structural
        break under the null hypothesis of no change.
    """
    statistic: float
    p_value: float


def cusum_olsresid_test(
    date: Union[str, pd.Series],
    df: pd.DataFrame,
    *,
    y: str,
    X: List[str],
    alpha: float = 0.05,
) -> CusumTestResult:
    """Perform the CUSUM test for parameter stability of OLS residuals.

    This test evaluates whether the coefficients of a linear
    regression model remain stable over time.  It is based on the
    cumulative sum of OLS residuals and compares them to critical
    values.  A significant result suggests structural change.

    Parameters
    ----------
    date : str or pandas.Series
        Column name or series containing dates.  Used to align
        results (unused otherwise).
    df : pandas.DataFrame
        DataFrame containing the data.
    y : str
        Name of the dependent variable column.
    X : list of str
        Names of independent variable columns.  A constant term is
        added internally.
    alpha : float, default 0.05
        Significance level used to interpret the p‑value.  Not used
        directly in the test calculation but informative for users.

    Returns
    -------
    CusumTestResult
        Dataclass containing the test statistic and p‑value.

    Raises
    ------
    ImportError
        If the ``statsmodels`` package is not installed.
    KeyError
        If ``y`` or any ``X`` columns are not in the DataFrame.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
    except Exception as e:
        raise ImportError(
            "statsmodels is required for cusum_olsresid_test. "
            "Please install statsmodels to use this function."
        ) from e
    # Validate columns
    missing_cols = [c for c in [y] + X if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns {missing_cols} not found in DataFrame")
    # Prepare design matrix and response
    y_vec = pd.to_numeric(df[y], errors='coerce').astype(float)
    X_mat = df[X].apply(pd.to_numeric, errors='coerce').astype(float)
    X_mat = sm.add_constant(X_mat, has_constant='add')
    # Fit OLS
    model = sm.OLS(y_vec, X_mat, missing='drop')
    res = model.fit()
    # Compute CUSUM test statistic and p-value
    # The CUSUM test expects a residual vector, not the results object
    resid = res.resid
    statistic, pval, _ = breaks_cusumolsresid(resid, ddof=len(X) + 1)
    return CusumTestResult(statistic=float(statistic), p_value=float(pval))