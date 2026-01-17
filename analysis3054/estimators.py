"""
Advanced estimators for conditional prediction and load‑based forecasting.

This module defines a collection of PhD‑level prediction routines
designed for the commodity markets.  These functions build on
statistical learning theory to infer conditional relationships
between variables, provide uncertainty quantification, and incorporate
exogenous inputs such as forecasted load profiles.  Each function
accepts pandas DataFrame inputs and returns structured results with
both point estimates and confidence intervals.

Included are:

* :func:`bayesian_linear_estimator` – Estimate a scalar response
  conditional on a predictor via Bayesian linear regression, yielding
  posterior means and credible intervals.

* :func:`gaussian_process_estimator` – Use Gaussian process
  regression to model non‑linear relationships between two variables
  and produce point predictions with uncertainty bounds.

* :func:`load_based_forecast` – Predict a consumption variable using
  forecasted load, historical load and burn (ratio) data, and current
  observations.  A Gaussian process is fitted to the historical
  relationship between load and burn ratio, and forecasts are
  generated for the entire span of the provided load forecast.  The
  function gracefully handles high‑frequency input data (e.g. five
  minute intervals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

@dataclass
class BayesianLinearResult:
    """Result of the Bayesian linear estimator.

    Attributes
    ----------
    predictions : pandas.DataFrame
        DataFrame containing the predicted mean and confidence
        intervals for each input value.  Columns are ``'mean'``,
        ``'lower'`` and ``'upper'``.
    model : object
        Fitted Bayesian Ridge model from scikit‑learn.
    """

    predictions: pd.DataFrame
    model: object


def bayesian_linear_estimator(
    x: Union[str, Iterable],
    y: Union[str, Iterable],
    df: pd.DataFrame,
    *,
    new_x: Union[Iterable, pd.Series, np.ndarray],
    alpha: float = 1.0,
    lambda_: float = 1.0,
    ci: float = 0.95,
) -> BayesianLinearResult:
    """Estimate ``y`` given ``x`` using Bayesian linear regression.

    This function fits a Bayesian linear model to the relationship
    between a predictor ``x`` and response ``y`` in the provided
    DataFrame.  It uses scikit‑learn's :class:`BayesianRidge` to
    approximate the posterior distribution of the regression
    coefficients and noise variance.  Predictions are computed for
    ``new_x`` values, along with credible intervals derived from the
    posterior predictive distribution.

    Parameters
    ----------
    x : str or iterable
        Name of the predictor column in ``df`` or a standalone array
        of predictor values.
    y : str or iterable
        Name of the response column in ``df`` or a standalone array
        of response values.
    df : pandas.DataFrame
        DataFrame containing the data.  Only used when ``x`` and ``y``
        are specified as column names.
    new_x : array‑like
        Values of the predictor for which to compute predictions and
        credible intervals.
    alpha : float, default 1.0
        Hyperparameter for the Bayesian prior over the weights in
        :class:`BayesianRidge`.  Higher values correspond to stronger
        regularisation.
    lambda_ : float, default 1.0
        Hyperparameter for the Bayesian prior over the noise variance.
    ci : float, default 0.95
        Credible interval level.  For example, ``0.95`` yields 95 %
        intervals (2.5th and 97.5th percentiles).

    Returns
    -------
    BayesianLinearResult
        Object containing the predictions and the fitted model.

    Notes
    -----
    The credible intervals are derived using the predictive mean and
    variance returned by :meth:`BayesianRidge.predict` with
    ``return_std=True``.  The normality assumption inherent in the
    BayesianRidge model is used to compute the quantiles.
    """
    from sklearn.linear_model import BayesianRidge
    from scipy.stats import norm
    # Extract data arrays
    if isinstance(x, str):
        if x not in df.columns:
            raise KeyError(f"Predictor column '{x}' not found in DataFrame")
        x_vals = df[[x]].astype(float).values
    else:
        x_vals = np.asarray(x, dtype=float).reshape(-1, 1)
    if isinstance(y, str):
        if y not in df.columns:
            raise KeyError(f"Response column '{y}' not found in DataFrame")
        y_vals = df[y].astype(float).values
    else:
        y_vals = np.asarray(y, dtype=float)
    # Fit Bayesian ridge regression
    model = BayesianRidge(alpha_init=alpha, lambda_init=lambda_)
    model.fit(x_vals, y_vals)
    # Prepare new_x
    new_x_arr = np.asarray(new_x, dtype=float).reshape(-1, 1)
    mean_pred, std_pred = model.predict(new_x_arr, return_std=True)
    # Compute credible intervals assuming Gaussian predictive distribution
    z = norm.ppf(0.5 + ci / 2)
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred
    pred_df = pd.DataFrame({'mean': mean_pred, 'lower': lower, 'upper': upper}, index=pd.Index(new_x_arr.flatten(), name='x'))
    return BayesianLinearResult(predictions=pred_df, model=model)


@dataclass
class GaussianProcessResult:
    """Result of the Gaussian process estimator.

    Attributes
    ----------
    predictions : pandas.DataFrame
        DataFrame with columns ``'mean'``, ``'lower'`` and ``'upper'``
        for the predicted response.  The index corresponds to the
        provided ``new_x`` values.
    model : object
        The fitted :class:`sklearn.gaussian_process.GaussianProcessRegressor`.
    """

    predictions: pd.DataFrame
    model: object


def gaussian_process_estimator(
    x: Union[str, Iterable],
    y: Union[str, Iterable],
    df: pd.DataFrame,
    *,
    new_x: Union[Iterable, pd.Series, np.ndarray],
    ci: float = 0.95,
    kernel: Optional[object] = None,
    alpha: float = 1e-6,
    random_state: Optional[int] = None,
) -> GaussianProcessResult:
    """Estimate ``y`` given ``x`` using Gaussian process regression.

    Gaussian process (GP) regression is a non‑parametric Bayesian
    approach that can model complex, non‑linear relationships with
    uncertainty quantification.  This function fits a GP model to
    ``x`` and ``y`` in the provided DataFrame and computes predictions
    with confidence intervals for ``new_x`` values.

    Parameters
    ----------
    x, y, df : see :func:`bayesian_linear_estimator`
    new_x : array‑like
        Predictor values at which to compute GP predictions.
    ci : float, default 0.95
        Confidence interval level.
    kernel : object or None, default None
        Kernel object to use in the GP.  If ``None``, a default kernel
        consisting of a constant kernel multiplied by an RBF kernel is
        used.
    alpha : float, default 1e-6
        Added to the diagonal of the kernel matrix during fitting for
        numerical stability.
    random_state : int or None, default None
        Random state passed to the GP regressor for reproducibility.

    Returns
    -------
    GaussianProcessResult
        Object containing predictions and the fitted GP model.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from scipy.stats import norm
    # Extract arrays
    if isinstance(x, str):
        if x not in df.columns:
            raise KeyError(f"Predictor column '{x}' not found in DataFrame")
        x_vals = df[[x]].astype(float).values
    else:
        x_vals = np.asarray(x, dtype=float).reshape(-1, 1)
    if isinstance(y, str):
        if y not in df.columns:
            raise KeyError(f"Response column '{y}' not found in DataFrame")
        y_vals = df[y].astype(float).values
    else:
        y_vals = np.asarray(y, dtype=float)
    # Define kernel if not provided
    if kernel is None:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    # Skip hyperparameter optimisation for faster execution.  This fixes the
    # kernel parameters at their initial values; set ``optimizer`` to
    # "fmin_l_bfgs_b" externally if optimisation is desired.
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        # Use the default optimizer to fit kernel hyperparameters.
        # This may be slower but typically yields better predictions.
        optimizer='fmin_l_bfgs_b',
        random_state=random_state,
    )
    gp.fit(x_vals, y_vals)
    # New x
    new_x_arr = np.asarray(new_x, dtype=float).reshape(-1, 1)
    mean_pred, std_pred = gp.predict(new_x_arr, return_std=True)
    z = norm.ppf(0.5 + ci / 2)
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred
    pred_df = pd.DataFrame({'mean': mean_pred, 'lower': lower, 'upper': upper}, index=pd.Index(new_x_arr.flatten(), name='x'))
    return GaussianProcessResult(predictions=pred_df, model=gp)


@dataclass
class LoadForecastResult:
    """Result of the load‑based forecast.

    Attributes
    ----------
    forecasts : pandas.DataFrame
        DataFrame indexed by the forecast load dates and containing
        three columns: ``'mean'``, ``'lower'`` and ``'upper'`` for the
        predicted consumption (burn).
    model : object
        Fitted GaussianProcessRegressor used to model the burn/load
        ratio.
    """

    forecasts: pd.DataFrame
    model: object


def load_based_forecast(
    forecast_load_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    date_col: str = 'date',
    load_col: str = 'load',
    burn_col: str = 'burn',
    ci: float = 0.95,
    kernel: Optional[object] = None,
    alpha: float = 1e-6,
    random_state: Optional[int] = None,
) -> LoadForecastResult:
    """Forecast burn (consumption) using forecasted load and historical ratios.

    This function estimates the relationship between load and burn
    (consumption) by fitting a Gaussian process regression model to the
    historical burn/load ratio as a function of load.  The ratio is
    computed as ``burn / load`` for all historical observations with
    non‑zero load.  The GP captures non‑linear dependencies and
    provides predictive distributions.  When applied to the
    ``forecast_load_df``, the model produces a point forecast and
    confidence bounds for burn across the entire forecast horizon.  If
    ``current_df`` contains more recent observations than the
    historical data, these are appended to ensure the model uses the
    latest available information.  High‑frequency data (e.g. five
    minute intervals) are handled transparently, and forecasts extend
    to the last timestamp in ``forecast_load_df``.

    Parameters
    ----------
    forecast_load_df : pandas.DataFrame
        DataFrame containing forecasted load values with a date column
        ``date_col`` and load column ``load_col``.  Predictions will
        be generated for all rows in this DataFrame.
    historical_df : pandas.DataFrame
        DataFrame with historical observations of load and burn.  Must
        include ``date_col``, ``load_col`` and ``burn_col``.
    current_df : pandas.DataFrame
        DataFrame with the most recent observations of load and burn
        (possibly overlapping with ``historical_df``).  These data are
        appended to the historical set before fitting the model.
    date_col, load_col, burn_col : str, default 'date','load','burn'
        Names of the columns containing the date, load and burn values.
    ci : float, default 0.95
        Confidence interval level for the burn forecasts.
    kernel, alpha, random_state : see :func:`gaussian_process_estimator`

    Returns
    -------
    LoadForecastResult
        Dataclass containing the forecasts and the fitted GP model.

    Notes
    -----
    The function drops rows with zero or missing load when computing
    the ratio to avoid division by zero.  If no valid historical
    observations remain, a ValueError is raised.  Forecasts for load
    values outside the range of the training data may carry greater
    uncertainty.
    """
    # Combine historical and current observations
    hist = historical_df[[date_col, load_col, burn_col]].copy()
    curr = current_df[[date_col, load_col, burn_col]].copy()
    combined = pd.concat([hist, curr], axis=0, ignore_index=True)
    # Drop rows with missing or zero load
    combined = combined.dropna(subset=[load_col, burn_col])
    combined = combined[combined[load_col] != 0]
    if combined.empty:
        raise ValueError("No valid historical data with non‑zero load")
    # Compute ratio
    combined['ratio'] = combined[burn_col].astype(float) / combined[load_col].astype(float)
    # Fit GP on load vs ratio
    gp_result = gaussian_process_estimator(
        x=load_col,
        y='ratio',
        df=combined[[load_col, 'ratio']],
        new_x=forecast_load_df[load_col].values,
        ci=ci,
        kernel=kernel,
        alpha=alpha,
        random_state=random_state,
    )
    # Multiply predicted ratio by forecast load to obtain burn
    load_vals = forecast_load_df[load_col].astype(float).values
    mean_burn = gp_result.predictions['mean'].values * load_vals
    lower_burn = gp_result.predictions['lower'].values * load_vals
    upper_burn = gp_result.predictions['upper'].values * load_vals
    forecast_df = pd.DataFrame(
        {
            'mean': mean_burn,
            'lower': lower_burn,
            'upper': upper_burn,
        },
        index=pd.to_datetime(forecast_load_df[date_col]).reset_index(drop=True)
    )
    forecast_df.index.name = date_col
    return LoadForecastResult(forecasts=forecast_df, model=gp_result.model)
