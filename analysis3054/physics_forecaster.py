"""Continuous-time, physics-inspired forecasting utilities.

This module implements a collection of numerical building blocks that
combine multi-resolution wavelet analysis, neural ordinary differential
equations (Neural ODEs), and fractal-dimension filters.  The goal is to
support forecasting workflows that treat time-series data as continuous
signals rather than purely discrete sequences.

Components
----------
* :func:`multi_resolution_wavelet_decompose` – Stationary wavelet
  transform (SWT) using the Symlet-5 wavelet to separate high-frequency
  noise from low-frequency structure.
* :class:`ODEFunc` and :class:`NeuralODEBlock` – A tanh-activated neural
  network that parameterizes a dynamical system, solved with
  ``torchdiffeq.odeint`` for arbitrarily sampled timestamps.
* :class:`SobolevLoss` – Loss that penalizes both value and derivative
  mismatch between predicted and reference trajectories.
* :func:`rolling_hurst_router` – Rolling Hurst exponent to detect
  trending vs. mean-reverting regimes and return numerical weights that
  can gate model pathways.
* Plotting helpers for scalograms, phase portraits, and continuous
  forecast trajectories.

Optional dependencies
---------------------
This module requires ``torch``, ``torchdiffeq`` and ``PyWavelets`` for
full functionality.  Import errors are raised lazily inside functions so
that the rest of the package can still be imported without these heavy
libraries installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

try:  # Optional dependency
    import pywt
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    pywt = None  # type: ignore[assignment]

try:  # Optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_TORCH_AVAILABLE = torch is not None and nn is not None
_MODULE_BASE = nn.Module if _TORCH_AVAILABLE else object

try:  # Optional dependency
    from torchdiffeq import odeint
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    odeint = None  # type: ignore[assignment]


@dataclass
class WaveletDecompositionResult:
    """Container for stationary wavelet decomposition outputs.

    Attributes
    ----------
    approximation : np.ndarray
        The final approximation coefficients (low-frequency content).
    details : List[np.ndarray]
        Detail coefficients for each level, ordered from coarse to fine.
    denoised : np.ndarray
        Reconstructed series after soft-thresholding the detail bands.
    coefficient_matrix : np.ndarray
        Matrix of shape ``(levels + 1, n_samples)`` stacking the
        approximation and each detail band for downstream models.
    """

    approximation: np.ndarray
    details: List[np.ndarray]
    denoised: np.ndarray
    coefficient_matrix: np.ndarray


@dataclass
class DMDResult:
    """Container for Dynamic Mode Decomposition outputs."""

    eigenvalues: np.ndarray
    koopman_operator: np.ndarray
    instability_index: float


class KoopmanSpectralDecomposer:
    """Exact Dynamic Mode Decomposition for regime stability sensing.

    The class implements a numerically stable variant of Exact DMD to
    approximate the Koopman operator of a rolling time-series window.  It
    exposes a single :meth:`decompose` method that returns the dominant
    eigenvalue magnitude – an ``Instability Index`` that signals whether the
    current system is decaying, oscillatory, or explosive.

    Notes
    -----
    * ``|lambda| < 1`` indicates decaying dynamics.
    * ``|lambda| = 1`` indicates marginally stable/oscillatory dynamics.
    * ``|lambda| > 1`` indicates exponential growth/decay (regime break).
    """

    def __init__(self, hankel_rows: int = 10, svd_truncation: Optional[int] = None):
        if hankel_rows < 2:
            raise ValueError("hankel_rows must be at least 2 to form shift pairs")
        if svd_truncation is not None and svd_truncation < 1:
            raise ValueError("svd_truncation must be positive when provided")
        self.hankel_rows = hankel_rows
        self.svd_truncation = svd_truncation

    def _build_hankel(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if window.ndim != 1:
            raise ValueError("window must be one-dimensional numeric data")
        if window.size < self.hankel_rows + 1:
            raise ValueError("window is too short for the requested Hankel size")

        windows = np.lib.stride_tricks.sliding_window_view(window, self.hankel_rows)
        snapshots = windows.T  # shape: (hankel_rows, n_snapshots)
        X_1 = snapshots[:, :-1]
        X_2 = snapshots[:, 1:]
        return X_1, X_2

    def decompose(self, window: Sequence[float]) -> DMDResult:
        """Compute Exact DMD eigenvalues and the instability index.

        Parameters
        ----------
        window : Sequence[float]
            Rolling history of the most recent observations.

        Returns
        -------
        DMDResult
            Contains the Koopman operator approximation, its eigenvalues,
            and the dominant eigenvalue magnitude ``|lambda|max``.
        """
        arr = np.asarray(window, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("window contains non-finite values that cannot be decomposed")
        X_1, X_2 = self._build_hankel(arr)

        U, s, Vh = np.linalg.svd(X_1, full_matrices=False)
        svd_tol = np.finfo(s.dtype).eps * max(X_1.shape) * s[0] if s.size else 0.0
        nonzero_count = int(np.count_nonzero(s > svd_tol))
        r = min(self.svd_truncation or nonzero_count, nonzero_count)
        if r == 0:
            raise ValueError("SVD produced zero singular values; cannot continue")

        U_r = U[:, :r]
        S_r_inv = np.diag(1.0 / s[:r])
        V_r = Vh[:r, :].T
        koopman_operator = U_r.T @ X_2 @ V_r @ S_r_inv
        eigenvalues = np.linalg.eigvals(koopman_operator)
        instability_index = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else np.nan

        return DMDResult(
            eigenvalues=eigenvalues,
            koopman_operator=koopman_operator,
            instability_index=instability_index,
        )


def _require_pywt() -> None:
    if pywt is None:  # pragma: no cover - simple dependency guard
        raise ImportError(
            "PyWavelets is required for wavelet decomposition. Install via 'pip install pywavelets'."
        )


def _require_torch() -> Tuple[object, object]:
    if torch is None or nn is None:  # pragma: no cover - simple dependency guard
        raise ImportError("torch is required for Neural ODE components. Install via 'pip install torch'.")
    return torch, nn


def _require_odeint() -> object:
    if odeint is None:  # pragma: no cover - simple dependency guard
        raise ImportError(
            "torchdiffeq is required for Neural ODE integration. Install via 'pip install torchdiffeq'."
        )
    return odeint


def multi_resolution_wavelet_decompose(
    series: Sequence[float], *, wavelet: str = "sym5", level: int = 3, threshold: Optional[float] = None
) -> WaveletDecompositionResult:
    """Apply a stationary wavelet transform (SWT) for multi-resolution analysis.

    Parameters
    ----------
    series : Sequence[float]
        One-dimensional numeric time series to decompose.
    wavelet : str, default "sym5"
        Mother wavelet to use.  Symlet-5 is the default per the
        specification for Symlet denoising.
    level : int, default 3
        Number of decomposition levels. Higher values yield finer
        frequency resolution.
    threshold : float or None, default None
        Soft-threshold applied to detail coefficients for denoising. If
        ``None``, a universal threshold based on the finest-scale detail
        is computed automatically.

    Returns
    -------
    WaveletDecompositionResult
        Contains approximation/detail coefficients, denoised series, and
        a coefficient matrix suitable for downstream neural attention or
        embedding layers.
    """

    _require_pywt()
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional")
    coeffs = pywt.swt(arr, wavelet, level=level, start_level=0, trim_approx=True)
    approximation = coeffs[-1][0]
    details = [detail for _, detail in coeffs]

    if threshold is None:
        sigma = np.median(np.abs(details[-1] - np.median(details[-1]))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(arr.size))

    denoised_pairs = []
    for approx_band, detail_band in coeffs:
        denoised_detail = pywt.threshold(detail_band, threshold, mode="soft")
        denoised_pairs.append((approx_band, denoised_detail))
    denoised = pywt.iswt(denoised_pairs, wavelet)

    coefficient_matrix = np.vstack([approximation] + details)
    return WaveletDecompositionResult(
        approximation=approximation,
        details=details,
        denoised=denoised,
        coefficient_matrix=coefficient_matrix,
    )


class ODEFunc(_MODULE_BASE):
    """Neural network parameterization of :math:`\frac{dh}{dt}`.

    The architecture uses ``tanh`` activations to keep gradients bounded
    while modeling the continuous-time evolution of the hidden state.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        _, nn_mod = _require_torch()
        super().__init__()
        self.net = nn_mod.Sequential(
            nn_mod.Linear(input_dim, hidden_dim),
            nn_mod.Tanh(),
            nn_mod.Linear(hidden_dim, hidden_dim),
            nn_mod.Tanh(),
            nn_mod.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, h):  # type: ignore[override]
        return self.net(h)


class NeuralODEBlock(_MODULE_BASE):
    """Solve a Neural ODE for arbitrary time grids using ``odeint``."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, rtol: float = 1e-5, atol: float = 1e-5):
        torch_mod, _ = _require_torch()
        _require_odeint()
        super().__init__()
        self.func = ODEFunc(input_dim=input_dim, hidden_dim=hidden_dim)
        self.rtol = rtol
        self.atol = atol
        # ensures device propagation when registered as submodule
        if hasattr(self, "register_buffer"):
            self.register_buffer("_dummy", torch_mod.tensor(0.0))

    def forward(self, h0: "torch.Tensor", t_grid: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        """Integrate the ODE from initial state ``h0`` along ``t_grid``.

        Parameters
        ----------
        h0 : torch.Tensor
            Initial state of shape ``(batch, features)``.
        t_grid : torch.Tensor
            One-dimensional tensor of time points (can be irregular).

        Returns
        -------
        torch.Tensor
            Trajectory of shape ``(batch, time, features)``.
        """

        if h0.ndim != 2:
            raise ValueError("h0 must have shape (batch, features)")
        if t_grid.ndim != 1:
            raise ValueError("t_grid must be one-dimensional")
        integrator = _require_odeint()
        solution = integrator(self.func, h0, t_grid, rtol=self.rtol, atol=self.atol)
        return solution.transpose(0, 1)


class SobolevLoss(_MODULE_BASE):
    """Sobolev loss combining value and derivative alignment."""

    def __init__(self, derivative_weight: float = 0.5):
        _, nn_mod = _require_torch()
        super().__init__()
        if not 0.0 <= derivative_weight <= 1.0:
            raise ValueError("derivative_weight must be between 0 and 1")
        self.derivative_weight = derivative_weight
        self.mse = nn_mod.MSELoss()

    def forward(
        self, predictions: "torch.Tensor", targets: "torch.Tensor", times: "torch.Tensor"
    ) -> "torch.Tensor":  # type: ignore[override]
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape")
        if predictions.ndim != 3:
            raise ValueError("predictions must have shape (batch, time, features)")
        if times.ndim != 1 or times.numel() != predictions.shape[1]:
            raise ValueError("times must be one-dimensional and align with the time dimension")

        base_loss = self.mse(predictions, targets)
        dt = torch.diff(times)
        if torch.any(dt == 0):
            raise ValueError("times must be strictly increasing for derivative computation")
        while dt.ndim < predictions.ndim - 1:
            dt = dt.unsqueeze(-1)
        pred_derivative = torch.diff(predictions, dim=1) / dt
        target_derivative = torch.diff(targets, dim=1) / dt
        derivative_loss = self.mse(pred_derivative, target_derivative)
        return (1 - self.derivative_weight) * base_loss + self.derivative_weight * derivative_loss


def _hurst_exponent(values: np.ndarray) -> float:
    diffs = np.diff(values)
    if diffs.size == 0:
        return np.nan
    cumulative = np.cumsum(diffs - diffs.mean())
    n = cumulative.size
    if n < 2:
        return np.nan
    ranges = np.maximum.accumulate(cumulative) - np.minimum.accumulate(cumulative)
    stds = np.array([np.std(cumulative[: i + 1]) for i in range(n)], dtype=float)
    stds[stds == 0] = np.nan
    valid = ~np.isnan(stds) & (ranges != 0)
    if not np.any(valid):
        return np.nan
    log_rs = np.log(ranges[valid] / stds[valid])
    log_n = np.log(np.arange(1, n + 1)[valid])
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def rolling_hurst_router(
    series: Sequence[float], *, window: int = 64, min_periods: int = 16, trend_weight: float = 0.65
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a rolling Hurst exponent and derive regime weights.

    Parameters
    ----------
    series : Sequence[float]
        Input time series.
    window : int, default 64
        Lookback window for each Hurst estimate.
    min_periods : int, default 16
        Minimum samples required to start estimating H.  Earlier entries
        are filled with ``NaN``.
    trend_weight : float, default 0.65
        Weight assigned to trend-following features when ``H > 0.5``.
        Mean-reversion weight is ``1 - trend_weight``.

    Returns
    -------
    hurst_values : np.ndarray
        Rolling H estimates (may contain NaN for insufficient history).
    regime_weights : np.ndarray
        Array of shape ``(len(series), 2)`` containing
        ``[trend_weight, mean_reversion_weight]`` at each step.
    """

    arr = np.asarray(series, dtype=float)
    hurst_vals = np.full_like(arr, np.nan, dtype=float)
    weights = np.full((arr.size, 2), np.nan, dtype=float)
    for idx in range(min_periods - 1, arr.size):
        start = max(0, idx - window + 1)
        segment = arr[start : idx + 1]
        h_val = _hurst_exponent(segment)
        hurst_vals[idx] = h_val
        if np.isnan(h_val):
            weights[idx] = [np.nan, np.nan]
            continue
        if h_val > 0.5:
            weights[idx] = [trend_weight, 1 - trend_weight]
        else:
            weights[idx] = [1 - trend_weight, trend_weight]
    return hurst_vals, weights


def plot_scalogram(time_axis: Iterable[float], series: Sequence[float], wavelet: str = "sym5") -> go.Figure:
    """Create a scalogram heatmap using the continuous wavelet transform."""

    _require_pywt()
    arr = np.asarray(series, dtype=float)
    scales = np.arange(1, min(64, arr.size // 2))
    coefficients, frequencies = pywt.cwt(arr, scales, wavelet)
    fig = go.Figure(
        data=go.Surface(x=list(time_axis), y=frequencies, z=np.abs(coefficients), colorscale="Viridis")
    )
    fig.update_layout(title="Scalogram (CWT Energy)", scene=dict(xaxis_title="Time", yaxis_title="Frequency"))
    return fig


def plot_phase_portrait(series: Sequence[float], time_axis: Iterable[float]) -> go.Figure:
    """Plot value vs. velocity to visualize attractors."""

    arr = np.asarray(series, dtype=float)
    t = np.asarray(list(time_axis), dtype=float)
    if arr.shape[0] != t.shape[0]:
        raise ValueError("series and time_axis must have the same length")
    dt = np.diff(t)
    if np.any(dt == 0):
        raise ValueError("time_axis must be strictly increasing")
    velocity = np.diff(arr) / dt
    fig = go.Figure(
        data=go.Scatter3d(
            x=arr[1:], y=velocity, z=t[1:], mode="lines+markers", marker=dict(size=2), line=dict(color="royalblue")
        )
    )
    fig.update_layout(title="Phase Portrait", scene=dict(xaxis_title="Value", yaxis_title="Velocity", zaxis_title="Time"))
    return fig


def plot_forecast_trajectory(
    time_axis: Iterable[float], forecast: Sequence[float], reference: Optional[Sequence[float]] = None
) -> go.Figure:
    """Plot continuous forecast trajectory with optional reference."""

    t = list(time_axis)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=list(forecast), mode="lines", name="Forecast"))
    if reference is not None:
        fig.add_trace(go.Scatter(x=t, y=list(reference), mode="lines", name="Reference", line=dict(dash="dash")))
    fig.update_layout(title="Neural ODE Forecast Trajectory", xaxis_title="Time", yaxis_title="Value")
    return fig


__all__ = [
    "WaveletDecompositionResult",
    "multi_resolution_wavelet_decompose",
    "ODEFunc",
    "NeuralODEBlock",
    "SobolevLoss",
    "rolling_hurst_router",
    "plot_scalogram",
    "plot_phase_portrait",
    "plot_forecast_trajectory",
    "DMDResult",
    "KoopmanSpectralDecomposer",
]
