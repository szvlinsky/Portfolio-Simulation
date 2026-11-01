"""Market model: correlated multi-asset prices and portfolio value.

Shared building block: used by the Monte Carlo notebook and, later, by the RL environment.
"""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sample_log_returns(
    mu: Array, cov: Array, size: tuple[int, ...], rng: np.random.Generator
) -> Array:
    """Draw correlated daily log-returns r = drift + L z, with cov = L L^T, z ~ N(0, I).

    mu is chosen so that E[S_{t+1}] = S_t * exp(mu). Returns shape (*size, n_assets).
    """
    drift = mu - 0.5 * np.diag(cov)  # Ito correction
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((*size, len(mu)))
    return drift + z @ chol.T


def simulate_asset_paths(
    s0: Array,
    mu: Array,
    cov: Array,
    n_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    """Simulate asset prices, shape (n_paths, n_days, n_assets)."""
    log_returns = sample_log_returns(mu, cov, (n_paths, n_days), rng)
    return s0 * np.exp(np.cumsum(log_returns, axis=1))


def portfolio_value(paths: Array, shares: Array) -> Array:
    """V = sum_i shares_i * S_i for every path and day, shape (n_paths, n_days)."""
    return paths @ shares


def expected_portfolio_value(s0: Array, mu: Array, shares: Array, n_days: int) -> float:
    """Exact E[V_T] = sum_i x_i * s0_i * exp(mu_i * T) in this model."""
    return float(shares @ (s0 * np.exp(mu * n_days)))

def calibrate(log_returns: Array) -> tuple[Array, Array]:
    """Estimate (mu, cov) of the daily model from log-returns, shape (days, n_assets).

    Sample mean and covariance give the drift d; mu = d + diag(cov)/2 (inverse Ito correction).
    """
    cov = np.cov(log_returns, rowvar=False)
    return log_returns.mean(axis=0) + 0.5 * np.diag(cov), cov