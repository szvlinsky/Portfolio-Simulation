import numpy as np

from portfolio_simulation.monte_carlo import mc_mean
from portfolio_simulation.simulation import (
    expected_portfolio_value,
    portfolio_value,
    sample_log_returns,
    simulate_asset_paths,
)

COV = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])


def test_paths_shape():
    rng = np.random.default_rng(0)
    paths = simulate_asset_paths(np.array([100.0, 50.0]), np.zeros(2), COV, 10, 5, rng)
    assert paths.shape == (5, 10, 2)


def test_single_step_shape():
    rng = np.random.default_rng(0)
    assert sample_log_returns(np.zeros(2), COV, (), rng).shape == (2,)


def test_sample_covariance_matches_target():
    rng = np.random.default_rng(0)
    r = sample_log_returns(np.zeros(2), COV, (200_000,), rng)
    assert np.allclose(np.cov(r.T), COV, rtol=0.05)


def test_mc_mean_matches_exact_expectation():
    rng = np.random.default_rng(1)
    s0, mu, shares = np.array([100.0, 50.0]), np.array([4e-4, 3e-4]), np.array([1.0, 2.0])
    paths = simulate_asset_paths(s0, mu, COV, 50, 50_000, rng)
    mean, se = mc_mean(portfolio_value(paths, shares)[:, -1])
    assert abs(mean - expected_portfolio_value(s0, mu, shares, 50)) < 4 * se

def test_calibrate_recovers_parameters():
    rng = np.random.default_rng(2)
    mu = np.array([4e-4, 3e-4])
    mu_hat, cov_hat = calibrate(sample_log_returns(mu, COV, (200_000,), rng))
    assert np.allclose(mu_hat, mu, atol=3e-4)
    assert np.allclose(cov_hat, COV, rtol=0.05)