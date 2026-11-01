import numpy as np
import pandas as pd

from portfolio_simulation.backtest import rolling_var_backtest
from portfolio_simulation.simulation import simulate_asset_paths


def test_backtest_output():
    rng = np.random.default_rng(0)
    cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    path = simulate_asset_paths(np.array([100.0, 50.0]), np.zeros(2), cov, 300, 1, rng)[0]
    prices = pd.DataFrame(
        path, index=pd.bdate_range("2020-01-01", periods=300), columns=["a", "b"]
    )
    bt = rolling_var_backtest(prices, np.array([0.5, 0.5]), window=100, horizon=20, n_paths=2_000)
    assert len(bt) == 9
    assert (bt["var"] > 0).all()
    assert bt["breach"].dtype == bool