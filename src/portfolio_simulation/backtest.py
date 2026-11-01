"""Rolling out-of-sample backtest of Monte Carlo Value at Risk."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from portfolio_simulation.data import log_returns
from portfolio_simulation.monte_carlo import value_at_risk
from portfolio_simulation.simulation import calibrate, portfolio_value, simulate_asset_paths


def rolling_var_backtest(
    prices: pd.DataFrame,
    weights: NDArray[np.float64],
    window: int = 500,
    horizon: int = 21,
    alpha: float = 0.05,
    n_paths: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Calibrate on the past `window` days, forecast VaR `horizon` days ahead, compare with reality.

    Windows do not overlap. Portfolio starts with value 1 split by `weights`.
    Returns one row per forecast: VaR (fraction of capital), realised return, breach flag.
    """
    rng = np.random.default_rng(seed)
    returns = log_returns(prices)
    rows = []
    for t in range(window, len(prices) - horizon, horizon):
        mu, cov = calibrate(returns.iloc[t - window : t].to_numpy())
        s0 = prices.iloc[t].to_numpy()
        shares = weights / s0
        paths = simulate_asset_paths(s0, mu, cov, horizon, n_paths, rng)
        var = value_at_risk(portfolio_value(paths, shares)[:, -1], 1.0, alpha)
        realised = float(shares @ prices.iloc[t + horizon].to_numpy() - 1.0)
        rows.append({"date": prices.index[t], "var": var,
                     "realised_return": realised, "breach": realised < -var})
    return pd.DataFrame(rows).set_index("date")