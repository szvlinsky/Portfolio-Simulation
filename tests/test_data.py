import numpy as np
import pandas as pd

from portfolio_simulation.data import load_prices, log_returns


def test_load_prices_and_log_returns(tmp_path):
    for ticker, closes in {"aaa": [100, 110, 121], "bbb": [10, 10, 20]}.items():
        pd.DataFrame(
            {"Date": pd.date_range("2024-01-01", periods=3), "Close": closes}
        ).to_csv(tmp_path / f"{ticker}.csv", index=False)

    returns = log_returns(load_prices(["aaa", "bbb"], tmp_path))
    assert returns.shape == (2, 2)
    assert np.allclose(returns["aaa"], np.log(1.1))