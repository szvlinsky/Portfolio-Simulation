"""Loading daily prices from CSV files and downloading them from Yahoo Finance."""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

def download_yahoo(
    ticker: str, data_dir: Path, start: str = "2019-01-01", suffix: str = ".WA"
) -> Path:
    """Download daily prices of a Warsaw-listed stock from Yahoo Finance to data_dir/<ticker>.csv.

    Prices are adjusted for splits and dividends (total-return prices).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="yfinance")
        df = yf.download(
            f"{ticker.upper()}{suffix}",
            start=start,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    if df.empty:
        raise ValueError(f"No data returned by Yahoo Finance for {ticker.upper()}{suffix}")
    path = Path(data_dir) / f"{ticker}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["Close"]].to_csv(path, index_label="Date")
    return path


def load_prices(tickers: list[str], data_dir: Path) -> pd.DataFrame:
    """Closing prices, one column per ticker, only dates present for all tickers."""
    closes = {
        t: pd.read_csv(Path(data_dir) / f"{t}.csv", index_col="Date", parse_dates=True)["Close"]
        for t in tickers
    }
    return pd.DataFrame(closes).dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns ln(S_t / S_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()