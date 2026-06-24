"""
data_loader.py

Pulls daily adjusted close prices for the 11-asset ETF universe via yfinance,
computes log returns, and derives annualized mean-return vector (mu) and
covariance matrix (Sigma) for use by both the classical Markowitz baseline
and the QUBO/QAOA combinatorial selection pipeline.

Universe (11 sector/asset-class ETFs):
    XLF - Financials
    XLE - Energy
    XLK - Technology
    XLV - Healthcare
    XLI - Industrials
    XLY - Consumer Discretionary
    XLP - Consumer Staples
    XLU - Utilities
    GLD - Gold
    TLT - Long-Term Treasuries (20+yr)
    VNQ - REITs
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

TICKERS = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "GLD", "TLT", "VNQ"]

SECTOR_NAMES = {
    "XLF": "Financials",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "GLD": "Gold",
    "TLT": "Long-Term Treasuries",
    "VNQ": "REITs",
}

TRADING_DAYS_PER_YEAR = 252


def fetch_prices(tickers=TICKERS, years=3.0, end_date=None, cache_path=None):
    """
    Download daily adjusted close prices for `tickers` over the trailing
    `years` window ending at `end_date` (default: today).

    Returns a DataFrame indexed by date, one column per ticker.
    """
    if end_date is None:
        end_date = datetime.today()
    start_date = end_date - timedelta(days=int(years * 365.25) + 10)

    raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,  # adjusted close baked into 'Close'
        progress=False,
        group_by="ticker",
    )

    # yfinance multi-ticker download returns a MultiIndex column DataFrame:
    # (ticker, field). Extract 'Close' (adjusted) per ticker.
    prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    prices = prices.dropna(how="any")  # keep only fully-overlapping trading days

    if cache_path:
        prices.to_csv(cache_path)

    return prices


def compute_returns_stats(prices, trading_days=TRADING_DAYS_PER_YEAR):
    """
    Given a price DataFrame (date x ticker), compute:
      - log_returns: daily log returns DataFrame
      - mu: annualized mean log return vector (numpy array, order = prices.columns)
      - sigma: annualized covariance matrix (numpy array)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")

    mu_daily = log_returns.mean().values
    sigma_daily = log_returns.cov().values

    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * trading_days

    return log_returns, mu_annual, sigma_annual


def rolling_covariance_stability(log_returns, window_days=252, step_days=21):
    """
    Quick diagnostic: compute the covariance matrix on rolling windows and
    report the Frobenius-norm relative change between consecutive windows.
    Large swings would suggest the chosen history length is too short/noisy
    to trust a single static Sigma estimate.
    """
    n = len(log_returns)
    mats = []
    starts = list(range(0, n - window_days, step_days))
    for s in starts:
        window = log_returns.iloc[s : s + window_days]
        mats.append(window.cov().values)

    rel_changes = []
    for i in range(1, len(mats)):
        diff_norm = np.linalg.norm(mats[i] - mats[i - 1], ord="fro")
        base_norm = np.linalg.norm(mats[i - 1], ord="fro")
        rel_changes.append(diff_norm / base_norm)

    return {
        "n_windows": len(mats),
        "mean_relative_change": float(np.mean(rel_changes)) if rel_changes else None,
        "max_relative_change": float(np.max(rel_changes)) if rel_changes else None,
    }


if __name__ == "__main__":
    print(f"Fetching {len(TICKERS)} ETFs, 3-year daily history...")
    prices = fetch_prices(years=3.0, cache_path="data/prices.csv")
    print(f"Price matrix shape: {prices.shape} (days x assets)")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    log_returns, mu, sigma = compute_returns_stats(prices)
    log_returns.to_csv("data/log_returns.csv")
    np.save("data/mu.npy", mu)
    np.save("data/sigma.npy", sigma)

    print("\nAnnualized mean returns (mu):")
    for t, m in zip(TICKERS, mu):
        print(f"  {t:5s} ({SECTOR_NAMES[t]:22s}): {m:+.4f}")

    print("\nAnnualized volatility (sqrt of diagonal of Sigma):")
    vol = np.sqrt(np.diag(sigma))
    for t, v in zip(TICKERS, vol):
        print(f"  {t:5s}: {v:.4f}")

    stability = rolling_covariance_stability(log_returns)
    print(f"\nRolling covariance stability check: {stability}")

    print("\nSaved: data/prices.csv, data/log_returns.csv, data/mu.npy, data/sigma.npy")
