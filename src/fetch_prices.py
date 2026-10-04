"""
fetch_prices.py
Fetches daily ETF price history from Yahoo Finance, aligns timestamps,
calculates returns & covariance matrices, and outputs clean CSVs to data/.
"""

import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

# Ensure src/ is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CORE_TICKERS, CRYPTO_ETF, PROXY_TICKER

START_DATE = '2010-01-01'

def fetch_and_update():
    print("=" * 60)
    print("FETCHING ASSET PRICES FROM YAHOO FINANCE")
    print("=" * 60)
    
    # 1. Download Core 12 Assets
    print(f"\n[1/3] Downloading core {len(CORE_TICKERS)} assets from {START_DATE} to latest...")
    raw_core = yf.download(CORE_TICKERS, start=START_DATE, auto_adjust=True)
    
    # Extract Close prices
    if 'Close' in raw_core:
        prices_core = raw_core['Close']
    else:
        prices_core = raw_core
        
    prices_core = prices_core[CORE_TICKERS]
    prices_core = prices_core.dropna()
    prices_core.index = pd.to_datetime(prices_core.index).strftime('%Y-%m-%d')
    
    # Save to data/
    out_stitched = DATA_DIR / "stitched_prices.csv"
    prices_core.to_csv(out_stitched)
    print(f" -> Saved {out_stitched.name}: {len(prices_core)} trading days ({prices_core.index[0]} to {prices_core.index[-1]})")
    
    returns_core = np.log(prices_core / prices_core.shift(1)).dropna()
    cov_core = returns_core.cov()
    out_cov = DATA_DIR / "covariance_matrix.csv"
    cov_core.to_csv(out_cov)
    print(f" -> Saved {out_cov.name} (12x12 daily covariance)")

    # 2. Download IBIT
    print(f"\n[2/3] Downloading {CRYPTO_ETF} (Bitcoin ETF)...")
    raw_ibit = yf.download(CRYPTO_ETF, start='2024-01-01', auto_adjust=True)
    ibit_close = raw_ibit['Close'] if 'Close' in raw_ibit else raw_ibit
    if isinstance(ibit_close, pd.DataFrame):
        ibit_close = ibit_close.iloc[:, 0]
    ibit_close = ibit_close.dropna()
    ibit_close.index = pd.to_datetime(ibit_close.index).strftime('%Y-%m-%d')
    print(f" -> IBIT trading days available: {len(ibit_close)} ({ibit_close.index[0]} to {ibit_close.index[-1]})")

    # Merge Core + IBIT
    prices_with_ibit = prices_core.copy()
    prices_with_ibit[CRYPTO_ETF] = ibit_close
    out_with_ibit = DATA_DIR / "stitched_prices_with_ibit.csv"
    prices_with_ibit.to_csv(out_with_ibit)
    print(f" -> Saved {out_with_ibit.name} ({len(prices_with_ibit)} rows, 13 assets)")

    # Save post-IBIT common period
    post_ibit = prices_with_ibit.dropna()
    out_post_2024 = DATA_DIR / "stitched_prices_post_2024_all13.csv"
    post_ibit.to_csv(out_post_2024)
    returns_post_ibit = np.log(post_ibit / post_ibit.shift(1)).dropna()
    cov_post_ibit = returns_post_ibit.cov()
    out_cov_post = DATA_DIR / "covariance_matrix_post_2024_all13.csv"
    cov_post_ibit.to_csv(out_cov_post)
    print(f" -> Saved {out_post_2024.name}: {len(post_ibit)} days concurrent for all 13 assets")
    print(f" -> Saved {out_cov_post.name} (13x13 daily covariance)")

    # 3. Optional: Synthetic Bitcoin proxy
    print(f"\n[3/3] Generating synthetic continuous track record for Bitcoin using {PROXY_TICKER} proxy...")
    raw_btc = yf.download(PROXY_TICKER, start='2014-09-01', auto_adjust=True)
    btc_close = raw_btc['Close'] if 'Close' in raw_btc else raw_btc
    if isinstance(btc_close, pd.DataFrame):
        btc_close = btc_close.iloc[:, 0]
    btc_close = btc_close.dropna()
    btc_close.index = pd.to_datetime(btc_close.index).strftime('%Y-%m-%d')
    
    inception_date = ibit_close.index[0]
    if inception_date in btc_close.index:
        ratio = ibit_close.loc[inception_date] / btc_close.loc[inception_date]
        synthetic_ibit_pre = btc_close.loc[:inception_date].iloc[:-1] * ratio
        synthetic_ibit = pd.concat([synthetic_ibit_pre, ibit_close])
        
        prices_synthetic = prices_core.copy()
        prices_synthetic[CRYPTO_ETF] = synthetic_ibit
        prices_synthetic_valid = prices_synthetic.dropna()
        out_synthetic_p = DATA_DIR / "stitched_prices_synthetic_ibit_2014_present.csv"
        out_synthetic_c = DATA_DIR / "covariance_matrix_synthetic_ibit_2014_present.csv"
        prices_synthetic_valid.to_csv(out_synthetic_p)
        returns_synthetic = np.log(prices_synthetic_valid / prices_synthetic_valid.shift(1)).dropna()
        returns_synthetic.cov().to_csv(out_synthetic_c)
        print(f" -> Saved {out_synthetic_p.name}: {len(prices_synthetic_valid)} days (2014-present)")

    print("\n" + "=" * 60)
    print("ALL PRICES FETCHED & UPDATED SUCCESSFULLY IN data/")
    print("=" * 60)

if __name__ == "__main__":
    fetch_and_update()
