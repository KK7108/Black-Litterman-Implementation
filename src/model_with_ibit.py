"""
model_with_ibit.py
13-Asset Black-Litterman Model including Bitcoin ETF (IBIT)
Uses the post-Jan 2024 common trading history and live market capitalizations.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from numpy.linalg import inv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR

print("=" * 60)
print("BLACK-LITTERMAN MODEL (13 ASSETS INCLUDING IBIT)")
print("=" * 60)

# 1. LOAD DATA
cov_file = DATA_DIR / "covariance_matrix_post_2024_all13.csv"
prices_file = DATA_DIR / "stitched_prices_post_2024_all13.csv"

if not cov_file.exists() or not prices_file.exists():
    print("Pre-calculated post-2024 data missing. Running fetch_prices.py first...")
    import fetch_prices
    fetch_prices.fetch_and_update()

print(f"\nLoading 13-asset covariance matrix from {cov_file.name}...")
cov_matrix = pd.read_csv(cov_file, index_col=0)
assets = list(cov_matrix.index)
print(f"Assets ({len(assets)}):", assets)

# 2. PULL LIVE MARKET CAPITALS & COMPUTE EQUILIBRIUM WEIGHTS
print("\nPulling live market data from Yahoo Finance...")
market_caps = {}
for ticker in assets:
    try:
        asset = yf.Ticker(ticker)
        size = asset.info.get('totalAssets') or asset.info.get('marketCap')
        if size is None:
            size = 10000000000
        market_caps[ticker] = size
        print(f" -> {ticker:4s}: ${size:,.2f}")
    except Exception as e:
        print(f" -> Failed to pull {ticker}, using fallback: {e}")
        market_caps[ticker] = 10000000000

caps_series = pd.Series(market_caps)[assets]
market_weights = caps_series / caps_series.sum()

# 3. CALCULATE IMPLIED EQUILIBRIUM RETURNS (Pi)
risk_aversion = 2.5
implied_daily = risk_aversion * cov_matrix.dot(market_weights)
annual_implied_returns = implied_daily * 252

print("\n" + "=" * 60)
print("MARKET EQUILIBRIUM (BASELINE) RETURNS & WEIGHTS")
print("=" * 60)
summary_base = pd.DataFrame({
    'Market Weight (%)': (market_weights * 100).round(2),
    'Implied Return (%)': (annual_implied_returns * 100).round(2)
})
print(summary_base)

# 4. FORMULATE VIEWS (P, Q, and Confidences)
# View 1: SPY outperforms IEV by 2.0%
# View 2: GLD absolute return of 5.5%
# View 3: HYG outperforms SHY by 1.5%
# View 4: IBIT absolute return of 15.0%
num_views = 4
P = np.zeros((num_views, len(assets)))

# View 1
P[0, assets.index('SPY')] = 1.0
P[0, assets.index('IEV')] = -1.0

# View 2
P[1, assets.index('GLD')] = 1.0

# View 3
P[2, assets.index('HYG')] = 1.0
P[2, assets.index('SHY')] = -1.0

# View 4 (Bitcoin View)
P[3, assets.index('IBIT')] = 1.0

Q = np.array([0.020, 0.055, 0.015, 0.150])
confidences = np.array([0.60, 0.75, 0.50, 0.50])

print("\n" + "=" * 60)
print("INVESTOR VIEWS MATRIX (P) AND RETURN TARGETS (Q)")
print("=" * 60)
print(pd.DataFrame(P, columns=assets, index=[
    'View 1: SPY > IEV by 2%',
    'View 2: GLD = 5.5%',
    'View 3: HYG > SHY by 1.5%',
    'View 4: IBIT = 15.0%'
]))
print("Target Returns (Q):", Q)
print("Confidences:", confidences)

# 5. OMEGA MATRIX (Idzorek Formulation)
tau = 0.025
annual_cov = cov_matrix * 252

omega_diagonals = []
for i in range(num_views):
    p_i = P[i, :]
    view_var = p_i.dot(tau * annual_cov).dot(p_i.T)
    # Scale uncertainty inversely with confidence
    conf = confidences[i]
    omega_val = view_var * ((1.0 - conf) / conf)
    omega_diagonals.append(omega_val)

Omega = np.diag(omega_diagonals)

# 6. MASTER BLACK-LITTERMAN EQUATION
# E(R) = [ (tau * Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
inv_tau_sigma = inv(tau * annual_cov.values)
inv_omega = inv(Omega)

A = inv_tau_sigma + P.T.dot(inv_omega).dot(P)
b = inv_tau_sigma.dot(annual_implied_returns.values) + P.T.dot(inv_omega).dot(Q)
bl_returns = inv(A).dot(b)
bl_series = pd.Series(bl_returns, index=assets)

print("\n" + "=" * 60)
print("POSTERIOR BLACK-LITTERMAN EXPECTED RETURNS")
print("=" * 60)
comparison_returns = pd.DataFrame({
    'Market Baseline (%)': (annual_implied_returns * 100).round(2),
    'Black-Litterman (%)': (bl_series * 100).round(2),
    'Difference (%)': ((bl_series - annual_implied_returns) * 100).round(2)
})
print(comparison_returns)

# 7. PORTFOLIO OPTIMIZATION (Mean-Variance with no-shorting constraint)
def objective(weights):
    p_ret = np.dot(weights, bl_series.values)
    p_var = np.dot(weights.T, annual_cov.values).dot(weights)
    # Utility = Expected Return - 0.5 * Lambda * Variance
    return -(p_ret - 0.5 * risk_aversion * p_var)

init_weights = np.ones(len(assets)) / len(assets)
bounds = tuple((0.0, 1.0) for _ in range(len(assets)))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

opt = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = pd.Series(opt.x, index=assets)

print("\n" + "=" * 60)
print("RECOMMENDED PORTFOLIO WEIGHTS")
print("=" * 60)
weights_df = pd.DataFrame({
    'Market Weight (%)': (market_weights * 100).round(2),
    'BL Weight (%)': (opt_weights * 100).round(2),
    'Active Tilt (%)': ((opt_weights - market_weights) * 100).round(2)
})
print(weights_df)

# Portfolio Statistics
mkt_ret = np.dot(market_weights.values, annual_implied_returns.values)
mkt_vol = np.sqrt(market_weights.values.T.dot(annual_cov.values).dot(market_weights.values))

bl_ret = np.dot(opt_weights.values, bl_series.values)
bl_vol = np.sqrt(opt_weights.values.T.dot(annual_cov.values).dot(opt_weights.values))

rf = 0.043 # approx 10Y Treasury yield
print("\n" + "=" * 60)
print("PORTFOLIO PERFORMANCE METRICS")
print("=" * 60)
print(f"Market Portfolio Return:    {mkt_ret * 100:.2f}% | Volatility: {mkt_vol * 100:.2f}% | Sharpe: {(mkt_ret - rf) / mkt_vol:.3f}")
print(f"Black-Litterman Portfolio:  {bl_ret * 100:.2f}% | Volatility: {bl_vol * 100:.2f}% | Sharpe: {(bl_ret - rf) / bl_vol:.3f}")
print("=" * 60)
