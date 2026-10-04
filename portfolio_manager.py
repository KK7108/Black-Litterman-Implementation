"""
portfolio_manager.py
Operational Portfolio Manager:
- Computes Black-Litterman target allocations
- Takes your total portfolio capital (or existing holdings)
- Fetches live closing prices
- Generates an actionable Trade Execution Blotter with exact shares to BUY/SELL
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from numpy.linalg import inv

def generate_target_weights(risk_aversion=2.5, max_weight=0.35):
    """
    Runs the 13-Asset Black-Litterman model to produce optimal target weights.
    """
    cov_file = "covariance_matrix_post_2024_all13.csv"
    if not os.path.exists(cov_file):
        import fetch_prices
        fetch_prices.fetch_and_update()
        
    cov_matrix = pd.read_csv(cov_file, index_col=0)
    assets = list(cov_matrix.index)
    
    # 1. Pull market caps
    caps = {}
    for t in assets:
        try:
            tk = yf.Ticker(t)
            sz = tk.info.get('totalAssets') or tk.info.get('marketCap') or 10_000_000_000
            caps[t] = sz
        except Exception:
            caps[t] = 10_000_000_000
            
    caps_series = pd.Series(caps)[assets]
    w_mkt = caps_series / caps_series.sum()
    
    # 2. Implied equilibrium returns
    annual_cov = cov_matrix * 252
    Pi = risk_aversion * annual_cov.dot(w_mkt)
    
    # 3. Macro & Quantitative Views
    # View 1: SPY beats IEV by 2.0% (Confidence: 60%)
    # View 2: GLD absolute return 5.5% (Confidence: 75%)
    # View 3: HYG beats SHY by 1.5% (Confidence: 50%)
    # View 4: IBIT expected return 15.0% (Confidence: 50%)
    P = np.zeros((4, len(assets)))
    P[0, assets.index('SPY')] = 1.0; P[0, assets.index('IEV')] = -1.0
    P[1, assets.index('GLD')] = 1.0
    P[2, assets.index('HYG')] = 1.0; P[2, assets.index('SHY')] = -1.0
    P[3, assets.index('IBIT')] = 1.0
    
    Q = np.array([0.020, 0.055, 0.015, 0.150])
    confidences = np.array([0.60, 0.75, 0.50, 0.50])
    
    # 4. Omega matrix
    tau = 0.025
    omega_diags = []
    for i in range(len(Q)):
        p_i = P[i, :]
        var_i = p_i.dot(tau * annual_cov).dot(p_i.T)
        c = confidences[i]
        omega_diags.append(var_i * ((1.0 - c) / c))
    Omega = np.diag(omega_diags)
    
    # 5. Posterior Black-Litterman returns
    inv_tau_cov = inv(tau * annual_cov.values)
    inv_omega = inv(Omega)
    M_inv = inv_tau_cov + P.T.dot(inv_omega).dot(P)
    post_returns = inv(M_inv).dot(inv_tau_cov.dot(Pi.values) + P.T.dot(inv_omega).dot(Q))
    
    # 6. Constrained Optimization
    def obj(w):
        return -(np.dot(w, post_returns) - 0.5 * risk_aversion * np.dot(w.T, annual_cov.values).dot(w))
        
    bounds = tuple((0.0, max_weight) for _ in range(len(assets)))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(obj, np.ones(len(assets))/len(assets), method='SLSQP', bounds=bounds, constraints=cons)
    
    return pd.Series(res.x, index=assets)

def create_trade_blotter(portfolio_capital=100_000, current_holdings=None, min_trade_usd=100):
    """
    Generates exact rebalancing orders given target weights and current portfolio holdings.
    """
    print("=" * 70)
    print(f"PORTFOLIO REBALANCING BLOTTER | CAPITAL: ${portfolio_capital:,.2f}")
    print("=" * 70)
    
    target_weights = generate_target_weights()
    assets = target_weights.index.tolist()
    
    # Fetch live market prices
    print("\nFetching latest prices from Yahoo Finance...")
    prices = {}
    for t in assets:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period='5d')
            p = hist['Close'].iloc[-1]
            prices[t] = p
        except Exception:
            prices[t] = 100.0
            
    prices_series = pd.Series(prices)[assets]
    
    # If no current holdings provided, assume starting from 100% cash
    if current_holdings is None:
        current_holdings = {t: 0 for t in assets}
    else:
        for t in assets:
            if t not in current_holdings:
                current_holdings[t] = 0
                
    curr_shares = pd.Series(current_holdings)[assets]
    curr_values = curr_shares * prices_series
    total_curr_invested = curr_values.sum()
    cash_balance = portfolio_capital - total_curr_invested
    
    target_values = target_weights * portfolio_capital
    target_shares = (target_values / prices_series).apply(np.floor).astype(int)
    
    trade_shares = target_shares - curr_shares
    trade_values = trade_shares * prices_series
    
    blotter = pd.DataFrame({
        'Price ($)': prices_series.round(2),
        'Target %': (target_weights * 100).round(2),
        'Target Val ($)': target_values.round(2),
        'Current Shares': curr_shares,
        'Target Shares': target_shares,
        'Trade (Shares)': trade_shares,
        'Trade Action': trade_shares.apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD')),
        'Trade Val ($)': trade_values.abs().round(2)
    })
    
    # Filter only actionable trades
    actionable = blotter[blotter['Trade (Shares)'] != 0].copy()
    
    print("\nTARGET ASSET ALLOCATION & ORDERS:")
    print(blotter[['Price ($)', 'Target %', 'Current Shares', 'Target Shares', 'Trade Action', 'Trade (Shares)', 'Trade Val ($)']].to_string())
    
    blotter.to_csv("trade_orders.csv")
    print(f"\n -> Full order blotter saved to trade_orders.csv")
    
    total_buy = blotter[blotter['Trade Action'] == 'BUY']['Trade Val ($)'].sum()
    total_sell = blotter[blotter['Trade Action'] == 'SELL']['Trade Val ($)'].sum()
    net_cash_flow = total_buy - total_sell
    
    print("\n" + "-" * 70)
    print(f"SUMMARY: Total Buy: ${total_buy:,.2f} | Total Sell: ${total_sell:,.2f} | Net Required: ${net_cash_flow:,.2f}")
    print(f"Est. Remaining Cash Buffer: ${portfolio_capital - (target_shares * prices_series).sum():,.2f}")
    print("-" * 70)
    
    return blotter

if __name__ == "__main__":
    # Example: Running for a $100,000 portfolio
    create_trade_blotter(portfolio_capital=100_000)
