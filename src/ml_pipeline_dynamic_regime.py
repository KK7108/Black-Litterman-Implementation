import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PERFORMANCE_DIR

print("=" * 65)
print("TWO-REGIME DYNAMIC EXPANDING BACKTEST (METHOD B: INCLUDING IBIT)")
print("=" * 65)

# 1. LOAD PRICING DATA
prices_file = DATA_DIR / "stitched_prices_with_ibit.csv"
print(f"\n[1/5] Loading historical prices with IBIT from {prices_file.name}...")
prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)

daily_returns = prices.pct_change()
vol_20d = daily_returns.rolling(window=20).std() * np.sqrt(252)
vol_60d = daily_returns.rolling(window=60).std() * np.sqrt(252)

monthly_prices = prices.resample('ME').last()
monthly_vol_20d = vol_20d.resample('ME').last()
monthly_vol_60d = vol_60d.resample('ME').last()

# 2. DYNAMIC FEATURE ENGINEERING (Respecting Asset Inception Dates)
print("\n[2/5] Engineering features dynamically per asset...")
all_assets_data = []

for ticker in monthly_prices.columns:
    p = monthly_prices[ticker]
    first_valid = p.first_valid_index()
    if first_valid is None:
        continue
        
    df = pd.DataFrame(index=monthly_prices.loc[first_valid:].index)
    df['Ticker'] = ticker
    
    # Momentum lookbacks
    df['Mom_1M'] = p.pct_change(1)
    df['Mom_3M'] = p.pct_change(3).fillna(df['Mom_1M'])
    df['Mom_6M'] = p.pct_change(6).fillna(df['Mom_3M'])
    df['Mom_12M'] = p.pct_change(12).fillna(df['Mom_6M'])
    
    # Volatility
    df['Vol_20D'] = monthly_vol_20d[ticker]
    df['Vol_60D'] = monthly_vol_60d[ticker].fillna(df['Vol_20D'])
    
    # Target (forward returns)
    df['Target_Future_3M_Ret'] = p.pct_change(3).shift(-3)
    df['Actual_Next_1M_Ret'] = p.pct_change(1).shift(-1)
    
    # Drop rows before asset had at least 1 month of return and volatility
    df = df.dropna(subset=['Mom_1M', 'Vol_20D', 'Actual_Next_1M_Ret'])
    all_assets_data.append(df)

ml_data = pd.concat(all_assets_data).sort_index()
print(f" -> Total asset-month feature rows: {len(ml_data)}")

# 3. EXPANDING WINDOW BACKTEST LOOP
print("\n[3/5] Running Expanding Window Walk-Forward Backtest (2015 - Present)...")
test_start_date = pd.to_datetime("2015-01-31")
ibit_launch_date = pd.to_datetime("2024-01-11")

all_months = ml_data.index.unique().sort_values()
feature_cols = ['Mom_1M', 'Mom_3M', 'Mom_6M', 'Mom_12M', 'Vol_20D', 'Vol_60D']

ml_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)

all_predictions = []
all_portfolio_weights = []

for current_test_month in all_months:
    if current_test_month < test_start_date:
        continue
        
    train_data = ml_data[ml_data.index < current_test_month].dropna(subset=['Target_Future_3M_Ret'])
    test_data = ml_data[ml_data.index == current_test_month]
    
    if train_data.empty or test_data.empty:
        continue
        
    # Fit Machine Learning model on past observations
    ml_model.fit(train_data[feature_cols], train_data['Target_Future_3M_Ret'])
    preds = ml_model.predict(test_data[feature_cols])
    
    tickers_list = test_data['Ticker'].values
    num_assets = len(tickers_list)
    
    # Record predictions
    month_preds = pd.DataFrame({
        'Date': current_test_month,
        'Ticker': tickers_list,
        'ML_Forecast_Return': preds
    })
    all_predictions.append(month_preds)
    
    # Estimate Dynamic Covariance Matrix from daily returns up to current month
    sub_daily = daily_returns.loc[:current_test_month, tickers_list].dropna(how='all')
    cov_matrix = sub_daily.cov().fillna(0).values * 252 # Annualized
    
    # Information Coefficient (Confidence Estimation)
    last_36_dates = train_data.index.unique()[-36:]
    recent_train = train_data[train_data.index.isin(last_36_dates)]
    recent_preds = ml_model.predict(recent_train[feature_cols])
    ic_mat = np.corrcoef(recent_preds, recent_train['Target_Future_3M_Ret'])
    ic = ic_mat[0, 1] if not np.isnan(ic_mat[0, 1]) else 0.0
    
    if ic <= 0.00:
        c = 0.10
    elif ic <= 0.05:
        c = 0.30
    elif ic <= 0.10:
        c = 0.50
    elif ic <= 0.15:
        c = 0.70
    else:
        c = 0.90
        
    # Construct Views Matrix (Top 3 Long vs Bottom 3 Short)
    sorted_preds = month_preds.sort_values(by='ML_Forecast_Return', ascending=False)
    num_views = min(3, num_assets // 2)
    P = np.zeros((num_views, num_assets))
    Q = np.zeros(num_views)
    
    for i in range(num_views):
        long_tk = sorted_preds.iloc[i]['Ticker']
        short_tk = sorted_preds.iloc[-(i+1)]['Ticker']
        
        long_idx = np.where(tickers_list == long_tk)[0][0]
        short_idx = np.where(tickers_list == short_tk)[0][0]
        
        P[i, long_idx] = 1.0
        P[i, short_idx] = -1.0
        Q[i] = sorted_preds.iloc[i]['ML_Forecast_Return'] - sorted_preds.iloc[-(i+1)]['ML_Forecast_Return']
        
    # Calculate Omega Matrix
    P_Sigma_P_T = np.dot(np.dot(P, cov_matrix), P.T)
    Omega = P_Sigma_P_T * ((1.0 - c) / c) + (np.eye(num_views) * 1e-6)
    
    # Black-Litterman Posterior Math
    w_mkt = np.ones(num_assets) / num_assets
    risk_aversion = 2.0
    tau = 0.05
    Pi = risk_aversion * np.dot(cov_matrix, w_mkt)
    
    tau_cov_inv = np.linalg.pinv(tau * cov_matrix)
    Omega_inv = np.linalg.inv(Omega)
    M_inv = tau_cov_inv + np.dot(np.dot(P.T, Omega_inv), P)
    post_cov = np.linalg.pinv(M_inv)
    
    term1 = np.dot(tau_cov_inv, Pi)
    term2 = np.dot(np.dot(P.T, Omega_inv), Q)
    post_returns = np.dot(post_cov, (term1 + term2))
    
    # Portfolio Optimization (Long-only, max 35% concentration per asset)
    def objective(w):
        port_ret = np.dot(w, post_returns)
        port_var = np.dot(w.T, np.dot(post_cov, w))
        return -(port_ret - (risk_aversion / 2) * port_var)
        
    bounds = tuple((0.0, 0.35) for _ in range(num_assets))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    res = minimize(objective, w_mkt, method='SLSQP', bounds=bounds, constraints=cons)
    
    w_df = pd.DataFrame({
        'Date': current_test_month,
        'Ticker': tickers_list,
        'Optimal_Weight': res.x,
        'Equal_Weight': w_mkt,
        'Actual_Return': test_data['Actual_Next_1M_Ret'].values
    })
    all_portfolio_weights.append(w_df)

print(" -> Expanding loop complete!")

# 4. SAVE PREDICTIONS & COMPUTE METRICS
df_all_preds = pd.concat(all_predictions)
df_all_preds.to_csv(PERFORMANCE_DIR / "ml_predictions_two_regime.csv", index=False)
print(" -> Saved outputs/performance/ml_predictions_two_regime.csv")

df_perf = pd.concat(all_portfolio_weights)
df_perf['ML_Cont'] = df_perf['Optimal_Weight'] * df_perf['Actual_Return']
df_perf['EQ_Cont'] = df_perf['Equal_Weight'] * df_perf['Actual_Return']

monthly_perf = df_perf.groupby('Date')[['ML_Cont', 'EQ_Cont']].sum()
cumulative_returns = (1 + monthly_perf).cumprod()

total_months = len(monthly_perf)
ann_ret_ml = cumulative_returns['ML_Cont'].iloc[-1] ** (12 / total_months) - 1
ann_ret_eq = cumulative_returns['EQ_Cont'].iloc[-1] ** (12 / total_months) - 1
ann_vol_ml = monthly_perf['ML_Cont'].std() * np.sqrt(12)
ann_vol_eq = monthly_perf['EQ_Cont'].std() * np.sqrt(12)
sharpe_ml = ann_ret_ml / ann_vol_ml
sharpe_eq = ann_ret_eq / ann_vol_eq

print("\n" + "=" * 65)
print("BACKTEST REPORT CARD (2015 - PRESENT | TWO-REGIME)")
print("=" * 65)
print(f"Total Months Tested: {total_months} (Regime 1: 108 mos | Regime 2: {total_months - 108} mos)")
print("-----------------------------------------------------------------")
print("STRATEGY               | ANN. RETURN | ANN. VOLATILITY | SHARPE ")
print("-----------------------------------------------------------------")
print(f"Machine Learning (BL)  |   {ann_ret_ml*100:>6.2f}%   |     {ann_vol_ml*100:>6.2f}%     |  {sharpe_ml:>5.2f} ")
print(f"Equal-Weight Benchmark |   {ann_ret_eq*100:>6.2f}%   |     {ann_vol_eq*100:>6.2f}%     |  {sharpe_eq:>5.2f} ")
print("=================================================================\n")

# YEAR BY YEAR BREAKDOWN
print("=" * 65)
print("                     YEAR-BY-YEAR BREAKDOWN                      ")
print("=" * 65)
print("YEAR | ML STRATEGY | EQUAL-WEIGHT | OUTPERFORMED? | REGIME")
print("-----------------------------------------------------------------")
yearly_perf = (1 + monthly_perf).resample('YE').prod() - 1
for yr, row in yearly_perf.iterrows():
    m_ret = row['ML_Cont'] * 100
    e_ret = row['EQ_Cont'] * 100
    outperf = "YES" if m_ret > e_ret else "NO"
    regime = "Regime 2 (13 Assets + IBIT)" if yr.year >= 2024 else "Regime 1 (12 Assets)"
    print(f"{yr.year} | {m_ret:>9.2f}%  | {e_ret:>10.2f}%  |     {outperf:3s}     | {regime}")
print("=================================================================\n")

# 5. VISUALIZATIONS
print("[5/5] Generating annotated performance charts...")

# Chart 1: Cumulative Performance
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns.index, cumulative_returns['ML_Cont'], label='Two-Regime ML (Black-Litterman)', linewidth=2.5, color='#1f77b4')
plt.plot(cumulative_returns.index, cumulative_returns['EQ_Cont'], label='Equal-Weight Benchmark', linewidth=2, linestyle='--', color='#7f7f7f')
plt.axvline(pd.to_datetime('2024-01-11'), color='#ff7f0e', linestyle=':', linewidth=2, label='IBIT Inception (Regime 2)')
plt.title('Two-Regime Dynamic Performance: ML-BL vs Benchmark (2015-2026)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Growth of $1 ($)')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(PERFORMANCE_DIR / 'chart_two_regime_performance.png', dpi=300)
print(" -> Saved outputs/performance/chart_two_regime_performance.png")

# Chart 2: Monthly Active Alpha
monthly_perf['Excess'] = monthly_perf['ML_Cont'] - monthly_perf['EQ_Cont']
plt.figure(figsize=(12, 6))
colors = ['#2ca02c' if x >= 0 else '#d62728' for x in monthly_perf['Excess']]
plt.bar(monthly_perf.index, monthly_perf['Excess'] * 100, color=colors, width=20, alpha=0.75)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(pd.to_datetime('2024-01-11'), color='#ff7f0e', linestyle=':', linewidth=2, label='IBIT Inception (Regime 2)')
plt.title('Monthly Active Alpha (%): ML-BL Strategy vs Benchmark', fontsize=14, fontweight='bold')
plt.ylabel('Active Spread (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(PERFORMANCE_DIR / 'chart_two_regime_alpha.png', dpi=300)
print(" -> Saved outputs/performance/chart_two_regime_alpha.png")

# Chart 3: Dynamic Asset Allocation Stacked Bar Chart
yearly_weights = df_perf.groupby([df_perf['Date'].dt.year, 'Ticker'])['Optimal_Weight'].mean().unstack().fillna(0)
yearly_weights.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20', edgecolor='white')
plt.title('Dynamic Asset Allocation Weights by Year (With IBIT from 2024)', fontsize=14, fontweight='bold')
plt.ylabel('Allocation Weight')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(PERFORMANCE_DIR / 'chart_two_regime_asset_allocation.png', dpi=300)
print(" -> Saved outputs/performance/chart_two_regime_asset_allocation.png")

print("\n" + "=" * 65)
print("TWO-REGIME BACKTEST COMPLETED SUCCESSFULLY!")
print("=" * 65)
