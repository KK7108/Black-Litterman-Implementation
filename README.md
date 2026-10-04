# Black-Litterman Multi-Asset Quantitative Allocation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/GitHub-KK7108%2FBlack--Litterman--Implementation-181717?logo=github)](https://github.com/KK7108/Black-Litterman-Implementation)

A production-grade implementation of the **Black-Litterman Asset Allocation Model** integrated with an **ElasticNet Machine Learning Pipeline** across a 13-asset ETF universe (Equities, Treasuries, Corporate Credit, Commodities, Real Estate, and Bitcoin ETF `IBIT`).

---

## 📁 Repository Directory Structure

```text
black-litterman/
├── README.md                              # Operational guide & system documentation
├── requirements.txt                       # Python library dependencies
│
├── src/                                   # Core Python algorithms and runners
│   ├── config.py                          # Centralized directory paths and configurations
│   ├── fetch_prices.py                    # Downloads prices from Yahoo Finance & updates covariance
│   ├── portfolio_manager.py               # Generates live rebalance trade blotter (BUY/SELL orders)
│   ├── model.py                           # Baseline 12-asset Black-Litterman engine
│   ├── model_with_ibit.py                 # 13-asset model including Bitcoin ETF (IBIT)
│   ├── ml_pipeline.py                     # 12-asset walk-forward ML backtest pipeline
│   └── ml_pipeline_dynamic_regime.py      # Two-Regime dynamic expanding backtest (Method B)
│
├── data/                                  # Cleaned historical prices and covariance matrices
│   ├── stitched_prices.csv                # Core 12-asset price series (2010 - present)
│   ├── covariance_matrix.csv              # Core 12-asset daily covariance matrix
│   ├── stitched_prices_with_ibit.csv      # 13-asset dataset (IBIT from Jan 2024 inception)
│   ├── stitched_prices_post_2024_all13.csv# 13-asset concurrent daily history (2024 - present)
│   ├── covariance_matrix_post_2024_all13.csv
│   ├── stitched_prices_synthetic_ibit_2014_present.csv # Spliced BTC-USD proxy (2014 - present)
│   └── covariance_matrix_synthetic_ibit_2014_present.csv
│
├── outputs/                               # Model outputs & trade reports
│   ├── portfolio/                         # Operational allocation outputs
│   │   ├── trade_orders.csv               # Live executable trade blotter with exact share counts
│   │   └── implied_equilibrium_returns.csv# Market capitalization baseline returns
│   │
│   └── performance/                       # Quantitative backtests & publication charts
│       ├── chart_two_regime_performance.png   # Cumulative wealth growth (2015-2026)
│       ├── chart_two_regime_alpha.png         # Monthly active alpha spread
│       ├── chart_two_regime_asset_allocation.png # Dynamic weights by year
│       ├── chart_cumulative_performance.png
│       ├── chart_monthly_alpha.png
│       ├── chart_asset_allocation.png
│       ├── ml_predictions.csv
│       └── ml_predictions_two_regime.csv
│
└── docs/                                  # Research notes & methodology guides
    └── BL_Implementation_Guide_for_Intern.md # Detailed mathematical derivation & Idzorek guide
```

---

## 🎯 The Multi-Asset ETF Universe

| Ticker | Asset Class | Underlying Index / Benchmark | Inception Period |
| :--- | :--- | :--- | :--- |
| **SPY** | US Large Cap Equities | S&P 500 | 1993 – Present |
| **HYG** | Corporate Bonds | High Yield Corporate Bond Index | 2007 – Present |
| **EFA** | International Developed | MSCI EAFE Index | 2001 – Present |
| **EEM** | Emerging Markets | MSCI Emerging Markets Index | 2003 – Present |
| **TLT** | US Long-Term Treasuries | 20+ Year Treasury Bond Index | 2002 – Present |
| **GLD** | Precious Metals (Gold) | Physical Gold Bullion | 2004 – Present |
| **SLV** | Precious Metals (Silver) | Physical Silver Bullion | 2006 – Present |
| **DBC** | Broad Commodities | DBIQ Optimum Yield Diversified Commodity | 2006 – Present |
| **VNQ** | Real Estate | MSCI US REIT Index | 2004 – Present |
| **IEV** | European Equities | S&P Europe 350 Index | 2000 – Present |
| **IEF** | US Intermediate Treasuries| 7-10 Year Treasury Bond Index | 2002 – Present |
| **SHY** | US Short-Term Treasuries | 1-3 Year Treasury Bond Index | 2002 – Present |
| **IBIT** | Digital Assets / Crypto | iShares Bitcoin Trust ETF | **Jan 11, 2024 – Present** |

---

## ⚡ Quickstart & Installation

```powershell
# 1. Clone repository
git clone https://github.com/KK7108/Black-Litterman-Implementation.git
cd Black-Litterman-Implementation

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Operate the System

### 1. Daily / Weekly Market Data Refresh
Downloads the latest daily close prices and dividend/split adjustments from Yahoo Finance, re-estimates covariance matrices, and stores them in `data/`:
```powershell
python src/fetch_prices.py
```

### 2. Live Portfolio Rebalancing & Order Blotter
Generates an actionable trade blotter calculating target dollar weights, live ETF prices, and exact share orders (BUY / SELL / HOLD) saved to `outputs/portfolio/trade_orders.csv`:
```powershell
python src/portfolio_manager.py
```
*To customize portfolio capital or provide current holdings, modify the function call in `src/portfolio_manager.py` (e.g., `create_trade_blotter(portfolio_capital=250_000)`).*

### 3. Run the Two-Regime Dynamic ML Backtest (Method B)
Executes walk-forward machine learning (ElasticNet) from 2015 to the present, dynamically expanding the investment universe from 12 to 13 assets when `IBIT` launches in January 2024:
```powershell
python src/ml_pipeline_dynamic_regime.py
```
*Outputs:*
* Prediction logs: `outputs/performance/ml_predictions_two_regime.csv`
* Visual plots: `outputs/performance/chart_two_regime_*.png`

### 4. Interactive 13-Asset Black-Litterman Analysis
Runs the multi-asset Black-Litterman model using live market capitalizations to evaluate implied equilibrium returns vs. subjective/quantitative views:
```powershell
python src/model_with_ibit.py
```

---

## 📊 Backtest Performance (2015 – Present | Two-Regime)

```text
=================================================================
STRATEGY               | ANN. RETURN | ANN. VOLATILITY | SHARPE 
-----------------------------------------------------------------
Machine Learning (BL)  |     9.56%   |      18.47%     |   0.52 
Equal-Weight Benchmark |     6.84%   |       9.77%     |   0.70 
=================================================================
```

### The IBIT Regime Transition:
* **Regime 1 (2015 – Jan 2024 | 108 Months):** 12 core assets. The portfolio balanced equities, fixed income, real estate, and commodities.
* **Regime 2 (Feb 2024 – Present | 33 Months):** `IBIT` dynamically entered the universe.
  * **2024 Strategy Return:** **+42.82%** (vs. +16.98% benchmark, **+25.84% active alpha**).
  * **2025 Strategy Return:** **+50.90%** (vs. +25.55% benchmark, **+25.35% active alpha**).

---

## 📈 Visual Performance Reports

All performance plots are exported in 300 DPI to `outputs/performance/`:

* **Cumulative Wealth:** `outputs/performance/chart_two_regime_performance.png`
* **Monthly Active Alpha:** `outputs/performance/chart_two_regime_alpha.png`
* **Asset Allocation Weights by Year:** `outputs/performance/chart_two_regime_asset_allocation.png`

---

## 📜 Methodology Reference
For complete mathematical formulas on implied returns ($\Pi = \lambda \Sigma w_{mkt}$), Idzorek's view uncertainty formulation ($\Omega$), and posterior returns, see [`docs/BL_Implementation_Guide_for_Intern.md`](docs/BL_Implementation_Guide_for_Intern.md).
