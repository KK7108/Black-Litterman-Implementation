"""
config.py
Centralized directory paths and project configurations.
"""

from pathlib import Path

# Base directories
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PORTFOLIO_DIR = OUTPUTS_DIR / "portfolio"
PERFORMANCE_DIR = OUTPUTS_DIR / "performance"
DOCS_DIR = ROOT_DIR / "docs"

# Ensure all output and data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

# Canonical asset list
CORE_TICKERS = [
    'SPY', 'HYG', 'EFA', 'EEM', 'TLT', 'GLD',
    'SLV', 'DBC', 'VNQ', 'IEV', 'IEF', 'SHY'
]
CRYPTO_ETF = 'IBIT'
PROXY_TICKER = 'BTC-USD'
