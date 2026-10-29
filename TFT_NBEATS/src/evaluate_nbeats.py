"""
Evaluation script for final N-BEATS model checkpoint.
CryptoPredict: BTC Next-Day Close Price Forecasting
"""

import sys
from pathlib import Path

# Add src to path if needed
src_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(src_dir))

from evaluate_models import run_evaluation

if __name__ == "__main__":
    run_evaluation()
