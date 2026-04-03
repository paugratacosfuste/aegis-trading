"""Download 1yr BTC/USDT 1h candles from Binance API.

Usage:
    python scripts/download_historical.py
"""

import logging
import sys

sys.path.insert(0, "src")

from aegis.backtest.data_loader import download_from_binance

logging.basicConfig(level=logging.INFO)


def main() -> None:
    candles = download_from_binance(
        symbol="BTCUSDT",
        interval="1h",
        start_str="1 Apr, 2025",
        end_str="1 Apr, 2026",
    )
    print(f"Downloaded {len(candles)} candles")
    if candles:
        print(f"First: {candles[0].timestamp} @ ${candles[0].close:,.2f}")
        print(f"Last:  {candles[-1].timestamp} @ ${candles[-1].close:,.2f}")


if __name__ == "__main__":
    main()
