"""
Yahoo-spezifische Datenlogik
- Historische Daten (Chunked)
- CSV speichern
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
YAHOO_DIR = os.path.join(BASE_DIR, "data", "yahoo_data")
os.makedirs(YAHOO_DIR, exist_ok=True)

def _csv_path(symbol: str, interval: str) -> str:
    return os.path.join(YAHOO_DIR, f"{symbol.replace('-','_')}_{interval}.csv")

def fetch_historical(symbol: str, interval="1d", start_str=None, end_str=None, save=True):
    start_str = start_str or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_str = end_str or datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(tickers=symbol, start=start_str, end=end_str, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    if save:
        df.to_csv(_csv_path(symbol, interval), index=False)
    return df
