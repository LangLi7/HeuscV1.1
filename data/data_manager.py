"""
Zentrale Verwaltung
- Cache / CSV
- Scheduler / Live Polling
- Nutzung von binance_data.py und yahoo_data.py
- Candle-Block Erstellung
"""

import os
import pandas as pd
from . import binance_data, yahoo_data
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CANDLES_DIR = os.path.join(BASE_DIR, "data", "candles")
os.makedirs(CANDLES_DIR, exist_ok=True)

_live_threads = {}  # optional für alle Quellen

def get_candles(source: str, symbol: str, interval: str, refresh=False):
    if source == "binance":
        path = os.path.join(BASE_DIR, "data", "binance_data", f"{symbol}_{interval}.csv")
        if refresh or not os.path.exists(path):
            return binance_data.fetch_historical(symbol, interval)
        else:
            return pd.read_csv(path)
    elif source == "yahoo":
        path = os.path.join(BASE_DIR, "data", "yahoo_data", f"{symbol.replace('-','_')}_{interval}.csv")
        if refresh or not os.path.exists(path):
            return yahoo_data.fetch_historical(symbol, interval)
        else:
            return pd.read_csv(path)
    else:
        raise ValueError("Unknown source. Use 'binance' or 'yahoo'.")

def create_and_save_blocks(source: str, symbol: str, interval: str, sequence_length: int = 3, out_name: str = None):
    df = get_candles(source, symbol, interval)
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    required = ['open','high','low','close','volume']
    if not all([c in df.columns for c in required]):
        raise ValueError(f"Missing required columns in CSV -> have {df.columns}")
    df['target'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(int)
    df = df.dropna().reset_index(drop=True)

    features = df[required].values
    blocks, targets = [], []
    for i in range(len(df) - sequence_length + 1):
        block = features[i:i+sequence_length].reshape(-1)
        target = int(df['target'].iloc[i+sequence_length-1])
        blocks.append(block)
        targets.append(target)

    out_df = pd.DataFrame(blocks)
    out_df['target'] = targets
    out_fname = out_name or f"{source}_{symbol}_{interval}_seq{sequence_length}.csv"
    out_path = os.path.join(CANDLES_DIR, out_fname)
    out_df.to_csv(out_path, index=False)
    return out_path
