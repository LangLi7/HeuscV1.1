# data/data_manager.py
"""
Data Manager für Heusc (Phase 2)
- Zentrale Verwaltung der historischen Daten (Binance & Yahoo)
- Live-Polling starten/stoppen
- CSV-Speicherung
- Candle-Preprocessing & Block-Erzeugung für Training
- Config-gesteuert
"""

import os
import threading
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# eigene Module
from .binance_data import fetch_historical_binance, start_live_poll_binance, stop_live_poll_binance
from .yahoo_data import fetch_historical_yahoo
from .candle_builder import create_sequences_from_candles
from . import read_settings

# -----------------------
# Pfade
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CANDLES_DIR = os.path.join(DATA_DIR, "candles")
os.makedirs(CANDLES_DIR, exist_ok=True)

# -----------------------
# Load config
# -----------------------
DATA_CONFIG = read_settings(os.path.join(BASE_DIR, "data", "settings", "data_config.json"))
DEFAULT_SEQ_LEN = DATA_CONFIG.get("sequence_length", 3)

# -----------------------
# Global live thread registry
# -----------------------
_live_threads = {}

# -----------------------
# Historische Daten laden oder aus Cache
# -----------------------
def get_candles(source: str, symbol: str, interval: str = None, start_str: str = None, end_str: str = None, refresh: bool = False):
    """
    Liefert DataFrame mit Candles:
    - source: 'binance' oder 'yahoo'
    - refresh=True: neu fetchen
    """
    interval = interval or DATA_CONFIG.get(f"{source}_default_interval", "1m")

    if source == "binance":
        # CSV-Pfad
        out_path = os.path.join(DATA_DIR, "binance_data", f"{symbol}_{interval}.csv")
        if refresh or (not os.path.exists(out_path)):
            return fetch_historical_binance(symbol, interval, start_str, end_str)
        else:
            return pd.read_csv(out_path)

    elif source == "yahoo":
        out_path = os.path.join(DATA_DIR, "yahoo_data", f"{symbol}_{interval}.csv")
        if refresh or (not os.path.exists(out_path)):
            return fetch_historical_yahoo(symbol, interval, start_str, end_str)
        else:
            return pd.read_csv(out_path)

    else:
        raise ValueError("Unknown source. Use 'binance' or 'yahoo'.")

# -----------------------
# Live-Polling
# -----------------------
def start_live_poll(source: str, symbol: str, interval: str = None):
    interval = interval or DATA_CONFIG.get(f"{source}_default_interval", "1m")
    key = f"{source}:{symbol}:{interval}"
    if key in _live_threads:
        return {"status": "already_running", "key": key}

    if source == "binance":
        thread, stop_event = start_live_poll_binance(symbol, interval)
    else:
        raise NotImplementedError("Live polling currently only for 'binance'")

    _live_threads[key] = {"thread": thread, "stop": stop_event}
    return {"status": "started", "key": key}

def stop_live_poll(source: str, symbol: str, interval: str):
    key = f"{source}:{symbol}:{interval}"
    if key not in _live_threads:
        return {"status": "not_running", "key": key}
    info = _live_threads.pop(key)
    info["stop"].set()
    return {"status": "stopped", "key": key}

def list_live_polls():
    return list(_live_threads.keys())

# -----------------------
# Blocks / Sequenzen erzeugen für Training
# -----------------------
def create_and_save_blocks(source: str, symbol: str, interval: str = None, sequence_length: int = None, out_name: str = None):
    """
    Erzeugt Sequenzen (X, y) aus gespeicherten Candles für Training
    """
    sequence_length = sequence_length or DEFAULT_SEQ_LEN
    df = get_candles(source, symbol, interval, refresh=False)

    X, y = create_sequences_from_candles(df, seq_len=sequence_length)

    out_fname = out_name or f"{source}_{symbol}_{interval}_seq{sequence_length}.csv"
    out_path = os.path.join(CANDLES_DIR, out_fname)
    df_blocks = pd.DataFrame(X.reshape(X.shape[0], -1))
    df_blocks['target'] = y
    df_blocks.to_csv(out_path, index=False)
    print(f"[DataManager] Saved blocks to {out_path}")
    return out_path
