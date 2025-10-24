"""
Binance-spezifische Datenlogik
- Historische Daten chunked
- Live Polling (1m / andere Intervalle)
- CSV speichern
"""

import os
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    from binance.client import Client as BinanceClient
except ImportError:
    BinanceClient = None

load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

_binance_client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY) if BinanceClient and BINANCE_API_KEY and BINANCE_SECRET_KEY else None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BINANCE_DIR = os.path.join(BASE_DIR, "data", "binance_data")
os.makedirs(BINANCE_DIR, exist_ok=True)

_INTERVAL_MAP = {
    "1s": 1, "5s": 5, "15s": 15, "30s": 30,
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400
}

_live_threads = {}

def _csv_path(symbol: str, interval: str) -> str:
    return os.path.join(BINANCE_DIR, f"{symbol}_{interval}.csv")

def fetch_historical(symbol: str, interval="1m", start_str=None, end_str=None, save=True, limit_per_call=1000):
    if _binance_client is None:
        raise RuntimeError("Binance Client nicht initialisiert")
    start_dt = datetime.fromisoformat(start_str) if start_str else datetime.utcnow() - timedelta(days=30)
    end_dt = datetime.fromisoformat(end_str) if end_str else datetime.utcnow()
    all_rows = []
    cur_start = start_dt

    while cur_start < end_dt:
        chunk_seconds = _INTERVAL_MAP.get(interval, 60) * limit_per_call
        next_end = min(end_dt, cur_start + timedelta(seconds=chunk_seconds))
        klines = _binance_client.get_historical_klines(symbol, interval,
                                                       start_str=cur_start.strftime("%Y-%m-%d %H:%M:%S"),
                                                       end_str=next_end.strftime("%Y-%m-%d %H:%M:%S"),
                                                       limit=limit_per_call)
        if klines:
            df_chunk = pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","trades",
                "taker_buy_base","taker_buy_quote","ignore"
            ])
            all_rows.append(df_chunk)
        cur_start = next_end + timedelta(seconds=1)
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all["open_time"] = pd.to_datetime(df_all["open_time"], unit="ms")
    if save:
        df_all.to_csv(_csv_path(symbol, interval), index=False)
    return df_all

def start_live_poll(symbol: str, interval: str):
    """
    Startet einen Thread, der regelmäßig neue Candles holt
    """
    key = f"{symbol}:{interval}"
    if key in _live_threads:
        return {"status": "already_running", "key": key}

    stop_event = threading.Event()

    def _poll():
        last_open_time = None
        path = _csv_path(symbol, interval)
        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            if "open_time" in df_existing.columns:
                last_open_time = pd.to_datetime(df_existing["open_time"].iloc[-1])
        while not stop_event.is_set():
            klines = _binance_client.get_klines(symbol=symbol, interval=interval, limit=2)
            if klines:
                k = klines[-1]
                open_time = datetime.utcfromtimestamp(int(k[0])/1000).replace(second=0, microsecond=0)
                if last_open_time is None or open_time > last_open_time:
                    candle = {
                        "open_time": open_time.isoformat(),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    }
                    df_new = pd.DataFrame([candle])
                    if not os.path.exists(path):
                        df_new.to_csv(path, index=False)
                    else:
                        df_new.to_csv(path, mode="a", header=False, index=False)
                    last_open_time = open_time
            time.sleep(min(5, max(1, _INTERVAL_MAP.get(interval, 60)//10)))

    thread = threading.Thread(target=_poll, daemon=True)
    _live_threads[key] = {"thread": thread, "stop": stop_event}
    thread.start()
    return {"status": "started", "key": key}
