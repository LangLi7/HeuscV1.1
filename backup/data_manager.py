# data/data_manager.py
"""
Data Manager für Heusc (Phase 1)
- historische Daten (Binance & Yahoo) downloaden (chunked / Zeitraum)
- Live-Polling (1m / andere Intervalle) starten/stoppen
- CSV-Speicherung in data/binance_data/ und data/yahoo_data/
- Hilfsfunktionen für File-Pfade
"""

import os
import time
import threading
import math
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import json

# externe libs
try:
    from binance.client import Client as BinanceClient
except Exception:
    BinanceClient = None

try:
    import yfinance as yf
except Exception:
    yf = None

# load .env
load_dotenv()

# Pfade
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Heusc/data/ -> Heusc/
DATA_DIR = os.path.join(BASE_DIR, "data")
BINANCE_DIR = os.path.join(DATA_DIR, "binance_data")
YAHOO_DIR = os.path.join(DATA_DIR, "yahoo_data")
CANDLES_DIR = os.path.join(DATA_DIR, "candles")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for p in [BINANCE_DIR, YAHOO_DIR, CANDLES_DIR, LOG_DIR]:
    os.makedirs(p, exist_ok=True)

# Globales Live-Thread-Registry
_live_threads = {}

# Load keys (use .env)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
if BinanceClient and BINANCE_API_KEY and BINANCE_SECRET_KEY:
    _binance_client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
else:
    _binance_client = None

# Helper: interval conversion (in seconds)
_INTERVAL_MAP = {
    "1s": 1,
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 3600,
    "4h": 4 * 3600,
    "1d": 24 * 3600,
    "1w": 7 * 24 * 3600,
    "1M": 30 * 24 * 3600  # grobe Monatsdefinition
}

# -----------------------
# Pfad-Utilities
# -----------------------
def _binance_csv_path(symbol: str, interval: str) -> str:
    fname = f"{symbol}_{interval}.csv"
    return os.path.join(BINANCE_DIR, fname)

def _yahoo_csv_path(symbol: str, interval: str) -> str:
    fname = f"{symbol.replace('-', '_')}_{interval}.csv"
    return os.path.join(YAHOO_DIR, fname)

# -----------------------
# Historische Daten (chunked) - Binance
# -----------------------
def fetch_historical_binance(symbol: str, interval: str = "1m",
                             start_str: str = None, end_str: str = None,
                             save: bool = True, limit_per_call: int = 1000):
    """
    Holt historische Klines von Binance über einen Zeitraum (chunked).
    start_str, end_str in "YYYY-MM-DD" oder "YYYY-MM-DD HH:MM:SS" Format (UTC).
    limit_per_call: wieviel Klines pro Request (1000 ist üblich).
    Liefert DataFrame zurück und speichert als CSV wenn save=True.
    """
    if _binance_client is None:
        raise RuntimeError("Binance Client nicht initialisiert. Setze BINANCE_API_KEY/SECRET in .env und installiere python-binance.")

    # Wenn kein Start angegeben -> 1 Monat zurück als Default
    if start_str is None and end_str is None:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = datetime.fromisoformat(start_str) if start_str else datetime.utcnow() - timedelta(days=30)
        end_dt = datetime.fromisoformat(end_str) if end_str else datetime.utcnow()

    # Convert to ms strings for Binance helper or use client.get_historical_klines
    out_path = _binance_csv_path(symbol, interval)
    all_rows = []
    cur_start = start_dt

    print(f"[DataManager] Starte Binance-Historic: {symbol} {interval} von {start_dt} bis {end_dt}")

    while cur_start < end_dt:
        # next_end is start + chunk_seconds * limit_per_call
        # approximate duration: limit_per_call * interval_seconds
        chunk_seconds = _INTERVAL_MAP.get(interval, 60) * limit_per_call
        next_end = min(end_dt, cur_start + timedelta(seconds=chunk_seconds))

        # Binance expects start_str and end_str in string form
        start_str_bin = cur_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str_bin = next_end.strftime("%Y-%m-%d %H:%M:%S")

        klines = _binance_client.get_historical_klines(symbol, interval, start_str=start_str_bin, end_str=end_str_bin, limit=limit_per_call)
        if not klines:
            # kein Data mehr — breche
            cur_start = next_end
            time.sleep(0.2)
            continue

        df_chunk = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        all_rows.append(df_chunk)
        print(f"[DataManager] Got chunk {cur_start} -> {next_end} rows={len(df_chunk)}")
        cur_start = next_end + timedelta(seconds=1)
        time.sleep(0.2)  # throttle

    if len(all_rows) == 0:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    # optional: konvertiere Zeiten (ms -> datetime)
    try:
        df_all["open_time"] = pd.to_datetime(df_all["open_time"], unit="ms")
    except Exception:
        pass

    if save:
        df_all.to_csv(out_path, index=False)
        print(f"[DataManager] Gespeichert: {out_path}")

    return df_all

# -----------------------
# Historische Daten - Yahoo
# -----------------------
def fetch_historical_yahoo(symbol: str, interval: str = "1d", start_str: str = None, end_str: str = None, save: bool = True):
    """
    Holt historische Daten von Yahoo via yfinance.
    interval: e.g. '1m','5m','1h','1d'
    start_str,end_str: ISO dates 'YYYY-MM-DD'
    """
    if yf is None:
        raise RuntimeError("yfinance nicht installiert. 'pip install yfinance'")

    if start_str is None:
        start_str = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_str is None:
        end_str = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"[DataManager] Starte Yahoo-Historic: {symbol} {interval} von {start_str} bis {end_str}")
    df = yf.download(tickers=symbol, start=start_str, end=end_str, interval=interval, progress=False)
    if df.empty:
        print("[DataManager] Keine Yahoo-Daten erhalten.")
        return df

    df = df.reset_index()
    out_path = _yahoo_csv_path(symbol, interval)
    if save:
        df.to_csv(out_path, index=False)
        print(f"[DataManager] Gespeichert: {out_path}")
    return df

# -----------------------
# Lade aus Cache / falls nicht vorhanden -> hole neu
# -----------------------
def get_candles(source: str, symbol: str, interval: str, start_str: str = None, end_str: str = None, refresh: bool = False):
    """
    Liefert DataFrame mit Candles:
      - source: 'binance' oder 'yahoo'
      - interval: z.B. '1m','1h','1d'
      - wenn refresh=True, wird neu gefetcht
    """
    if source == "binance":
        path = _binance_csv_path(symbol, interval)
        if refresh or (not os.path.exists(path)):
            return fetch_historical_binance(symbol, interval, start_str, end_str)
        else:
            return pd.read_csv(path)
    elif source == "yahoo":
        path = _yahoo_csv_path(symbol, interval)
        if refresh or (not os.path.exists(path)):
            return fetch_historical_yahoo(symbol, interval, start_str, end_str)
        else:
            return pd.read_csv(path)
    else:
        raise ValueError("Unknown source. Use 'binance' or 'yahoo'.")

# -----------------------
# Live polling (REST) - Appends neue Candle wenn fertig
# -----------------------
def _poll_live_binance(symbol: str, interval: str, stop_event: threading.Event):
    """
    Pollt Binance REST API alle interval Sekunden und hängt neue Candle an CSV an.
    Useful for 1m candles: get last kline and check if timestamp changed.
    """
    path = _binance_csv_path(symbol, interval)
    last_open_time = None

    if os.path.exists(path):
        try:
            df_existing = pd.read_csv(path)
            if "open_time" in df_existing.columns:
                last_open_time = pd.to_datetime(df_existing["open_time"].iloc[-1])
        except Exception:
            last_open_time = None

    print(f"[LivePoll] Start polling {symbol} {interval} -> file: {path}")

    while not stop_event.is_set():
        try:
            klines = _binance_client.get_klines(symbol=symbol, interval=interval, limit=2)
            if not klines:
                time.sleep(1)
                continue
            k = klines[-1]
            open_time_ms = int(k[0])
            open_time = datetime.utcfromtimestamp(open_time_ms / 1000).replace(second=0, microsecond=0)

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
                print(f"[LivePoll] New candle saved {candle}")
            # sleep for fraction of interval to re-check quickly near boundary
            time.sleep(min(5, max(1, _INTERVAL_MAP.get(interval, 60) // 10)))
        except Exception as e:
            print(f"[LivePoll][Error] {e}")
            time.sleep(5)

def start_live_poll(source: str, symbol: str, interval: str):
    """
    Startet einen Live-Poller-Thread. Returns thread_id.
    """
    key = f"{source}:{symbol}:{interval}"
    if key in _live_threads:
        return {"status": "already_running", "key": key}

    stop_event = threading.Event()
    if source == "binance":
        if _binance_client is None:
            raise RuntimeError("Binance Client nicht initialisiert.")
        thread = threading.Thread(target=_poll_live_binance, args=(symbol, interval, stop_event), daemon=True)
    else:
        raise NotImplementedError("Live polling currently supports only 'binance'")

    _live_threads[key] = {"thread": thread, "stop": stop_event}
    thread.start()
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
# Aggregation: z.B. 1s -> 1m candles
# -----------------------
def aggregate_ticks_to_candles(ticks_df: pd.DataFrame, timeframe: str = "1m"):
    """
    ticks_df: columns ['timestamp','price','volume'] timestamp: datetime or iso str
    timeframe: '1m', '5m' ...
    returns DataFrame with open_time, open, high, low, close, volume
    """
    if ticks_df.empty:
        return pd.DataFrame()

    df = ticks_df.copy()
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    rule = timeframe
    # pandas uses 'T' for minute alias: '1T' = 1 min
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        rule = f"{minutes}T"
    elif timeframe.endswith('s'):
        seconds = int(timeframe[:-1])
        rule = f"{seconds}S"
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        rule = f"{hours}H"
    elif timeframe == '1d':
        rule = '1D'

    agg = df['price'].resample(rule).ohlc()
    vol = df['volume'].resample(rule).sum()
    res = agg.join(vol).dropna()
    res = res.reset_index().rename(columns={
        'index': 'open_time',
        'volume': 'volume'
    })
    res['open_time'] = res['timestamp'] if 'timestamp' in res.columns else res['open_time']
    # keep columns open_time, open, high, low, close, volume
    res = res[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    return res

# -----------------------
# Save processed candle-block dataset (X blocks) as CSV
# -----------------------
def create_and_save_blocks(source: str, symbol: str, interval: str, sequence_length: int = 3, out_name: str = None):
    """
    Erzeugt Sequenzen (blocks) für Training aus gespeicherten candle CSV,
    speichert als CSV in data/candles/
    """
    if source == "binance":
        path = _binance_csv_path(symbol, interval)
    else:
        path = _yahoo_csv_path(symbol, interval)

    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file: {path}")

    df = pd.read_csv(path)
    # Standardisiere Spalten: ensure open,high,low,close,volume are present
    # Binance raw may use open_time etc.
    col_map = {}
    for c in df.columns:
        if c.lower() in ['open', 'high', 'low', 'close', 'volume']:
            col_map[c] = c.lower()
        if c.lower() in ['open_time', 'timestamp', 'time', 'date']:
            col_map[c] = 'open_time'
    df = df.rename(columns=col_map)
    required = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    if not all([c in df.columns for c in required]):
        raise ValueError(f"Missing required columns in {path} -> have {df.columns}")

    # Create label column (direction) but as code requested: don't store labels in raw, we'll create blocks and store with target
    df['target'] = (df['close'].astype(float).shift(-1) > df['open'].astype(float).shift(-1)).astype(int)
    df = df.dropna().reset_index(drop=True)

    features = df[['open','high','low','close','volume']].values
    blocks = []
    targets = []
    for i in range(len(df) - sequence_length + 1):
        block = features[i:i+sequence_length].reshape(-1)  # flatten block -> easier to save in CSV
        target = int(df['target'].iloc[i+sequence_length-1])
        blocks.append(block)
        targets.append(target)

    out_df = pd.DataFrame(blocks)
    out_df['target'] = targets
    out_fname = out_name or f"{source}_{symbol}_{interval}_seq{sequence_length}.csv"
    out_path = os.path.join(CANDLES_DIR, out_fname)
    out_df.to_csv(out_path, index=False)
    print(f"[DataManager] Saved blocks to {out_path}")
    return out_path

# -----------------------
# Small util: read config json
# -----------------------
def read_settings(settings_path: str):
    with open(settings_path, "r") as f:
        return json.load(f)

# End of data_manager.py
