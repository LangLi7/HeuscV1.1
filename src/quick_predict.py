# src/eval_last7d.py
# -*- coding: utf-8 -*-
"""
Heusc V1.1 – Evaluation der letzten 7 Tage (1m Candles)
Lädt automatisch Modell + Indikatoren + Live-Daten von Binance,
berechnet Vorhersagen und führt einen Quick-Backtest aus.
"""

import os
import sys
import json
import time
import math
import argparse
import platform
import importlib
import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Paket-Imports (funktioniert als Modul und Direktstart)
# ----------------------------------------------------------

# --- Preprocessing: Skaliere wie im Training (MinMaxScaler) ---
from sklearn.preprocessing import MinMaxScaler

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.log import log_error_once, log_info, log_exception
    from src.loader import SystemLoader, TFTrainingLoader
    from src.indicator_calculator import TechnicalIndicatorCalculator
    from src.invest_profit_calculator import Investment
    from src.model_cnn_lstm import build_callbacks, build_hybrid_cnn_lstm, load_settings
else:
    from .log import log_error_once, log_info, log_exception
    from .loader import SystemLoader, TFTrainingLoader
    from .indicator_calculator import TechnicalIndicatorCalculator
    from .invest_profit_calculator import Investment
    from .model_cnn_lstm import build_callbacks, build_hybrid_cnn_lstm, load_settings

# TensorFlow laden
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    raise RuntimeError("TensorFlow/Keras nicht installiert. Bitte zuerst: pip install tensorflow") from e

# Optional-Module
try:
    import src.indicator_calculator as ic
except Exception:
    ic = None
try:
    import src.data_loader as dl
except Exception:
    dl = None

# ----------------------------------------------------------
# .env laden (automatisch aus Repo-Root)
# ----------------------------------------------------------
from dotenv import load_dotenv
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "hybrid_cnn_lstm.keras")
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1m"
LOOKBACK_DAYS = 7
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LIVE_LOG_PATH = os.path.join(LOG_DIR, "live_predictions.log")
EVAL_JSON = os.path.join(LOG_DIR, "model_eval.json")
TRADE_LOG_JSON = os.path.join(LOG_DIR, "trade_log.json")

THRESHOLD = 0.5
TAKER_FEE_PCT = 0.001
SPREAD_PCT = 0.0005

# ----------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------
def log(msg: str):
    """Konsolen- und Datei-Logger"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [EvalLast7d] {msg}"
    print(line)
    with open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_read_settings():
    """Versucht, config/settings.json zu finden."""
    candidates = [
        os.path.join(ROOT, "config", "settings.json"),
        os.path.join(ROOT, "settings.json"),
        os.path.join(ROOT, "data", "settings", "data_config.json"),
        os.path.join(ROOT, "data", "settings", "model_config.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                try:
                    return json.load(f), p
                except Exception as e:
                    log(f"Fehler beim Laden {p}: {e}")
    return None, None


# ----------------------------------------------------------
# Live-Marktdaten laden (Binance)
# ----------------------------------------------------------
def load_market_data(symbol: str, start_utc: datetime, end_utc: datetime, interval: str, csv_fallback: str = None):
    """
    Holt Marktdaten aus data/binance_data.py
    Verwendet fetch_historical() und wandelt Spalten automatisch um.
    """
    import pandas as pd
    import data.binance_data as bdata

    log(f"Fetching Binance data for {symbol} {interval} from {start_utc} to {end_utc}")

    try:
        df = bdata.fetch_historical(
            symbol=symbol,
            interval=interval,
            start_str=start_utc.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_utc.strftime("%Y-%m-%d %H:%M:%S"),
            save=False
        )

        # ✅ Spaltenanpassung für Binance-Format
        if "open_time" in df.columns:
            df = df.rename(columns={
                "open_time": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })[["timestamp", "open", "high", "low", "close", "volume"]]
            # Timestamp in datetime umwandeln
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        if _is_valid_ohlcv(df):
            log(f"✅ Binance Fetch erfolgreich: {len(df)} Zeilen geladen.")
            return _normalize_df(df)
        else:
            raise ValueError(f"Ungültige Spalten nach Umwandlung: {list(df.columns)}")

    except Exception as e:
        log(f"❌ Binance Fetch fehlgeschlagen: {e}")
        raise RuntimeError("❌ Kein Datenfetch möglich – bitte prüfe Binance API oder Parameter.") from e

def _is_valid_ohlcv(df):
    return isinstance(df, pd.DataFrame) and \
           {"timestamp", "open", "high", "low", "close", "volume"} <= set(df.columns)


def _normalize_df(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


# ----------------------------------------------------------
# Features & Indikatoren
# ----------------------------------------------------------
def apply_indicators(df, settings):
    if ic and hasattr(ic, "calculate_indicators"):
        try:
            out = ic.calculate_indicators(df, settings)
            if isinstance(out, pd.DataFrame):
                return out
        except Exception as e:
            log(f"indicator_calculator Fehler: {e}")
    # Fallback-Minimalfeatures
    eps = 1e-9
    df = df.copy()
    df["ret1"] = (df["close"] / df["close"].shift(1) - 1).fillna(0)
    df["hl_range"] = ((df["high"] - df["low"]) / (df["close"].shift(1) + eps)).fillna(0)
    df["ocl_spread"] = ((df["open"] - df["close"]) / (df["close"].shift(1) + eps)).fillna(0)
    return df


def align_features(df, seq_len, n_feat):
    drop_cols = {"timestamp", "symbol"}
    num_df = df[[c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]].copy()
    feats = list(num_df.columns)
    # pad oder trim
    if len(feats) < n_feat:
        for k in range(n_feat - len(feats)):
            pad = f"pad_{k}"
            num_df[pad] = 0.0
            feats.append(pad)
    elif len(feats) > n_feat:
        feats = feats[:n_feat]
        num_df = num_df[feats]
    arr = num_df.values.astype(np.float32)
    X, idx = [], []
    for i in range(seq_len, len(df)):
        X.append(arr[i - seq_len:i, :])
        idx.append(df["timestamp"].iloc[i])
    return np.stack(X, axis=0), feats, idx


# ----------------------------------------------------------
# Backtest
# ----------------------------------------------------------
def quick_backtest(close, signals):
    bal, pos, eq_list, trades, wins = 1000.0, 0.0, [], 0, 0
    last_buy = None
    for price, sig in zip(close.values, signals):
        buy = price * (1 + SPREAD_PCT)
        sell = price * (1 - SPREAD_PCT)
        long = sig > THRESHOLD
        if long and pos == 0:
            cash = bal * (1 - TAKER_FEE_PCT)
            pos = cash / buy
            bal = 0.0
            last_buy = buy
            trades += 1
        elif not long and pos > 0:
            cash = pos * sell * (1 - TAKER_FEE_PCT)
            if sell > (last_buy or sell):
                wins += 1
            bal, pos, last_buy = cash, 0.0, None
            trades += 1
        eq_list.append(bal + pos * sell)
    ret = eq_list[-1] / 1000 - 1
    winrate = wins / trades if trades else 0
    mdd, peak = 0, -1e9
    for e in eq_list:
        peak = max(peak, e)
        dd = e / peak - 1
        mdd = min(mdd, dd)
    return {
        "final_balance": round(eq_list[-1], 2),
        "return_pct": round(ret * 100, 2),
        "trades": trades,
        "winrate": round(winrate * 100, 2),
        "max_drawdown_pct": round(mdd * 100, 2),
        "equity_points": eq_list[-200:]
    }


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--interval", default=DEFAULT_INTERVAL)
    ap.add_argument("--days", type=int, default=LOOKBACK_DAYS)
    args = ap.parse_args()

    host, osys = platform.node(), f"{platform.system()} {platform.release()} ({platform.machine()})"
    now, start = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=args.days)
    log(f"Start | Host={host} | OS={osys} | Symbol={args.symbol} | Interval={args.interval} | Window={args.days}d")

    model = load_model(args.model)
    in_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    _, seq_len, n_feat = in_shape
    log(f"Model input: seq_len={seq_len}, n_feat={n_feat}")

    settings, settings_path = safe_read_settings()
    log(f"settings.json: {settings_path or 'nicht gefunden'}")

    df = load_market_data(args.symbol, start, now, args.interval)
    log(f"Datenpunkte: {len(df)} | Zeitraum {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")

    df_feat = apply_indicators(df, settings or {})
    X, feats, idx = align_features(df_feat, seq_len, n_feat)
    if X.shape[0] == 0:
        log("Zu wenige Samples -> Abbruch")
        return
    log(f"Samples: {X.shape[0]} | Features: {len(feats)}")

    scores = model.predict(X, verbose=0).squeeze()
    scores = np.array([scores]) if scores.ndim == 0 else scores
    log("Letzte 10 Scores:")
    for t, s in list(zip(idx, scores))[-10:]:
        log(f"{t.isoformat()} | score={s:.4f} | decision={'BUY' if s>THRESHOLD else 'SELL'}")

    close_series = df.set_index("timestamp")["close"].reindex(idx).ffill().bfill()
    results = quick_backtest(close_series, scores)

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump({"eval": {"symbol": args.symbol, "seq_len": int(seq_len), "n_feat": int(n_feat)},
                   "backtest_summary": results}, f, indent=2)
    log(f"Fertig. ROI={results['return_pct']}% | Trades={results['trades']} | Winrate={results['winrate']}% | MDD={results['max_drawdown_pct']}%")
    log(f"Details in {EVAL_JSON}")


if __name__ == "__main__":
    main()
