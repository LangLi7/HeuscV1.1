import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Importiere deine Datenquellenmodule (du hast sie in data/binance_data.py und data/yahoo_data.py)
from data.binance_data import BinanceFetcher
from data.yahoo_data import YahooFetcher

# ----------------------------------------------
# Hilfsfunktion: JSON-Config laden
# ----------------------------------------------
def load_config(config_path="config/settings.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

# ----------------------------------------------
# Daten normalisieren
# ----------------------------------------------
def normalize_data(df, feature_cols, scale_per_symbol=True):
    scalers = {}
    scaled_data = pd.DataFrame(index=df.index)

    if scale_per_symbol and "symbol" in df.columns:
        for sym in df["symbol"].unique():
            sym_data = df[df["symbol"] == sym]
            scaler = MinMaxScaler()
            scaled_sym = pd.DataFrame(
                scaler.fit_transform(sym_data[feature_cols]),
                columns=feature_cols,
                index=sym_data.index
            )
            scaled_sym["symbol"] = sym
            scaled_data = pd.concat([scaled_data, scaled_sym])
            scalers[sym] = scaler
    else:
        scaler = MinMaxScaler()
        scaled_all = pd.DataFrame(
            scaler.fit_transform(df[feature_cols]),
            columns=feature_cols,
            index=df.index
        )
        scaled_data = scaled_all
        scalers["global"] = scaler

    return scaled_data.sort_index(), scalers

# ----------------------------------------------
# Candle Chunks generieren
# ----------------------------------------------
def create_chunks(data, block_size):
    X, y = [], []
    features = ["open", "high", "low", "close", "volume"]

    for i in range(len(data) - block_size - 1):
        block = data[features].iloc[i:i + block_size].values
        next_close = data["close"].iloc[i + block_size]
        current_close = data["close"].iloc[i + block_size - 1]
        direction = 1 if next_close > current_close else 0
        X.append(block)
        y.append(direction)

    return np.array(X), np.array(y)

# ----------------------------------------------
# Split in Train/Val/Test
# ----------------------------------------------
def split_data(X, y, split_config):
    n = len(X)
    train_end = int(n * split_config["train"])
    val_end = train_end + int(n * split_config["validation"])

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ----------------------------------------------
# Haupt-Preprocessing Pipeline
# ----------------------------------------------
def preprocess():
    cfg = load_config()
    data_cfg = cfg["data"]

    sources = data_cfg["sources"]
    symbols = data_cfg["symbols"]
    features = data_cfg["features"]
    interval = data_cfg["interval"]
    period = data_cfg["period"]
    chunk_days = data_cfg["chunk_days"]
    block_size = data_cfg["candle_block_size"]
    split_cfg = data_cfg["split"]
    paths = data_cfg["paths"]
    preprocess_cfg = data_cfg.get("preprocessing", {})

    os.makedirs(paths["csv"], exist_ok=True)

    all_data = []

    for source in sources:
        for symbol in symbols:
            print(f"📦 Lade Daten: {symbol} von {source}...")

            if source == "binance":
                fetcher = BinanceFetcher(symbol, interval, period)
                df = fetcher.get_data()
            elif source == "yahoo":
                fetcher = YahooFetcher(symbol, interval, period)
                df = fetcher.get_data()
            else:
                raise ValueError(f"Unbekannte Datenquelle: {source}")

            df["symbol"] = symbol
            df = df[["symbol"] + features]
            df.dropna(inplace=True)
            all_data.append(df)

    # Zusammenführen und Normalisieren
    combined = pd.concat(all_data).sort_index()
    scaled, scalers = normalize_data(
        combined, features,
        scale_per_symbol=preprocess_cfg.get("scale_per_symbol", True)
    )

    # Chunks bilden
    X, y = create_chunks(scaled, block_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, split_cfg)

    # Speichern als .npz (kompakt)
    np.savez(
        Path(paths["csv"]) / "preprocessed_data.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    print("✅ Preprocessing abgeschlossen!")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        "scalers": scalers,
        "shapes": {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape
        }
    }

if __name__ == "__main__":
    preprocess()
