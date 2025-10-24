# data/candle_builder.py
"""
Candle Builder / Feature helpers.
- Aggregation von Tick-Daten
- Feature-Engineering (einfach)
- Utility: create sequences (X,y) fuer Modelltraining
"""

import pandas as pd
import numpy as np

def add_basic_features(df):
    """
    Erwartet df mit open, high, low, close, volume, open_time
    Fügt return, body, range, sma_5, sma_20 als Beispiel hinzu.
    """
    df = df.copy().reset_index(drop=True)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df['ret'] = df['close'].pct_change().fillna(0)
    df['body'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    df['range'] = (df['high'] - df['low']) / (df['open'] + 1e-9)
    df['sma_5'] = df['close'].rolling(5).mean().fillna(method='bfill')
    df['sma_20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df = df.fillna(0)
    return df

def create_sequences_from_candles(df, seq_len=3, feature_cols=None, label_mode="direction"):
    """
    Erzeugt (X,y) Arrays für Training.
    feature_cols default: open, high, low, close, volume + engineered if present
    label_mode: 'direction' or 'regression' (next close)
    """
    df = df.copy().reset_index(drop=True)
    if feature_cols is None:
        # prefer engineered columns if present
        candidates = ['open','high','low','close','volume','ret','body','range','sma_5','sma_20']
        feature_cols = [c for c in candidates if c in df.columns]

    if label_mode == "direction":
        df['label'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(int)
    elif label_mode == "regression":
        df['label'] = df['close'].shift(-1)
    else:
        raise ValueError("label_mode unknown")

    df = df.dropna().reset_index(drop=True)
    X = []
    y = []
    for i in range(len(df) - seq_len + 1):
        block = df[feature_cols].iloc[i:i+seq_len].values
        X.append(block)
        y.append(df['label'].iloc[i+seq_len-1])
    X = np.array(X)
    y = np.array(y)
    return X, y
