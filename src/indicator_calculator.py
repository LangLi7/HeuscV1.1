# ============================================================================
# TECHNICAL INDICATOR CALCULATOR
# Berechnet RSI, MACD, EMA, SMA, Bollinger Bands, ATR aus OHLCV Daten
# ============================================================================
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class TechnicalIndicatorCalculator:
    """
    Berechnet technische Indikatoren aus OHLCV Daten.
    Keine externe Library nötig - alles mit Pandas/NumPy

    Unterstützte Indikatoren:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - EMA (Exponential Moving Average)
    - SMA (Simple Moving Average)
    - Bollinger Bands
    - ATR (Average True Range)
    - ROC (Rate of Change)
    """

    # ============================================================================
    # GRUNDLEGENDER SICHERHEITSCHECK
    # ============================================================================
    @staticmethod
    def _validate_df(df: pd.DataFrame, min_len: int = 20) -> bool:
        """
        Überprüft, ob das DataFrame ausreichend groß ist.
        Gibt False zurück, wenn es leer oder zu kurz ist.
        """
        if df is None or df.empty:
            print("[IndicatorCalc] ⚠ Kein gültiges DataFrame – überspringe Berechnung.")
            return False
        if len(df) < min_len:
            print(f"[IndicatorCalc] ⚠ Zu wenige Zeilen ({len(df)}) für Indikatorberechnung – überspringe.")
            return False
        return True

    # ============================================================================
    # INDIVIDUELLE INDIKATOREN
    # ============================================================================
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(close: pd.Series,
                       fast: int = 12,
                       slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_ema(close: pd.Series, period: int = 20) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(close: pd.Series, period: int = 20) -> pd.Series:
        return close.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_bands(close: pd.Series,
                                  period: int = 20,
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    @staticmethod
    def calculate_atr(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 12) -> pd.Series:
        change = close.diff(period)
        roc = (change / close.shift(period)) * 100
        return roc

    # ============================================================================
    # GESAMTE BERECHNUNG MIT JSON-STEUERUNG
    # ============================================================================
    @classmethod
    def calculate_all_indicators(cls,
                                 df: pd.DataFrame,
                                 ohlcv_columns: Dict[str, str] = None,
                                 indicators: Optional[list] = None) -> pd.DataFrame:
        """
        Berechnet ALLE Indikatoren auf einmal.
        Dynamisch anpassbar über settings.json (data.indicators)

        Args:
            df: DataFrame mit OHLCV Daten
            ohlcv_columns: Mapping der Spaltennamen
            indicators: Liste aktiver Indikatoren, z. B. ['RSI', 'MACD', 'EMA_12']

        Returns:
            DataFrame mit allen berechneten Indikatoren
        """
        # Sicherheitscheck
        if not cls._validate_df(df):
            return df

        # Standardspalten
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

        # Standardindikatoren
        if indicators is None:
            indicators = ['RSI', 'MACD', 'EMA_12', 'EMA_26', 'SMA_20', 'BB', 'ATR']

        result_df = df.copy()

        # Dynamische Verarbeitung
        for name in indicators:
            try:
                if name == "RSI":
                    result_df["RSI"] = cls.calculate_rsi(df[ohlcv_columns["close"]])
                elif name == "MACD":
                    macd, signal, hist = cls.calculate_macd(df[ohlcv_columns["close"]])
                    result_df["MACD"], result_df["MACD_signal"], result_df["MACD_hist"] = macd, signal, hist
                elif name.startswith("EMA_"):
                    period = int(name.split("_")[1])
                    result_df[name] = cls.calculate_ema(df[ohlcv_columns["close"]], period=period)
                elif name.startswith("SMA_"):
                    period = int(name.split("_")[1])
                    result_df[name] = cls.calculate_sma(df[ohlcv_columns["close"]], period=period)
                elif name == "BB":
                    upper, middle, lower = cls.calculate_bollinger_bands(df[ohlcv_columns["close"]])
                    result_df["BB_upper"], result_df["BB_middle"], result_df["BB_lower"] = upper, middle, lower
                elif name == "ATR":
                    result_df["ATR"] = cls.calculate_atr(
                        df[ohlcv_columns["high"]],
                        df[ohlcv_columns["low"]],
                        df[ohlcv_columns["close"]],
                    )
                elif name == "ROC":
                    result_df["ROC"] = cls.calculate_roc(df[ohlcv_columns["close"]])
            except Exception as e:
                print(f"[IndicatorCalc] ⚠ Fehler bei {name}: {e}")

        return result_df


# ============================================================================
# BEISPIEL-VERWENDUNG MIT SETTINGS.JSON UND REALER CSV
# ============================================================================
if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Basisverzeichnis relativ zur main.py (eine Ebene höher als src)
    #project_root = Path(__file__).resolve().parent.parent # geht 1 Ordner hoch (von src → Projekt)
    #settings_path = project_root / "config" / "settings.json"
    base_dir = Path(__file__).resolve().parent.parent  
    settings_path = base_dir / "config" / "settings.json"


    print("=" * 80)
    print("TECHNICAL INDICATOR CALCULATOR – TESTLAUF MIT CSV-DATEIEN")
    print("=" * 80)

    # Pfad zur settings.json
    settings_path = Path("config/settings.json")

    if not settings_path.exists():
        raise FileNotFoundError("⚠ settings.json nicht gefunden unter: config/settings.json")

    # Settings laden
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    data_cfg = settings.get("data", {})
    custom_csvs = data_cfg.get("custom_csv", data_cfg.get("paths", {}).get("custom_csv", {}))
    indicators = data_cfg.get("indicators", ["RSI", "MACD", "EMA_12", "SMA_20", "BB"])
    max_rows_preview = 10  # nur die ersten 10 Zeilen anzeigen

    calc = TechnicalIndicatorCalculator()

    if not custom_csvs:
        print("⚠ Keine CSV-Dateien in settings.json definiert (custom_csv). Beende Test.")
        exit(0)

    # Jede CSV nacheinander laden und verarbeiten
    for symbol, csv_path in custom_csvs.items():
        path = Path(csv_path)
        if not path.exists():
            print(f"[{symbol}] ⚠ Datei nicht gefunden: {path}")
            continue

        print("\n" + "=" * 80)
        print(f"[{symbol}] Lade CSV: {path}")
        print("=" * 80)

        # CSV laden
        df = pd.read_csv(path)
        print(f"[{symbol}] CSV geladen – {len(df)} Zeilen, Spalten: {list(df.columns)}")

        # Indikatoren berechnen
        df_ind = calc.calculate_all_indicators(df, indicators=indicators)
        df_ind_clean = df_ind.dropna()

        # Nur die ersten Zeilen zeigen
        print(f"\n[{symbol}] Vorschau der berechneten Indikatoren (erste {max_rows_preview} Zeilen):")
        print(df_ind_clean.head(max_rows_preview))

        print(f"\n[{symbol}] Gesamt: {len(df_ind)} Zeilen → Nach dropna: {len(df_ind_clean)} gültige Zeilen.")
        print(f"[{symbol}] Fertig! Ready für CNN-LSTM-Training.\n")

    print("=" * 80)
    print("ALLE CSVs VERARBEITET")
    print("=" * 80)

