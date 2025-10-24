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
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI (Relative Strength Index) berechnen
        
        Zeigt Momentum an (0-100):
        - RSI > 70: Überkauft (möglicher Rückgang)
        - RSI < 30: Überverkauft (möglicher Anstieg)
        - RSI = 50: Neutral
        
        Args:
            close: Series mit Close-Preisen
            period: Zeitraum (Standard: 14)
        
        Returns:
            Series mit RSI Werten (0-100)
        """
        # Preisänderungen
        delta = close.diff()
        
        # Gewinne und Verluste trennen
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Durchschnitte
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RS und RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(close: pd.Series, 
                      fast: int = 12, 
                      slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) berechnen
        
        Zeigt Trend und Momentum an:
        - MACD > Signal: Bullish (Aufwärtstrend)
        - MACD < Signal: Bearish (Abwärtstrend)
        - Histogram wächst: Momentum verstärkt sich
        
        Args:
            close: Series mit Close-Preisen
            fast: EMA-Period schnell (Standard: 12)
            slow: EMA-Period langsam (Standard: 26)
            signal: Signal Line Period (Standard: 9)
        
        Returns:
            Tuple: (MACD Line, Signal Line, Histogram)
        """
        # EMAs berechnen
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_ema(close: pd.Series, period: int = 20) -> pd.Series:
        """
        EMA (Exponential Moving Average) berechnen
        
        Args:
            close: Series mit Close-Preisen
            period: Zeitraum
        
        Returns:
            Series mit EMA Werten
        """
        return close.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(close: pd.Series, period: int = 20) -> pd.Series:
        """
        SMA (Simple Moving Average) berechnen
        
        Args:
            close: Series mit Close-Preisen
            period: Zeitraum
        
        Returns:
            Series mit SMA Werten
        """
        return close.rolling(window=period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, 
                                  period: int = 20, 
                                  std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands berechnen
        
        Zeigt Volatilität und Support/Resistance an:
        - Preis > Upper Band: Überkauft
        - Preis < Lower Band: Überverkauft
        
        Args:
            close: Series mit Close-Preisen
            period: Zeitraum für SMA (Standard: 20)
            std_dev: Standard Abweichungen (Standard: 2.0)
        
        Returns:
            Tuple: (Upper Band, Middle Band (SMA), Lower Band)
        """
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
        """
        ATR (Average True Range) berechnen - für Volatilität
        
        Args:
            high, low, close: Series mit Preisen
            period: Zeitraum (Standard: 14)
        
        Returns:
            Series mit ATR Werten
        """
        # True Range berechnen
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR als durchschnittlicher True Range
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_roc(close: pd.Series, period: int = 12) -> pd.Series:
        """
        ROC (Rate of Change) berechnen - Momentum
        
        Args:
            close: Series mit Close-Preisen
            period: Zeitraum (Standard: 12)
        
        Returns:
            Series mit ROC Werten (%)
        """
        change = close.diff(period)
        roc = (change / close.shift(period)) * 100
        
        return roc
    
    @classmethod
    def calculate_all_indicators(cls, 
                                 df: pd.DataFrame,
                                 ohlcv_columns: Dict[str, str] = None,
                                 indicators: Optional[list] = None) -> pd.DataFrame:
        """
        Berechnet ALLE Indikatoren auf einmal
        
        Args:
            df: DataFrame mit OHLCV Daten
            ohlcv_columns: Dict mit Spaltennamen 
                          {'open': 'open', 'high': 'high', ...}
            indicators: Liste von Indikatoren zum Berechnen
                       Optionen: ['RSI', 'MACD', 'EMA_12', 'EMA_26', 
                                 'SMA_20', 'BB', 'ATR', 'ROC']
        
        Returns:
            DataFrame mit allen berechneten Indikatoren
        
        Example:
            >>> df_with_indicators = calc.calculate_all_indicators(
            ...     df,
            ...     indicators=['RSI', 'MACD', 'EMA_12', 'EMA_26']
            ... )
        """
        # Standard Spaltennamen
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # Alle Indikatoren standard
        if indicators is None:
            indicators = ['RSI', 'MACD', 'EMA_12', 'EMA_26', 'SMA_20', 'BB', 'ATR']
        
        result_df = df.copy()
        
        # RSI
        if 'RSI' in indicators:
            result_df['RSI'] = cls.calculate_rsi(df[ohlcv_columns['close']], period=14)
        
        # MACD
        if 'MACD' in indicators:
            macd, signal, hist = cls.calculate_macd(df[ohlcv_columns['close']])
            result_df['MACD'] = macd
            result_df['MACD_signal'] = signal
            result_df['MACD_hist'] = hist
        
        # EMA
        if 'EMA_12' in indicators:
            result_df['EMA_12'] = cls.calculate_ema(df[ohlcv_columns['close']], period=12)
        
        if 'EMA_26' in indicators:
            result_df['EMA_26'] = cls.calculate_ema(df[ohlcv_columns['close']], period=26)
        
        # SMA
        if 'SMA_20' in indicators:
            result_df['SMA_20'] = cls.calculate_sma(df[ohlcv_columns['close']], period=20)
        
        # Bollinger Bands
        if 'BB' in indicators:
            upper, middle, lower = cls.calculate_bollinger_bands(df[ohlcv_columns['close']])
            result_df['BB_upper'] = upper
            result_df['BB_middle'] = middle
            result_df['BB_lower'] = lower
        
        # ATR
        if 'ATR' in indicators:
            result_df['ATR'] = cls.calculate_atr(
                df[ohlcv_columns['high']],
                df[ohlcv_columns['low']],
                df[ohlcv_columns['close']]
            )
        
        # ROC
        if 'ROC' in indicators:
            result_df['ROC'] = cls.calculate_roc(df[ohlcv_columns['close']], period=12)
        
        return result_df


# ============================================================================
# BEISPIEL-VERWENDUNG
# ============================================================================

if __name__ == "__main__":
    
    # Beispiel 1: Einzelne Indikatoren
    print("=" * 80)
    print("BEISPIEL 1: Einzelne Indikatoren berechnen")
    print("=" * 80)
    
    calc = TechnicalIndicatorCalculator()
    
    # Dummy-Daten
    np.random.seed(42)
    close_prices = pd.Series(np.cumsum(np.random.randn(100)) + 95000)
    
    # RSI berechnen
    rsi = calc.calculate_rsi(close_prices, period=14)
    print(f"RSI: {rsi.tail()}")
    
    # MACD berechnen
    macd, signal, hist = calc.calculate_macd(close_prices)
    print(f"\nMACD: {macd.tail()}")
    print(f"Signal: {signal.tail()}")
    
    # Beispiel 2: Alle Indikatoren
    print("\n" + "=" * 80)
    print("BEISPIEL 2: Alle Indikatoren auf einmal")
    print("=" * 80)
    
    # DataFrame mit OHLCV Daten
    dates = pd.date_range('2025-10-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100),
        'high': close_prices + np.abs(np.random.randn(100)) + 100,
        'low': close_prices - np.abs(np.random.randn(100)) - 100,
        'close': close_prices,
        'volume': np.random.uniform(1, 10, 100)
    }, index=dates)
    
    # Alle Indikatoren berechnen
    df_indicators = calc.calculate_all_indicators(
        df,
        indicators=['RSI', 'MACD', 'EMA_12', 'EMA_26', 'SMA_20', 'BB', 'ATR']
    )
    
    print(f"\nDataFrame Shape: {df_indicators.shape}")
    print(f"Spalten: {list(df_indicators.columns)}")
    print(f"\nLetzten 5 Zeilen:")
    print(df_indicators.tail())
    
    # Beispiel 3: NaN entfernen (wichtig!)
    print("\n" + "=" * 80)
    print("BEISPIEL 3: NaN entfernen (für Model-Input)")
    print("=" * 80)
    
    df_clean = df_indicators.dropna()
    print(f"Original: {len(df_indicators)} Zeilen")
    print(f"Nach dropna: {len(df_clean)} Zeilen")
    print(f"\nReady für CNN-LSTM Model!")
