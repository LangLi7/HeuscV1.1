import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance.client import Client
import yfinance as yf
import time
import math

# Lade .env aus dem übergeordneten Verzeichnis
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)


class DataFetcher:
    def __init__(self, source='binance', symbol='BTCUSDT', interval='1m', output_dir='data/', chunk_size=1000):
        self.source = source.lower()
        self.symbol = symbol
        self.interval = interval
        self.chunk_size = chunk_size
        self.output_dir = os.path.join(output_dir, self.source)
        os.makedirs(self.output_dir, exist_ok=True)

        if self.source == 'binance':
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_SECRET_KEY')
            if not api_key or not api_secret:
                raise ValueError("Binance API Keys fehlen in .env!")
            self.client = Client(api_key, api_secret)
        elif self.source == 'yahoo':
            pass
        else:
            raise ValueError("Ungültige Datenquelle! ('binance' oder 'yahoo')")

    def fetch_chunked_data(self, start_date, end_date):
        """
        Holt Daten in kleinen Chunks (wegen API-Limits) und speichert sie zusammengeführt als CSV.
        """
        print(f"[INFO] Lade Daten von {self.source.upper()} für {self.symbol}...")

        if self.source == 'binance':
            return self._fetch_binance_data(start_date, end_date)
        elif self.source == 'yahoo':
            return self._fetch_yahoo_data(start_date, end_date)

    def _fetch_binance_data(self, start_date, end_date):
        df_list = []
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        while start_ts < end_ts:
            candles = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=self.chunk_size
            )

            if not candles:
                break

            temp_df = pd.DataFrame(candles, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit='ms')
            df_list.append(temp_df)
            start_ts = int(temp_df["timestamp"].iloc[-1].timestamp() * 1000)

            time.sleep(0.1)

        df = pd.concat(df_list, ignore_index=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        csv_path = os.path.join(self.output_dir, f"{self.symbol}_{self.interval}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Binance-Daten gespeichert unter: {csv_path}")
        return df

    def _fetch_yahoo_data(self, start_date, end_date):
        data = yf.download(
            self.symbol,
            start=start_date,
            end=end_date,
            interval=self.interval
        )

        data.reset_index(inplace=True)
        data.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        csv_path = os.path.join(self.output_dir, f"{self.symbol}_{self.interval}.csv")
        data.to_csv(csv_path, index=False)
        print(f"[OK] Yahoo-Daten gespeichert unter: {csv_path}")
        return data
