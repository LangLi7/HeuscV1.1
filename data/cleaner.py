import pandas as pd
import os
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

df = pd.read_csv("data/binance/" + "BTCUSDT-1m-1y-binance-2025-09-13_11-37-53.csv")
df.drop(columns=['color'], inplace=True)
df.to_csv('BTCUSDT-1m-1y-binance-2025-09-13_11-37-53.csv', index=False)
