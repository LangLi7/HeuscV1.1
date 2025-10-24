/data
 ├── __init__.py
 ├── data_manager.py         ← zentrale Verwaltung (Scheduler, Cache, Save)
 ├── binance_data.py         ← Binance-spezifische API-Logik (Chunk + Live)
 ├── yahoo_data.py           ← Yahoo-spezifische API-Logik (Chunk + Update)
 ├── preprocess.py           ← Candle-Preprocessing, Scaling, Blockbildung
 └── settings/
      ├── data_config.json
      └── model_config.json
