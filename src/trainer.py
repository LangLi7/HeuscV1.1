
"""Training pipeline for the hybrid CNN-LSTM model.

The trainer loads the configuration from ``config/settings.json``, enriches the
market data with technical indicators and runs a paper-trading style simulation
based on the predictions.  It couples the data processing pipeline with the
model declared in :mod:`model_cnn_lstm` and provides a single high-level
:class:`MarketSimulationTrainer` to orchestrate the workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import datetime
import os
import json
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model

from .log import log_error_once, log_info, log_exception
from .loader import SystemLoader, TFTrainingLoader
from .indicator_calculator import TechnicalIndicatorCalculator
from .invest_profit_calculator import Investment
from .model_cnn_lstm import build_callbacks, build_hybrid_cnn_lstm, load_settings

def log_error_once(message: str, logs_root: str = "logs"):
    """Loggt einen Fehler einmal pro Tag in logs/errors_<YYYY-MM-DD>.log."""
    os.makedirs(logs_root, exist_ok=True)
    today = datetime.date.today().isoformat()
    log_file = os.path.join(logs_root, f"errors_{today}.log")

    # Datei einlesen (wenn vorhanden)
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if any(message in line for line in lines):
            # Fehler wurde heute bereits geloggt → überspringen
            return

    # Anhängen
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

@dataclass
class DatasetSplit:
    """Container for train/validation/test splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: List[Dict[str, float]]


class MarketSimulationTrainer:
    """High level trainer that reads configuration and runs the whole pipeline."""

    def __init__(self, settings_path: Path | str = Path("config/settings.json")) -> None:
        self.settings_path = Path(settings_path)
        self.settings = load_settings(self.settings_path)
        self.data_config = self.settings.get("data", {})
        self.model_config = self.settings.get("model", {})
        self.training_config = self.settings.get("training", {})
        self.simulation_config = self.settings.get("simulation", {})

        self.feature_columns: List[str] = []
        self.scaler: Optional[MinMaxScaler | StandardScaler] = None
        self.model: Optional[Model] = None
        self.history = None
        self.test_metadata: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Data ingestion & preprocessing
    # ------------------------------------------------------------------
    def _detect_csv_path(self, symbol: str, source: str) -> Path:
        """
        Prüft zuerst, ob ein Custom-Pfad im settings.json angegeben ist.
        Fällt ansonsten auf die Standard-Suchlogik zurück.
        """
        base_dir = Path(self.data_config.get("paths", {}).get("csv", "data"))
        custom_map = self.data_config.get("paths", {}).get("custom_csv", {})

        # Prüfe ob im JSON ein direkter Pfad angegeben ist
        if symbol in custom_map:
            custom_path = Path(custom_map[symbol])
            if custom_path.exists():
                print(f"[Trainer] ✅ Custom CSV erkannt für {symbol}: {custom_path}")
                return custom_path
            else:
                raise FileNotFoundError(f"[Trainer] ⚠ Custom CSV-Datei nicht gefunden: {custom_path}")

        # Standardpfade fallback
        candidates = [
            base_dir / source / f"{symbol}_{self.data_config.get('interval', '1m')}.csv",
            base_dir / f"{symbol}_{self.data_config.get('interval', '1m')}.csv",
            base_dir / source / f"{symbol}.csv",
            base_dir / f"{symbol}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                print(f"[Trainer] 📁 Verwende CSV-Datei: {candidate}")
                return candidate

        raise FileNotFoundError(f"[Trainer] ❌ Keine CSV-Datei für Symbol '{symbol}' und Quelle '{source}' gefunden.")

    def _load_raw_data(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        sources = self.data_config.get("sources", [])
        symbols = self.data_config.get("symbols", [])

        if not sources or not symbols:
            raise ValueError("'sources' und 'symbols' müssen in settings.json definiert sein.")

        for source in sources:
            for symbol in symbols:
                try:
                    csv_path = self._detect_csv_path(symbol, source)
                    df = pd.read_csv(csv_path)
                    frames.append(df)  # 🔹 FEHLTE bisher!
                    log_info(f"[Trainer] ✅ CSV geladen: {csv_path}")
                except FileNotFoundError as e:
                    log_error_once(f"[Trainer] ❌ {source}:{symbol} - Datei fehlt",
                                self.data_config.get("paths", {}).get("logs", "logs"))
                    print(f"[Trainer] ⚠ {e} → übersprungen.")
                    continue

        # 🔹 Fallback, falls keine CSV erfolgreich geladen wurde
        if not frames:
            log_error_once("[Trainer] ❌ Keine gültigen Datenquellen gefunden – keine CSVs geladen.",
                        self.data_config.get("paths", {}).get("logs", "logs"))
            print("[Trainer] ❌ Keine gültigen Datenquellen gefunden – CSV-Liste ist leer.")
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values(["symbol", "timestamp"], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.log import log_info, log_error_once, log_exception
        indicator_calc = TechnicalIndicatorCalculator()
        enriched_frames: List[pd.DataFrame] = []

        if df.empty:
            log_error_once("[Trainer] ⚠ Keine Daten zum Berechnen von Indikatoren – DataFrame ist leer.",
                        self.data_config.get("paths", {}).get("logs", "logs"))
            print("[Trainer] ⚠ Keine Daten zum Berechnen von Indikatoren – DataFrame ist leer.")
            return df  # einfach leeren Frame zurückgeben

        for symbol, group in df.groupby("symbol", sort=False):
            try:
                enriched = indicator_calc.calculate_all_indicators(group)
                if enriched is not None and not enriched.empty:
                    enriched_frames.append(enriched)
                    log_info(f"[Trainer] ✅ Indikatoren berechnet für {symbol}")
                else:
                    log_error_once(f"[Trainer] ⚠ Keine Indikatoren für {symbol} berechnet (leer).",
                                self.data_config.get('paths', {}).get('logs', 'logs'))
            except Exception as e:
                log_exception(e)
                print(f"[Trainer] ⚠ Fehler bei Indikatorenberechnung für {symbol}: {e}")

        if not enriched_frames:
            log_error_once("[Trainer] ❌ Keine Indikatoren-Frames erzeugt – concat abgebrochen.",
                        self.data_config.get("paths", {}).get("logs", "logs"))
            print("[Trainer] ❌ Keine Indikatoren-Frames erzeugt – concat abgebrochen.")
            return df  # Rückgabe: Originaldaten ohne Indikatoren

        enriched_df = pd.concat(enriched_frames, ignore_index=True)
        enriched_df.sort_values(["symbol", "timestamp"], inplace=True)
        return enriched_df 

    def _scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        preprocessing_cfg = self.data_config.get("preprocessing", {})
        scaler_type = preprocessing_cfg.get("scaler", "minmax").lower()
        scale_per_symbol = bool(preprocessing_cfg.get("scale_per_symbol", False))

        if scaler_type == "standard":
            scaler_cls = StandardScaler
        else:
            scaler_cls = MinMaxScaler

        feature_cols = self.data_config.get("features", ["open", "high", "low", "close", "volume"])
        indicator_cols = [col for col in df.columns if col not in feature_cols + ["timestamp", "symbol"]]
        all_features = feature_cols + indicator_cols

        if scale_per_symbol:
            scaled_groups = []
            for _, group in df.groupby("symbol", sort=False):
                scaler = scaler_cls()
                scaled_values = scaler.fit_transform(group[all_features])
                scaled_group = group.copy()
                scaled_group[all_features] = scaled_values
                scaled_groups.append(scaled_group)
            scaled_df = pd.concat(scaled_groups, ignore_index=True)
            self.scaler = None
        else:
            self.scaler = scaler_cls()
            scaled_values = self.scaler.fit_transform(df[all_features])
            scaled_df = df.copy()
            scaled_df[all_features] = scaled_values

        self.feature_columns = all_features
        scaled_df.sort_values(["symbol", "timestamp"], inplace=True)
        scaled_df.dropna(inplace=True)
        scaled_df.reset_index(drop=True, inplace=True)
        return scaled_df, all_features

    def _build_sequences(
        self, df: pd.DataFrame, feature_cols: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        block_size = int(self.data_config.get("candle_block_size", 60))
        target_cfg = self.data_config.get("target", {})
        future_steps = int(target_cfg.get("future_steps", 1))
        threshold = float(target_cfg.get("threshold", 0.0))
        method = target_cfg.get("method", "binary").lower()

        sequences: List[np.ndarray] = []
        labels: List[float] = []
        metadata: List[Dict[str, float]] = []

        for _, group in df.groupby("symbol", sort=False):
            values = group[feature_cols].to_numpy(dtype=np.float32)
            closes = group["close"].to_numpy(dtype=np.float32)

            for start in range(0, len(group) - block_size - future_steps + 1):
                end = start + block_size
                future_index = end + future_steps - 1
                window = values[start:end]
                current_price = float(closes[end - 1])
                future_price = float(closes[future_index])
                change = (future_price - current_price) / current_price if current_price else 0.0

                if method == "binary":
                    label = 1.0 if change > threshold else 0.0
                elif method == "regression":
                    label = change
                else:
                    raise ValueError(f"Unbekannte Zielmethode: {method}")

                sequences.append(window)
                labels.append(label)
                metadata.append(
                    {
                        "current_price": current_price,
                        "future_price": future_price,
                        "change": change,
                        "symbol": group["symbol"].iloc[0],
                        "start_index": float(start),
                        "end_index": float(future_index),
                    }
                )

        X = np.asarray(sequences, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        return X, y, metadata

    def _split_dataset(self, X: np.ndarray, y: np.ndarray, meta: List[Dict[str, float]]) -> DatasetSplit:
        train_ratio = float(self.data_config.get("split", {}).get("train", 0.8))
        val_ratio = float(self.data_config.get("split", {}).get("validation", 0.1))

        total = len(X)
        if total == 0:
            raise ValueError("Es konnten keine Sequenzen erzeugt werden - prüfen Sie die Datenbasis.")

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        meta_test = meta[val_end:]

        if len(X_val) == 0 or len(X_test) == 0:
            raise ValueError(
                "Der Datensatz ist zu klein für die geforderten Splits. Bitte mehr Daten bereitstellen oder die Split-Werte anpassen."
            )

        return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test, meta_test)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def prepare_dataset(self) -> DatasetSplit:
        raw_df = self._load_raw_data()
        enriched_df = self._apply_indicators(raw_df)
        scaled_df, features = self._scale_features(enriched_df)
        X, y, meta = self._build_sequences(scaled_df, features)
        splits = self._split_dataset(X, y, meta)
        self.test_metadata = splits.meta_test
        return splits

    def _determine_class_weights(self, y_train: np.ndarray) -> Optional[Dict[int, float]]:
        class_weight_cfg = self.training_config.get("class_weight")
        if class_weight_cfg == "balanced":
            classes = np.unique(y_train)
            if len(classes) <= 1:
                return None
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.astype(int))
            return {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        if isinstance(class_weight_cfg, dict):
            return {int(k): float(v) for k, v in class_weight_cfg.items()}
        return None

    def train(self, dataset: Optional[DatasetSplit] = None) -> Model:
        if dataset is None:
            dataset = self.prepare_dataset()

        # === Parameter aus settings.json ===
        gpu_index = int(self.training_config.get("gpu_index", 0))
        use_gpu = bool(self.training_config.get("use_gpu", True))
        use_mixed = bool(self.training_config.get("mixed_precision", True))
        use_xla = bool(self.training_config.get("xla", True))
        batch_size = int(self.training_config.get("batch_size", 64))
        shuffle_buffer = int(self.training_config.get("shuffle_buffer", 2048))
        prefetch_val = self.training_config.get("prefetch", "auto")

        print("\n⚙️ TensorFlow Training Setup (HEUSC Optimiert)\n")
        print("🧩 HEUSC TRAINING CONFIG")
        print(f"GPU Index: {gpu_index}")
        print(f"Mixed Precision: {'✅' if use_mixed else '❌'}")
        print(f"XLA Compiler: {'✅' if use_xla else '❌'}")
        print(f"Batch Size: {batch_size} | Prefetch: {prefetch_val}")
        print(f"Shuffle Buffer: {shuffle_buffer}")
        print("-" * 50)

        # === GPU-Konfiguration ===
        gpus = tf.config.list_physical_devices("GPU")
        if use_gpu and gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_index], "GPU")
                print(f"✅ GPU {gpu_index} aktiviert: {gpus[gpu_index].name}")
            except Exception as e:
                print(f"⚠️ Fehler beim Setzen der GPU {gpu_index}: {e}")
        else:
            print("⚠️ GPU deaktiviert oder nicht verfügbar → CPU-Fallback")

        # === Mixed Precision ===
        if use_mixed:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                print("✅ Mixed Precision aktiviert (Tensor Cores aktiv)")
            except Exception as e:
                print(f"⚠️ Mixed Precision nicht verfügbar: {e}")

        # === XLA Compiler ===
        if use_xla:
            try:
                tf.config.optimizer.set_jit(True)
                print("✅ XLA Compiler aktiviert")
            except Exception:
                print("⚠️ XLA nicht verfügbar")

        # === Modellaufbau ===
        input_shape = dataset.X_train.shape[1:]
        print(f"📐 Input Shape: {input_shape}")
        self.model = build_hybrid_cnn_lstm(input_shape, settings=self.settings)

        # === Log-Setup ===
        logs_root = Path(self.data_config.get("paths", {}).get("logs", "logs"))
        logs_root.mkdir(parents=True, exist_ok=True)
        
        callbacks_list = build_callbacks(self.training_config, logs_root)

        # ❌ entfernt Keras-eigene Fortschrittsleiste
        callbacks_list = [cb for cb in callbacks_list if cb.__class__.__name__ != "ProgbarLogger"]
        
        tf_loader = TFTrainingLoader(
            total_epochs=int(self.training_config.get("epochs", 50)),
            total_batches=len(dataset.X_train) // int(self.training_config.get("batch_size", 64)),
            update_interval=0.1
        )
        callbacks_list.append(tf_loader)

        # === Dataset-Optimierung ===
        AUTOTUNE = tf.data.AUTOTUNE
        prefetch_size = AUTOTUNE if prefetch_val == "auto" else int(prefetch_val)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((dataset.X_train, dataset.y_train))
            .shuffle(shuffle_buffer)
            .batch(batch_size)
            .prefetch(prefetch_size)
        )

        val_ds = (
            tf.data.Dataset.from_tensor_slices((dataset.X_val, dataset.y_val))
            .batch(batch_size)
            .prefetch(prefetch_size)
        )

        print(f"\n🚀 Training gestartet (Batch Size {batch_size})\n")

        # === Class Weights ===
        class_weight = self._determine_class_weights(dataset.y_train)

        # === Training ===
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(self.training_config.get("epochs", 50)),
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=0, # ⬅️ wichtig: unterdrückt Keras-eigene Logs 0 custom, 1 Vorgabe
        )

        # === Modell speichern ===
        if self.training_config.get("save_best_only", False):
            models_dir = Path(self.data_config.get("paths", {}).get("models", "models"))
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / "hybrid_cnn_lstm.keras"
            self.model.save(model_path)
            print(f"💾 Modell gespeichert unter {model_path}")

        print("\n✅ Training abgeschlossen!\n")
        return self.model
    
    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        # === System Loader für gesamte Pipeline ===
        loader = SystemLoader("HEUSC – Training Pipeline", total_steps=4)

        # === Schritt 1: Dataset vorbereiten ===
        loader.update(info="Lade und verarbeite Datensätze...")
        dataset = self.prepare_dataset()

        # === Schritt 2: Modell-Training ===
        loader.update(info="Trainiere Modell...")
        self.train(dataset)

        # === Schritt 3: Simulation ===
        loader.update(info="Starte Simulation...")
        results = self.simulate(dataset)

        # === Schritt 4: Abschluss ===
        loader.update(info="Speichere Ergebnisse...")
        loader.done("Pipeline abgeschlossen")

        return results

    # ------------------------------------------------------------------
    # Simulation / Paper trading
    # ------------------------------------------------------------------
    def simulate(self, dataset: DatasetSplit) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Das Modell wurde noch nicht trainiert.")

        simulation_cfg = self.simulation_config
        if not simulation_cfg.get("enabled", True):
            return {"enabled": False}

        confidence_threshold = float(simulation_cfg.get("confidence_threshold", 0.7))
        initial_balance = float(simulation_cfg.get("initial_balance", 1000.0))
        trade_fee = float(simulation_cfg.get("trade_fee", 0.001))
        reward_multiplier = float(simulation_cfg.get("reward_multiplier", 1.0))
        invest_fraction = float(simulation_cfg.get("invest_fraction", 0.1))

        predictions = self.model.predict(dataset.X_test)
        if predictions.ndim > 1:
            predictions = predictions.squeeze(axis=-1)

        balance = initial_balance
        trades = []

        for idx, probability in enumerate(predictions):
            meta = dataset.meta_test[idx]
            if probability < confidence_threshold:
                continue

            amount_to_invest = balance * invest_fraction
            if amount_to_invest <= 0:
                continue

            buy_price = meta["current_price"]
            sell_price = meta["future_price"]

            fee_amount = amount_to_invest * trade_fee * 2
            investment = Investment(buy_price, sell_price, amount_to_invest, gebuehren=fee_amount)
            profit = investment.berechne_gewinn() * reward_multiplier
            balance = profit

            trades.append(
                {
                    "probability": float(probability),
                    "symbol": meta["symbol"],
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "profit": profit,
                    "balance_after_trade": balance,
                }
            )

        if trades and simulation_cfg.get("log_trades", False):
            logs_dir = Path(simulation_cfg.get("path", "simulation/logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)
            trades_df = pd.DataFrame(trades)
            log_file = logs_dir / "paper_trading.csv"
            trades_df.to_csv(log_file, index=False)

        return {
            "enabled": True,
            "trades": len(trades),
            "final_balance": balance,
            "profit": balance - initial_balance,
        }

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        dataset = self.prepare_dataset()
        self.train(dataset)
        return self.simulate(dataset)


__all__ = ["MarketSimulationTrainer", "DatasetSplit"]

if __name__ == "__main__":
    
    settings_path = str("config/settings.json")
    trainer = MarketSimulationTrainer(Path(settings_path))
    results = trainer.run()
    print(json.dumps(results, indent=2, ensure_ascii=False))