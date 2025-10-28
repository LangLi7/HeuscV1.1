"""Command line entry point for training and simulation."""
from __future__ import annotations

import os
import json
from pathlib import Path

# === GPU / TensorFlow Setup ===

# 🧹 TensorFlow / XLA / oneDNN Output minimieren
# Diese Variable muss *vor* jedem TensorFlow-Import gesetzt werden
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # 0 = alle Logs, 1 = Warnungen, 2 = Fehler, 3 = nur kritische
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # deaktiviert oneDNN float16 Fallback-Warnungen
#os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

print("\n🔍 Checking TensorFlow GPU configuration...\n")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Haupt-GPU (RTX 3060) aktivieren
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU aktiviert: {gpus[0].name}")
    else:
        print("⚠️ Keine GPU gefunden → CPU-Fallback aktiv")
except RuntimeError as e:
    print(f"⚠️ GPU-Initialisierung übersprungen: {e}")

# --- erst jetzt Module importieren, die TensorFlow nutzen ---
from src.trainer import MarketSimulationTrainer
from src.gpu_check import gpu_info
from src.loader import SystemLoader


def main(settings_path: str = "config/settings.json") -> None:
    trainer = MarketSimulationTrainer(Path(settings_path))
    results = trainer.run()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app_loader = SystemLoader("HEUSC Engine Initialisierung", total_steps=2)
    app_loader.update(info="Prüfe GPU und Umgebung...")
    gpu_info()
    app_loader.update(info="Starte Training...")
    main()
    app_loader.done("HEUSC Engine bereit.")
