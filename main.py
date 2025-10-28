"""Command line entry point for training and simulation."""
from __future__ import annotations

import json
import os

import tensorflow as tf

from pathlib import Path

from src.trainer import MarketSimulationTrainer
from src.gpu_check import gpu_info
from src.loader import SystemLoader

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

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