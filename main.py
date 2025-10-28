"""Command line entry point for training and simulation."""
from __future__ import annotations

import json

from pathlib import Path

from src.trainer import MarketSimulationTrainer
from src.gpu_check import gpu_info


def main(settings_path: str = "config/settings.json") -> None:
    trainer = MarketSimulationTrainer(Path(settings_path))
    results = trainer.run()
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    gpu_info()
    main()