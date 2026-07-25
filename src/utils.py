from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_history(history: tf.keras.callbacks.History, output_dir: str | Path) -> None:
    output_dir = ensure_directory(output_dir)
    frame = pd.DataFrame(history.history)
    frame.to_csv(output_dir / "history.csv", index=False)

    for metric, val_metric in [("loss", "val_loss"), ("accuracy", "val_accuracy")]:
        if metric not in history.history or val_metric not in history.history:
            continue

        plt.figure(figsize=(7, 5))
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[val_metric], label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=160)
        plt.close()
