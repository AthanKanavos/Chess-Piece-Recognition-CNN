from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASS_NAMES, load_datasets
from src.utils import ensure_directory, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained chess piece classifier."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = ensure_directory(args.output_dir or model_path.parent)

    _, val_ds = load_datasets(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        use_augmentation=False,
    )

    model = tf.keras.models.load_model(model_path)
    evaluation = model.evaluate(val_ds, return_dict=True, verbose=1)

    probabilities = model.predict(val_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in val_ds])

    indices = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(labels, predictions, labels=indices)
    report = classification_report(
        labels,
        predictions,
        labels=indices,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    save_json(
        {
            **{key: float(value) for key, value in evaluation.items()},
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        output_dir / "evaluation.json",
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(indices, CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(indices, CLASS_NAMES)

    for row in indices:
        for col in indices:
            plt.text(col, row, str(cm[row, col]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nEvaluation results:")
    for key, value in evaluation.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
