from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["bishop", "king", "knight", "pawn", "queen", "rook"]


def _resolve_class_directories(data_dir: Path) -> Dict[str, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    available = {
        path.name.lower(): path
        for path in data_dir.iterdir()
        if path.is_dir()
    }

    missing = [name for name in CLASS_NAMES if name not in available]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {data_dir}: {missing}"
        )

    return {name: available[name] for name in CLASS_NAMES}


def _create_normalized_view(data_dir: Path) -> Path:
    """
    TensorFlow expects exact folder names when class_names is provided.
    The original dataset may use capitalized names, so this function returns
    the source directory and class ordering is inferred after validation.
    """
    _resolve_class_directories(data_dir)
    return data_dir


def _augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )


def _prepare(
    dataset: tf.data.Dataset,
    training: bool,
    use_augmentation: bool,
) -> tf.data.Dataset:
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

    if training and use_augmentation:
        augment = _augmentation()

        def preprocess(images, labels):
            images = tf.cast(images, tf.float32)
            images = augment(images, training=True)
            return rescale(images), labels
    else:
        def preprocess(images, labels):
            images = tf.cast(images, tf.float32)
            return rescale(images), labels

    return dataset.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def load_datasets(
    data_dir: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 128,
    validation_split: float = 0.20,
    seed: int = 42,
    use_augmentation: bool = True,
):
    data_dir = _create_normalized_view(Path(data_dir))

    discovered_names = sorted(
        path.name for path in data_dir.iterdir() if path.is_dir()
    )

    lower_to_actual = {name.lower(): name for name in discovered_names}
    ordered_actual_names = [lower_to_actual[name] for name in CLASS_NAMES]

    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=ordered_actual_names,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        subset="training",
        shuffle=True,
        **common,
    )

    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        subset="validation",
        shuffle=False,
        **common,
    )

    train_ds = _prepare(train_raw, training=True, use_augmentation=use_augmentation)
    val_ds = _prepare(val_raw, training=False, use_augmentation=False)

    return train_ds, val_ds


def compute_class_weights(data_dir: str | Path) -> Dict[int, float]:
    data_dir = Path(data_dir)
    class_dirs = _resolve_class_directories(data_dir)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        count = sum(
            1
            for path in class_dirs[class_name].rglob("*")
            if path.is_file() and path.suffix.lower() in valid_extensions
        )
        labels.extend([class_index] * count)

    if not labels:
        raise ValueError(f"No images found under {data_dir}")

    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_array,
    )

    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}
