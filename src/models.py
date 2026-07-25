from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Model, layers


NUM_CLASSES = 6


def _conv(
    x: tf.Tensor,
    filters: int,
    name: str,
    batch_norm: bool = False,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation=None if batch_norm else "relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)

    if batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)

    return x


def _head(
    x: tf.Tensor,
    dense_units: int = 256,
    dropout_rate: float = 0.50,
) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name="prediction")(x)


def build_architecture_1(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64, 128),
    block_dropout: float = 0.25,
) -> Model:
    """(Conv2D x2 - MaxPooling2D - Dropout) x3."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters, start=1):
        x = _conv(x, block_filters, f"block_{block}_conv_1")
        x = _conv(x, block_filters, f"block_{block}_conv_2")
        x = layers.MaxPooling2D(pool_size=2, name=f"block_{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block_{block}_dropout")(x)

    return Model(inputs, _head(x), name="chess_architecture_1")


def build_architecture_2(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64, 128),
    block_dropout: float = 0.25,
) -> Model:
    """((Conv2D - BatchNorm) - Conv2D - MaxPooling2D - Dropout) x3."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters, start=1):
        x = _conv(
            x,
            block_filters,
            f"block_{block}_conv_1",
            batch_norm=True,
        )
        x = _conv(
            x,
            block_filters,
            f"block_{block}_conv_2",
            batch_norm=False,
        )
        x = layers.MaxPooling2D(pool_size=2, name=f"block_{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block_{block}_dropout")(x)

    return Model(inputs, _head(x), name="chess_architecture_2")


def build_architecture_3(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64, 128),
    block_dropout: float = 0.25,
) -> Model:
    """(Conv2D x3 - MaxPooling2D - Dropout) x3."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters, start=1):
        for conv_index in range(1, 4):
            x = _conv(
                x,
                block_filters,
                f"block_{block}_conv_{conv_index}",
            )
        x = layers.MaxPooling2D(pool_size=2, name=f"block_{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block_{block}_dropout")(x)

    return Model(inputs, _head(x), name="chess_architecture_3")


def build_model(
    architecture: int,
    input_shape=(224, 224, 3),
    learning_rate: float = 1e-3,
) -> Model:
    builders = {
        1: build_architecture_1,
        2: build_architecture_2,
        3: build_architecture_3,
    }

    if architecture not in builders:
        raise ValueError("architecture must be one of: 1, 2, 3")

    model = builders[architecture](input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
