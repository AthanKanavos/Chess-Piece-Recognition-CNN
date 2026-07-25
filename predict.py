from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the class of one chess piece image."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)
    image = tf.keras.utils.load_img(
        args.image_path,
        target_size=(args.image_size, args.image_size),
    )
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0) / 255.0

    probabilities = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index]:.4f}")

    print("\nClass probabilities:")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
