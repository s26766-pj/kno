import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_and_preprocess_image(path: str) -> np.ndarray:
    """
    Load an image from disk and preprocess it for the Fashion MNIST models.

    Steps:
    - Resize the image to 28x28 pixels.
    - Convert it to grayscale (single channel).
    - Apply a negative transform by inverting pixel values.
    - Return an array of shape (1, 28, 28) suitable for model.predict().
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Load image as grayscale and resize to 28x28
    img = keras.utils.load_img(path, color_mode="grayscale", target_size=(28, 28))
    img_array = keras.utils.img_to_array(img)  # shape (28, 28, 1), float32 in [0, 255]

    # Apply negative (invert pixel values)
    img_array = 255.0 - img_array

    # Remove channel dimension to match training (28, 28)
    img_array = np.squeeze(img_array, axis=-1)  # (28, 28)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28)

    return img_array


def classify_image(model_path: str, image_path: str) -> None:
    """
    Load a trained model from disk and classify a single input image.

    The function loads the Keras model (handling legacy softmax names),
    runs inference on the preprocessed image, and prints the predicted
    class index, human-readable class name, and prediction confidence.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Map legacy 'softmax_v2' activation name (from older TF/Keras) to the
    # current softmax implementation so older saved models still load.
    custom_objects = {"softmax_v2": tf.nn.softmax}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    img_batch = load_and_preprocess_image(image_path)

    # Run prediction
    preds = model.predict(img_batch)
    probs = tf.nn.softmax(preds[0]).numpy()

    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Predicted class: {class_idx} ({class_name})")
    print(f"Confidence: {confidence:.4f}")


def main() -> None:
    """
    Entry point for the prediction script.

    Reads the image path and optional model path from the command line,
    then performs classification and prints the result to stdout.
    """
    parser = argparse.ArgumentParser(description="Classify an image using a trained Fashion MNIST model.")
    parser.add_argument("image", help="Path to the input image file.")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "model_fc.keras"),
        help="Path to the trained Keras model file (default: models/model_fc.keras).",
    )
    args = parser.parse_args()

    classify_image(args.model, args.image)


if __name__ == "__main__":
    main()

