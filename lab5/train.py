import argparse
import json
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from nnv import NNV

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f"Train images dimensions: {train_images.shape}")
print(f"Test images dimensions: {test_images.shape}")

plt.figure(figsize=(10, 5))
plt.imshow(train_images[0])
plt.colormaps()
plt.show()

# Normalization layer to standardize input images; we adapt once on the training set
normalization_layer = keras.layers.Normalization(input_shape=(28, 28), axis=None)
normalization_layer.adapt(train_images)

# Expose normalization statistics for use in other modules (e.g. Keras Tuner builders)
NORM_MEAN = normalization_layer.mean.numpy()
NORM_VAR = normalization_layer.variance.numpy()


def create_fc_model() -> keras.Model:
    """Create a simple fully connected model."""
    model = keras.Sequential([
        normalization_layer,
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model


def create_cnn_model() -> keras.Model:
    """Create a simple convolutional model."""
    model = keras.Sequential([
        normalization_layer,
        keras.layers.Reshape((28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fashion MNIST model (fc or cnn).")
    parser.add_argument(
        "--arch",
        choices=["fc", "cnn"],
        default="fc",
        help="Model architecture to train: 'fc' (fully connected) or 'cnn' (convolutional).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable simple data augmentation on the training images.",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    models_dir = "models"
    metrics_dir = "metrics"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Choose base name for files (optionally suffixed with "_augment")
    if args.arch == "fc":
        model = create_fc_model()
        base_name = "model_fc"
    else:
        model = create_cnn_model()
        base_name = "model_cnn"

    if args.augment:
        base_name = f"{base_name}_augment"

    model_filename = os.path.join(models_dir, f"{base_name}.keras")

    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    plt.rcParams["figure.figsize"] = (200, 50)

    layersList = [
        {"title": "Input\n(784 flatten)", "units": 784, "color": "Blue"},
        {"title": "Hidden 1\n(relu: 128)", "units": 128},
        {"title": "Output\n(softmax: 10)", "units": 10, "color": "Green"},
    ]
    NNV(layersList, spacing_layer=10, max_num_nodes_visible=20, node_radius=1, font_size=24).render()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Optionally apply simple data augmentation to training images using tf.image,
    # inspired by TensorFlow's data augmentation tutorial:
    # https://www.tensorflow.org/tutorials/images/data_augmentation
    if args.augment:
        images_tf = tf.convert_to_tensor(train_images, dtype=tf.float32)
        images_tf = tf.expand_dims(images_tf, axis=-1)  # (N, 28, 28, 1)
        images_tf = tf.image.random_flip_left_right(images_tf)
        images_tf = tf.image.random_brightness(images_tf, max_delta=0.2)
        images_tf = tf.squeeze(images_tf, axis=-1)  # back to (N, 28, 28)
        train_images_aug = images_tf.numpy()
    else:
        train_images_aug = train_images

    history = model.fit(train_images_aug, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100}%")

    # Predictions and confusion (error) matrix
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        test_labels, y_pred, num_classes=10
    ).numpy()

    # Plot and save confusion matrix as PNG (in metrics folder)
    cm_png = os.path.join(metrics_dir, f"{base_name}_confusion.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_png}")

    # Plot and save training loss/accuracy as PNG (in metrics folder)
    history_png = os.path.join(metrics_dir, f"{base_name}_history.png")
    plt.figure(figsize=(6, 4))
    plt.plot(history.history.get("loss", []), label="loss")
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_png)
    plt.close()
    print(f"Saved training history plot to {history_png}")

    # Save metrics (including loss and confusion matrix) in metrics folder
    metrics = {
        "architecture": args.arch,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": confusion_matrix.tolist(),
    }
    metrics_filename = os.path.join(metrics_dir, f"{base_name}_metrics.json")
    with open(metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_filename}")

    # Save the trained model depending on the chosen architecture
    model.save(model_filename)
    print(f"Saved model to {model_filename}")

    plt.figure(figsize=(10, 5))
    plt.imshow(test_images[1])
    plt.colormaps()
    plt.show()


if __name__ == "__main__":
    main()