import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from nnv import NNV

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f"Train images dimensions: {train_images.shape}")
print(f"Test images dimensions: {test_images.shape}")

plt.figure(figsize=(10,5))
plt.imshow(train_images[0])
plt.colormaps()
plt.show()

train_images = train_images /255.
test_images = test_images / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


plt.rcParams["figure.figsize"] = 200,50

layersList = [
    {"title":"Input\n(784 flatten)", "units": 784, "color": "Blue"},
    {"title":"Hidden 1\n(relu: 128)", "units": 128},
    {"title":"Output\n(softmax: 10)", "units": 10,"color": "Green"},
]
NNV(layersList, spacing_layer=10, max_num_nodes_visible=20, node_radius=1, font_size=24).render()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Model Accuracy: {test_acc * 100}%")

predictions = model.predict(test_images)
predictions[1]

np.argmax(predictions[1])

test_labels[1]

plt.figure(figsize=(10,5))
plt.imshow(test_images[1])
plt.colormaps()
plt.show()