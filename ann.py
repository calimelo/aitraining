import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load dataset (MNIST Handwritten Digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define an ANN model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input Layer
    keras.layers.Dense(128, activation='relu'),  # Hidden Layer
    keras.layers.Dense(10, activation='softmax') # Output Layer
])

# Compile & Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

# Save the model
model.save("mnist_ann.h5")