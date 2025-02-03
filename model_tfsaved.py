import tensorflow as tf

# Create & train a model
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu')])
model.compile(optimizer='adam', loss='mse')

# Save in TensorFlow format
model.save("my_model.keras")

# Load model
loaded_model = tf.keras.models.load_model("my_model.keras")


