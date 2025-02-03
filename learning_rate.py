import tensorflow as tf
import time
# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

lr = 0.001  # Choose an optimal LR
lr2 = 0.0001  # Choose an optimal LR
lr3 = 0.00001  # Choose an optimal LR


starttime = time.time()
# Compile model with different learning rates
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Training with learning rate: {lr}")
endtime = time.time()
print(f"Time taken: {endtime - starttime:.4f} seconds")

#this time we will use a learning rate of 0.0001
starttime = time.time()
optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr2)

model2.compile(optimizer=optimizer2, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Training with learning rate: {lr2}")
endtime = time.time()
print(f"Time taken: {endtime - starttime:.4f} seconds")

#this time we will use a learning rate of 0.00001
starttime = time.time()
optimizer3 = tf.keras.optimizers.Adam(learning_rate=lr3)

starttime = time.time()
model3.compile(optimizer=optimizer3, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Training with learning rate: {lr3}")
endtime = time.time()
print(f"Time taken: {endtime - starttime:.4f} seconds")
