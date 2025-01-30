import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create a dataset for apples, oranges, and bananas
data = {
    'Weight': [150, 160, 170, 180, 140, 130, 120, 190, 200, 210, 110, 115, 105],
    'Texture': [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2],  # 1 = Smooth, 0 = Bumpy, 2 = Soft
    'Fruit': ['Apple', 'Apple', 'Apple', 'Apple', 'Orange', 'Orange', 'Orange', 'Apple', 'Apple', 'Apple', 'Banana', 'Banana', 'Banana']
}

# Convert to DataFrame
df = pd.DataFrame(data)
X = df[['Weight', 'Texture']]
y = df['Fruit']

# Encode labels (Apple=0, Orange=1, Banana=2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple deep learning model
model = keras.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs = 8
# Train the model
model.fit(X_train, y_train, epochs=epochs, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Convert predictions back to fruit names
predicted_fruits = label_encoder.inverse_transform(predicted_classes)
actual_fruits = label_encoder.inverse_transform(y_test)

# Show some predictions
results_df = pd.DataFrame({'Actual': actual_fruits, 'Predicted': predicted_fruits})
print(results_df.head())

#show the results in a plot
plt.scatter(df['Weight'], df['Texture'], c=y_encoded)
plt.xlabel('Weight')
plt.ylabel('Texture')
plt.title('Apples, Oranges, and Bananas')
plt.savefig('deepl_apples_oranges_bananas' + str(epochs) + '.png') 

#save the model
model.save('deepl_apples_oranges_bananas.keras')

import joblib
#save the label encoder
encodername = 'deepl_apples_oranges_bananas_labelencoder.joblib'
joblib.dump(label_encoder, encodername)

#save weights
model.save_weights('deepl_apples_oranges_bananas.weights.h5')