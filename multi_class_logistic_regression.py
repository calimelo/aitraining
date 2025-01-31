import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Training data (Weight, Size, Color Intensity) -> Fruit Type
X = np.array([
    [150, 7, 0.9], 
    [120, 8, 0.7], 
    [130, 6, 0.8], 
    [180, 9, 1.0], 
    [110, 7, 0.6], 
    [140, 6.5, 0.85]
])  # Features

y = np.array([0, 1, 2, 0, 1, 2])  # Labels (0 = Apple, 1 = Banana, 2 = Orange)

# Standardize the data for better accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Multi-Class Logistic Regression Model
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
model.fit(X_scaled, y)

# Predict fruit for (Weight = 135g, Size = 7.5cm, Color Intensity = 0.85)
new_fruit = np.array([[135, 7.5, 0.85]])
new_fruit_scaled = scaler.transform(new_fruit)
prediction = model.predict(new_fruit_scaled)
probabilities = model.predict_proba(new_fruit_scaled)

# Print results
fruit_classes = ["Apple 🍎", "Banana 🍌", "Orange 🍊"]
predicted_fruit = fruit_classes[prediction[0]]

print(f"Predicted Fruit: {predicted_fruit}")
print(f"Class Probabilities: {probabilities[0]}")

# Plot1
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=100, edgecolors='k', label="Training Data")
# ax.scatter(new_fruit[0][0], new_fruit[0][1], new_fruit[0][2], color='red', s=200, label="Prediction")
# ax.set_xlabel("Weight (g)")
# ax.set_ylabel("Size (cm)")
# ax.set_zlabel("Color Intensity")
# plt.legend()
# plt.title("Multi-Class Logistic Regression")
# plt.savefig("multi_class_logistic_regression.png")

# Plot2
#change font
plt.rcParams.update({'font.size': 14})
#change font family from dejavu to something that supports emojis
plt.rcParams.update({'font.family': 'Arial'})
plt.bar(fruit_classes, probabilities[0], color=['red', 'yellow', 'orange'])
plt.ylabel("Probability")
plt.title("Multi-Class Logistic Regression Prediction")
plt.savefig("multi_class_logistic_regression_probabilities.png")