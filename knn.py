import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training Data (Weight, Size) -> Apple 🍏 = 0, Banana 🍌 = 1, Orange 🍊 = 2
X = np.array([
    [150, 7], [160, 7.5], [170, 8],  # Apples 🍏
    [180, 9], [190, 9.5], [200, 10],  # Bananas 🍌
    [140, 6.5], [145, 6.8], [135, 6.2]  # Oranges 🍊
])  
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # Labels (0 = Apple, 1 = Banana, 2 = Orange)

# Train the KNN Model (K=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Predict if a fruit with (Weight = 155g, Size = 7.3cm) is an Apple, Banana, or Orange
new_fruit = np.array([[155, 7.3]])
prediction = model.predict(new_fruit)

fruit_classes = ["Apple 🍏", "Banana 🍌", "Orange 🍊"]
predicted_fruit = fruit_classes[prediction[0]]

print(f"Predicted Fruit: {predicted_fruit}")

# Plot the decision boundary, add fruit reference colors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', label="Training Data", colorizer='cmap')
plt.scatter(155, 7.3, color='green', marker='X', s=200, label="New Fruit")
plt.xlabel("Weight (g)")
plt.ylabel("Size (cm)")

plt.legend()
plt.title("KNN Decision Boundary")
plt.savefig("knn.png")