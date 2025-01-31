import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Training Data (Weight, Size) -> 0 = Apple 🍏, 1 = Banana 🍌
X = np.array([
    [150, 7], [160, 7.5], [170, 8],  # Apples 🍏
    [180, 9], [190, 9.5], [200, 10]  # Bananas 🍌
])  
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Apple, 1 = Banana

# Train the SVM Model
model = SVC(kernel='linear')
model.fit(X, y)

# Predict if a fruit with (Weight = 175g, Size = 8.5cm) is an Apple or Banana
new_fruit = np.array([[175, 8.5]])
prediction = model.predict(new_fruit)

print(f"Predicted Fruit: {'Banana 🍌' if prediction[0] == 1 else 'Apple 🍏'}")

# Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.scatter(175, 8.5, color='green', marker='X', s=200, label="New Fruit")
plt.xlabel("Weight (g)")
plt.ylabel("Size (cm)")
plt.legend()
plt.title("SVM Decision Boundary")
plt.savefig("support_vector_machine.png")
