from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Training data (Weather, Temperature) -> Play Outside (Yes = 1, No = 0)
X = np.array([[1, 75], [0, 50], [1, 85], [0, 40], [1, 60]])  # Features (Good Weather = 1, Bad Weather = 0)
y = np.array([1, 0, 1, 0, 1])  # Labels (1 = Play, 0 = Stay Inside)

# Train the Random Forest Model (with 10 trees)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Predict if a child will play outside on a 70°F day with good weather
new_data = np.array([[1, 70]])
prediction = model.predict(new_data)

print(f"Will the child play outside? {'Yes' if prediction[0] == 1 else 'No'}")

# Plot Decision Boundary
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[yy.ravel(), xx.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green')))
plt.scatter(X[:, 1], X[:, 0], c=y, cmap=ListedColormap(('red', 'green')), edgecolors='k')
plt.scatter(70, 1, color='blue', marker='X', s=200, label="New Data (70°F, Good Weather)")
plt.xlabel("Temperature (°F)")
plt.ylabel("Weather (0: Bad, 1: Good)")
plt.legend()
plt.title("Random Forest Classifier Decision Boundary")
plt.savefig("random_forest.png")