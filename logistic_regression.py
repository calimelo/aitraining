import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Training Data (Capital Letters, Links) -> Spam (1) or Not Spam (0)
X = np.array([[5, 1], [20, 5], [15, 3], [2, 0], [30, 7]])  # Features
y = np.array([0, 1, 1, 0, 1])  # Labels (0 = Not Spam, 1 = Spam)

# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Predict if an email with (10 capital letters, 2 links) is Spam
new_email = np.array([[10, 2]])
prediction = model.predict(new_email)
probability = model.predict_proba(new_email)[0][1]  # Probability of being spam

print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Not Spam'}")
print(f"Spam Probability: {probability:.2f}")

# Plot Decision Boundary
x_range = np.linspace(0, 35, 100)
y_range = np.linspace(0, 10, 100)
X1, X2 = np.meshgrid(x_range, y_range)
X_grid = np.c_[X1.ravel(), X2.ravel()]
y_pred_grid = model.predict(X_grid).reshape(X1.shape)

plt.contourf(X1, X2, y_pred_grid, alpha=0.3, cmap="coolwarm")
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors='k', label="Training Data")
plt.scatter(10, 2, color='green', marker='X', s=200, label="New Email")
plt.xlabel("Capital Letters")
plt.ylabel("Links in Email")
plt.legend()
plt.title("Logistic Regression Decision Boundary")
plt.savefig("logistic_regression.png")