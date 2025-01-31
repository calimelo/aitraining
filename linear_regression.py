import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training data (Temperature in °F and Money Made in $)
X = np.array([70, 75, 80, 85, 90]).reshape(-1, 1)  # Temperature
y = np.array([30, 35, 40, 45, 50])  # Money made

# Create the model
model = LinearRegression()
model.fit(X, y)

# Predict earnings for 80°F
predicted_earnings = model.predict([[82]])

print(f"Predicted earnings for 80°F: ${predicted_earnings[0]:.2f}")

# Plot the data and regression line
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.scatter(80, predicted_earnings, color='green', label="Prediction (80°F)")
plt.xlabel("Temperature (°F)")
plt.ylabel("Money Made ($)")
plt.legend()
plt.savefig("linear_regression.png")
