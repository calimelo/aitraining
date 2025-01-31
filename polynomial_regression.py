import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Training data (Temperature in °F -> Ice Cream Sales in $)
X = np.array([50, 60, 70, 80, 90]).reshape(-1, 1)
y = np.array([100, 150, 300, 600, 1000])

# Transform input features into polynomial features (degree = 2 for quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the Polynomial Regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict sales for 82°F
X_test = np.array([[82]])
X_test_poly = poly.transform(X_test)
predicted_sales = model.predict(X_test_poly)

print(f"Predicted ice cream sales at 82°F: ${predicted_sales[0]:.2f}")

# Plot the results
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X_poly), color='red', label="Polynomial Regression Curve")
plt.scatter(85, predicted_sales, color='green', label="Prediction (82°F)")
plt.xlabel("Temperature (°F)")
plt.ylabel("Ice Cream Sales ($)")
plt.legend()
plt.savefig("polynomial_regression.png")
