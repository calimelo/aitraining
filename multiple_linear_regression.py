import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training data (Temperature, Ads, Holiday) -> Earnings
X = np.array([
    [70, 200, 0], 
    [75, 250, 0], 
    [80, 300, 1], 
    [85, 350, 1], 
    [90, 400, 0]
])  # Inputs (Features)

y = np.array([30, 35, 50, 55, 50])  # Output (Earnings in $)

# Create the model
model = LinearRegression()
model.fit(X, y)

# Predict earnings for (Temperature = 82°F, Ads = $280, Holiday = 1)
new_data = np.array([[82, 280, 1]])
predicted_earnings = model.predict(new_data)

print(f"Predicted earnings for 82°F, $280 ads, and Holiday: ${predicted_earnings[0]:.2f}")

# Plot the data and regression line
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label="Actual Data")
ax.scatter(new_data[0][0], new_data[0][1], predicted_earnings, color='green', label="Prediction")
ax.set_xlabel("Temperature (°F)")
ax.set_ylabel("Ads ($)")
ax.set_zlabel("Earnings ($)")
plt.legend()
plt.savefig("multiple_linear_regression.png")