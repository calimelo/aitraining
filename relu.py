import numpy as np
import matplotlib.pyplot as plt

# Define ReLU and Leaky ReLU functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate input values
x = np.linspace(-5, 5, 100)

# Compute outputs
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# Plot the activation functions
plt.figure(figsize=(8, 5))
plt.plot(x, y_relu, label="ReLU", linewidth=2)
plt.plot(x, y_leaky_relu, label="Leaky ReLU", linestyle="dashed", linewidth=2)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.legend()
plt.title("ReLU vs. Leaky ReLU")
plt.savefig("relu.png")
