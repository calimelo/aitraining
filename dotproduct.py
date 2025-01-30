import numpy as np

# Define two vectors
A = np.array([2, 3, 4])
B = np.array([1, 0, 2])

# Compute dot product
dot_product = np.dot(A, B)

print(f"Dot Product: {dot_product}")

#plotting the vectors
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label="A")
ax.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', label="B")

# Set plot limits
ax.set_xlim(0, 3)

ax.set_ylim(0, 4)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

ax.set_title(f"Dot Product = {dot_product}")

# Add grid and legend
ax.grid()
ax.legend()

# Save the plot
plt.savefig('dot_product.png')