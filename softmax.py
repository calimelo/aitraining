import numpy as np

# Logits (Raw scores from a neural network)
logits = np.array([2.0, 1.0, 0.1])

# Apply Softmax
softmax_probs = np.exp(logits) / np.sum(np.exp(logits)) # Softmax function
#total is 3.1
# 2.0/3.1 = 0.6
# 1.0/3.1 = 0.32258065
# 0.1/3.1 = 0.03225806

# Print probabilities

print("Sum of probabilities:", np.sum(softmax_probs))
#print one by one
print("Softmax Probabilities:", softmax_probs[0])
print("Softmax Probabilities:", softmax_probs[1])
print("Softmax Probabilities:", softmax_probs[2])

print("Softmax Probabilities:", softmax_probs)



# The softmax function is different from simple division because it involves exponentiation and normalization. Here's a step-by-step explanation:

# Exponentiation: Each element in the input array (logits) is exponentiated. This means exp(2.0), exp(1.0), and exp(0.1) are calculated.
# Sum of Exponentiated Values: The sum of these exponentiated values is computed.
# Normalization: Each exponentiated value is divided by the sum of the exponentiated values to get the softmax probabilities.
# Let's break down the calculations:

# Calculate the exponentials:

# exp(2.0) ≈ 7.389
# exp(1.0) ≈ 2.718
# exp(0.1) ≈ 1.105
# Sum of exponentials:

# 7.389 + 2.718 + 1.105 ≈ 11.212
# Normalize each exponentiated value by dividing by the sum:

# 7.389 / 11.212 ≈ 0.659
# 2.718 / 11.212 ≈ 0.242
# 1.105 / 11.212 ≈ 0.099

# Print simple division for comparison
total = np.sum(logits)
print("Simple division results:")
# print(2/3.1) 
# print(1/3.1)
# print(0.1/3.1)
print(logits[0] / total)
print(logits[1] / total)
print(logits[2] / total)