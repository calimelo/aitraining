import numpy as np

# Create a 2D tensor (matrix)
tensor_numpy = np.array([[1, 2, 3], [4, 5, 6]])
print(tensor_numpy.shape)  # Output: (2, 3)

import torch

# Create a PyTorch tensor
tensor_pytorch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(tensor_pytorch.shape)  # Output: torch.Size([2, 2])

import tensorflow as tf

# Create a TensorFlow tensor
tensor_tf = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print(tensor_tf.shape)  # Output: (2, 2)
