import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

# Sample text data
doc1 = "I love deep learning and AI"
doc2 = "Deep learning is amazing for AI"

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Compute Euclidean Distance
distance = euclidean_distances(tfidf_matrix[0], tfidf_matrix[1])

print(f"Euclidean Distance between sentences: {distance[0][0]:.4f}")

# Extract vectors for plotting
doc1_vector = tfidf_matrix.toarray()[0]
doc2_vector = tfidf_matrix.toarray()[1]

# Plot the vectors
fig, ax = plt.subplots()
ax.quiver(0, 0, doc1_vector[0], doc1_vector[1], angles='xy', scale_units='xy', scale=1, color='r', label="Doc1")
ax.quiver(0, 0, doc2_vector[0], doc2_vector[1], angles='xy', scale_units='xy', scale=1, color='b', label="Doc2")

# Set plot limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("TF-IDF Dimension 1")
ax.set_ylabel("TF-IDF Dimension 2")
ax.set_title(f"Euclidean Distance = {distance[0][0]:.4f}")

# Add grid and legend
ax.grid()
ax.legend()

# save the plot
plt.savefig('euclidean_distance.png')
