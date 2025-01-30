import faiss
import numpy as np

# Define vector dimensions (e.g., 512D)
vector_dim = 512  

# Create random vectors (simulating embeddings)
num_vectors = 10000  # Store 10,000 vectors
vectors = np.random.rand(num_vectors, vector_dim).astype('float32')

# Initialize FAISS index (L2 search)
index = faiss.IndexFlatL2(vector_dim)
index.add(vectors)  # Store vectors in the database

# Query: Create a new random vector & find nearest neighbors
query_vector = np.random.rand(1, vector_dim).astype('float32')
distances, indices = index.search(query_vector, k=3)  # Find 3 closest matches

# Print search results
print("Query Vector Index:", indices[0])
print("Distances:", distances[0])
