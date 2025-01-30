from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample sentences
sentence_1 = "AI is transforming the world."
sentence_2 = "Artificial intelligence is changing everything."

# Generate embeddings
embedding_1 = model.encode(sentence_1)
embedding_2 = model.encode(sentence_2)

# Print first 5 values of the embeddings
print("Embedding 1:", embedding_1[:5])
print("Embedding 2:", embedding_2[:5])

# Compute similarity (Cosine Similarity)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embedding_1], [embedding_2])

print(f"Similarity Score: {similarity[0][0]:.2f}")
