from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample knowledge base (documents)
docs = [
    "GPT-4 has multimodal capabilities.",
    "AI models are improving real-time translation.",
    "Self-driving cars use deep learning for object detection.",
]

# Convert documents to vector embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)

# Create a FAISS index for retrieval
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# User query
query = "What are the latest AI advancements?"
query_embedding = model.encode([query])

# Retrieve the most relevant document
D, I = index.search(np.array(query_embedding), k=1)
retrieved_doc = docs[I[0][0]]

# Generate a response using ChatGPT (simulated here)
response = f"Based on my search: {retrieved_doc}"

print(response)
