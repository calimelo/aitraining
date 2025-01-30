import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Step 1: Sample knowledge base (documents)
docs = [
    "GPT-4 has multimodal capabilities, meaning it can process text and images.",
    "AI models are improving real-time translation, making it easier to communicate across languages.",
    "Self-driving cars use deep learning for object detection and navigation.",
    "Quantum computing has the potential to revolutionize AI by solving complex problems much faster.",
    "Large language models like ChatGPT rely on transformer architectures to generate responses."
]

# Step 2: Convert documents into embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)

# Step 3: Create a FAISS index for fast retrieval
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Step 4: User query
query = "What are the latest advancements in AI?"
query_embedding = model.encode([query])

# Step 5: Retrieve the most relevant document
D, I = index.search(np.array(query_embedding), k=1)
retrieved_doc = docs[I[0][0]]

# Step 6: Generate a response using a language model (simulated here)
gpt_pipeline = pipeline("text-generation", model="gpt2")
prompt = f"Based on the following information, answer the query:\n\n{retrieved_doc}\n\nQuery: {query}\nAnswer: "
response = gpt_pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Step 7: Print results
print("Retrieved Document:", retrieved_doc)
print("Generated Response:", response)