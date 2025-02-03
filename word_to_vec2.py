from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize "king"
inputs = tokenizer("king", return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract vector (mean pooling over layers)
king_vector = outputs.last_hidden_state.mean(dim=1)

print("Vector for 'king':", king_vector[0][:10])
