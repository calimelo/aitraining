from transformers import AutoTokenizer, AutoModel
import torch

# Load Pretrained BERT Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Sentence to be converted
sentence = "King Arthur ruled the kingdom."

# Tokenize and Convert to Model Input
inputs = tokenizer(sentence, return_tensors="pt")

# Generate Sentence Embedding
with torch.no_grad():
    outputs = model(**inputs)

# Extract Sentence Vector (Mean of Token Embeddings)
sentence_vector = outputs.last_hidden_state.mean(dim=1)

print("Sentence Embedding:", sentence_vector)
