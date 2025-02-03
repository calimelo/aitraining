from sentence_transformers import SentenceTransformer

# Load SBERT Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert a Sentence to Vector
sentence = "King Arthur ruled the kingdom."
sentence_vector = model.encode(sentence)

print("Sentence Vector:", sentence_vector)
# print length
print(len(sentence_vector))

#another example
sentence = "The king ruled the kingdom."
sentence_vector = model.encode(sentence)
# print length
print(len(sentence_vector))