import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_md")

# Convert words into vectors
word1 = nlp("king").vector
word2 = nlp("queen").vector

# Compare similarity
similarity = nlp("king").similarity(nlp("queen"))

print(f"Vector for 'king': {word1[:5]}...")  # Shows first 5 numbers in vector
print(f"Vector for 'queen': {word2[:5]}...")
print(f"Similarity between 'king' and 'queen': {similarity:.2f}")  # Closer to 1 means more similar

#plot the vectors
import matplotlib.pyplot as plt
plt.scatter(word1, word2)
plt.xlabel('King')
plt.ylabel('Queen')
plt.title('Word Vector Similarity')
#save the plot
plt.savefig('word_vector_similarity.png')