import gensim.downloader as api

# Load pretrained Word2Vec model
model = api.load("word2vec-google-news-300")

# Get vector for "king"
king_vector = model["king"]

# Print first 10 dimensions
print("Vector for 'king':", king_vector[:10])
