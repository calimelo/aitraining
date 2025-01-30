from transformers import AutoTokenizer

# Load a pre-trained tokenizer (GPT-style model)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Sample long sentence
sentence = "The quick brown fox jumps over the lazy dog. Tokenization helps AI process text efficiently."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Convert tokens to numbers
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
