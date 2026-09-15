from sentence_transformers import SentenceTransformer

# Load the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "I got very bad marks in my exam"

# Convert text into an embedding
embedding = model.encode(text)

print("Text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nEmbedding size:")
print(len(embedding))