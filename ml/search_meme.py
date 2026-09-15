import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load the dataset
data = pd.read_csv("dataset/memes.csv")

# 2. Load the AI embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Get meme captions
captions = data["caption"].tolist()

# 4. Convert all meme captions into embeddings
meme_embeddings = model.encode(captions)


# 5. User's query
query = "I got very bad marks in my exam"

# 6. Convert user's query into an embedding
query_embedding = model.encode([query])


# 7. Calculate similarity
similarities = cosine_similarity(
    query_embedding,
    meme_embeddings
)[0]


# 8. Add similarity score to our dataset
data["similarity"] = similarities


# 9. Sort memes from most similar to least similar
results = data.sort_values(
    by="similarity",
    ascending=False
)


# 10. Display top 5 results
print("\nUser Query:")
print(query)

print("\nTop Matching Memes:\n")

for _, row in results.head(5).iterrows():
    print(f"Movie: {row['movie']}")
    print(f"Character: {row['character']}")
    print(f"Caption: {row['caption']}")
    print(f"Similarity: {row['similarity']:.2f}")
    print("-" * 40)