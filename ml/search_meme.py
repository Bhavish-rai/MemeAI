import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Load the real meme dataset
# -----------------------------------

data = pd.read_csv("dataset/memes_real.csv")


# -----------------------------------
# 2. Load the AI model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 3. Get meme captions
# -----------------------------------

captions = data["caption"].tolist()


# -----------------------------------
# 4. Convert all meme captions
#    into embeddings
# -----------------------------------

print("\nCreating meme embeddings...")

meme_embeddings = model.encode(
    captions,
    show_progress_bar=True
)


# -----------------------------------
# 5. Get query from user
# -----------------------------------

query = input("\nEnter what you want a meme about: ")


# -----------------------------------
# 6. Convert query into embedding
# -----------------------------------

query_embedding = model.encode([query])


# -----------------------------------
# 7. Calculate similarity
# -----------------------------------

similarities = cosine_similarity(
    query_embedding,
    meme_embeddings
)[0]


# -----------------------------------
# 8. Add similarity scores
# -----------------------------------

data["similarity"] = similarities


# -----------------------------------
# 9. Sort by similarity
# -----------------------------------

results = data.sort_values(
    by="similarity",
    ascending=False
)


# -----------------------------------
# 10. Display top results
# -----------------------------------

print("\n" + "=" * 60)
print("                    MEME AI")
print("=" * 60)

print(f"\nQuery: {query}")

print("\nTop Matching Memes:\n")

for index, (_, row) in enumerate(
    results.head(5).iterrows(),
    start=1
):

    print(f"{index}. Image: {row['image']}")
    print(f"   Caption: {row['caption']}")
    print(f"   Sentiment: {row['sentiment']}")
    print(f"   Similarity: {row['similarity']:.3f}")

    print("-" * 60)