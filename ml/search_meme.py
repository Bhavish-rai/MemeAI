import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Define paths
# -----------------------------------

DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/meme_embeddings.npy"


# -----------------------------------
# 2. Load the meme dataset
# -----------------------------------

data = pd.read_csv(DATASET_FILE)

print("\nMeme dataset loaded!")
print(f"Total memes: {len(data)}")


# -----------------------------------
# 3. Load the AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 4. Load saved meme embeddings
# -----------------------------------

print("Loading saved meme embeddings...")

meme_embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {meme_embeddings.shape}")


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
# 9. Sort results
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