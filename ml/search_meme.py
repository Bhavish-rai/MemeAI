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
# 2. Load dataset
# -----------------------------------

data = pd.read_csv(DATASET_FILE)

print("\nMeme dataset loaded!")
print(f"Total memes: {len(data)}")


# -----------------------------------
# 3. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 4. Load saved embeddings
# -----------------------------------

print("Loading saved meme embeddings...")

meme_embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embeddings loaded: {meme_embeddings.shape}")


# -----------------------------------
# 5. Validate dataset and embeddings
# -----------------------------------

if len(data) != len(meme_embeddings):
    raise ValueError(
        "Dataset and embedding count do not match."
    )


# -----------------------------------
# 6. Get user query
# -----------------------------------

query = input(
    "\nEnter what you want a meme about: "
)


# -----------------------------------
# 7. Convert query into embedding
# -----------------------------------

query_embedding = model.encode([query])


# -----------------------------------
# 8. Calculate similarity
# -----------------------------------

similarities = cosine_similarity(
    query_embedding,
    meme_embeddings
)[0]


# -----------------------------------
# 9. Add similarity scores
# -----------------------------------

data["similarity"] = similarities


# -----------------------------------
# 10. Sort results
# -----------------------------------

results = data.sort_values(
    by="similarity",
    ascending=False
)


# -----------------------------------
# 11. Display top 5 results
# -----------------------------------

print("\n" + "=" * 70)
print("                         MEME AI")
print("=" * 70)

print(f"\nQuery: {query}")

print("\nTop Matching Memes:\n")


for index, (_, row) in enumerate(
    results.head(5).iterrows(),
    start=1
):

    print(f"{index}. Image: {row['image']}")

    print(
        f"   Caption: {row['caption']}"
    )

    print(
        f"   Sentiment: {row['sentiment']}"
    )

    print(
        f"   Similarity: {row['similarity']:.3f}"
    )

    print("-" * 70)