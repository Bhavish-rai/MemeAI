import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Load the dataset
# -----------------------------------

data = pd.read_csv("dataset/memes.csv")


# -----------------------------------
# 2. Load the AI model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 3. Create a rich description
# -----------------------------------

data["description"] = (
    "Caption: " + data["caption"] +
    ". Tags: " + data["tags"] +
    ". Emotion: " + data["emotion"] +
    ". Movie: " + data["movie"] +
    ". Character: " + data["character"]
)


# -----------------------------------
# 4. Convert meme descriptions
#    into embeddings
# -----------------------------------

meme_embeddings = model.encode(
    data["description"].tolist()
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
# 10. Apply similarity threshold
# -----------------------------------

THRESHOLD = 0.40

relevant_results = results[
    results["similarity"] >= THRESHOLD
]


# -----------------------------------
# 11. Display results
# -----------------------------------

print("\n" + "=" * 50)
print("                 🎭 MEME AI")
print("=" * 50)

print(f"\nQuery: {query}")

if relevant_results.empty:

    print("\n😕 No highly relevant meme found.")
    print("Try describing your situation differently.")

else:

    print("\n🔥 Related Memes:\n")

    for index, (_, row) in enumerate(
        relevant_results.head(5).iterrows(),
        start=1
    ):

        print(f"{index}. Movie: {row['movie']}")
        print(f"   Character: {row['character']}")
        print(f"   Emotion: {row['emotion']}")
        print(f"   Caption: {row['caption']}")
        print(f"   Similarity: {row['similarity']:.2f}")

        print("-" * 50)