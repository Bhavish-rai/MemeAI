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
#    for every meme
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
# 5. User query
# -----------------------------------

query = input("Enter what you want a meme about: ")


# -----------------------------------
# 6. Convert user query into embedding
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
# 8. Add similarity score
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
# 10. Display top 5 memes
# -----------------------------------

print("\nUser Query:")
print(query)

print("\nTop Matching Memes:\n")

for _, row in results.head(5).iterrows():

    print(f"Movie: {row['movie']}")
    print(f"Character: {row['character']}")
    print(f"Emotion: {row['emotion']}")
    print(f"Caption: {row['caption']}")
    print(f"Similarity: {row['similarity']:.2f}")

    print("-" * 40)