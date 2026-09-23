import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Define paths
# -----------------------------------

DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/meme_embeddings.npy"
EVALUATION_FILE = "dataset/evaluation_queries.csv"


# -----------------------------------
# 2. Load dataset
# -----------------------------------

data = pd.read_csv(DATASET_FILE)

print("\nMeme dataset loaded!")
print(f"Total memes: {len(data)}")


# -----------------------------------
# 3. Load evaluation queries
# -----------------------------------

evaluation_data = pd.read_csv(
    EVALUATION_FILE
)

print(
    f"Evaluation queries: "
    f"{len(evaluation_data)}"
)


# -----------------------------------
# 4. Load embeddings
# -----------------------------------

meme_embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    f"Embeddings shape: "
    f"{meme_embeddings.shape}"
)


# -----------------------------------
# 5. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------
# 6. Evaluation
# -----------------------------------

recall_at_5 = 0

total_queries = len(
    evaluation_data
)


print("\n" + "=" * 70)
print("                 MEME AI EVALUATION")
print("=" * 70)


for _, row in evaluation_data.iterrows():

    query = row["query"]

    keywords = [
        keyword.strip().lower()
        for keyword in row["keywords"].split("|")
    ]

    # -----------------------------------
    # Create query embedding
    # -----------------------------------

    query_embedding = model.encode(
        [query]
    )

    # -----------------------------------
    # Calculate similarity
    # -----------------------------------

    similarities = cosine_similarity(
        query_embedding,
        meme_embeddings
    )[0]

    # -----------------------------------
    # Get top 5 results
    # -----------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    # -----------------------------------
    # Check relevance
    # -----------------------------------

    relevant_found = False

    for index in top_indices:

        caption = str(
            data.iloc[index]["caption"]
        ).lower()

        if any(
            keyword in caption
            for keyword in keywords
        ):
            relevant_found = True
            break

    if relevant_found:
        recall_at_5 += 1

    # -----------------------------------
    # Display
    # -----------------------------------

    print(f"\nQuery: {query}")

    print(
        f"Relevant result in Top-5: "
        f"{'YES' if relevant_found else 'NO'}"
    )


# -----------------------------------
# 7. Calculate Recall@5
# -----------------------------------

recall_percentage = (
    recall_at_5 / total_queries
) * 100


# -----------------------------------
# 8. Display final result
# -----------------------------------

print("\n" + "=" * 70)
print("                    FINAL RESULT")
print("=" * 70)

print(
    f"\nTotal queries: {total_queries}"
)

print(
    f"Queries with relevant result in Top-5: "
    f"{recall_at_5}"
)

print(
    f"Recall@5: "
    f"{recall_percentage:.2f}%"
)

print("\nEvaluation completed.")