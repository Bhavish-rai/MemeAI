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

print("\nDataset loaded!")
print(f"Total memes: {len(data)}")


# -----------------------------------
# 3. Load embeddings
# -----------------------------------

meme_embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    f"Embeddings loaded: {meme_embeddings.shape}"
)


# -----------------------------------
# 4. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------
# 5. Evaluation queries
# -----------------------------------

test_queries = [
    {
        "query": "exam tomorrow and I have not studied",
        "keywords": [
            "exam",
            "study",
            "studying"
        ]
    },
    {
        "query": "my friend borrowed money and did not return it",
        "keywords": [
            "money",
            "friend",
            "borrowed"
        ]
    },
    {
        "query": "college assignment deadline",
        "keywords": [
            "assignment",
            "college",
            "deadline"
        ]
    },
    {
        "query": "I finished my project successfully",
        "keywords": [
            "project",
            "success",
            "completed"
        ]
    },
    {
        "query": "everyone is studying except me",
        "keywords": [
            "study",
            "studying",
            "student"
        ]
    },
    {
        "query": "feeling angry at someone",
        "keywords": [
            "angry",
            "anger"
        ]
    },
    {
        "query": "feeling happy after success",
        "keywords": [
            "happy",
            "success"
        ]
    },
    {
        "query": "college life is difficult",
        "keywords": [
            "college",
            "student"
        ]
    }
]


# -----------------------------------
# 6. Evaluation variables
# -----------------------------------

top_1_correct = 0
top_5_correct = 0

results = []


# -----------------------------------
# 7. Evaluate each query
# -----------------------------------

print("\n" + "=" * 70)
print("                    MEME AI EVALUATION")
print("=" * 70)


for test in test_queries:

    query = test["query"]
    keywords = test["keywords"]

    # Create query embedding
    query_embedding = model.encode(
        [query]
    )

    # Calculate similarity
    similarities = cosine_similarity(
        query_embedding,
        meme_embeddings
    )[0]

    # Get ranking
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    top_1_index = ranked_indices[0]

    top_5_indices = ranked_indices[:5]

    # Get captions
    top_1_caption = str(
        data.iloc[top_1_index]["caption"]
    ).lower()

    top_5_captions = [
        str(data.iloc[index]["caption"]).lower()
        for index in top_5_indices
    ]

    # -----------------------------------
    # Check keyword match
    # -----------------------------------

    top_1_match = any(
        keyword.lower() in top_1_caption
        for keyword in keywords
    )

    top_5_match = any(
        any(
            keyword.lower() in caption
            for keyword in keywords
        )
        for caption in top_5_captions
    )

    if top_1_match:
        top_1_correct += 1

    if top_5_match:
        top_5_correct += 1

    # Save result
    results.append({
        "query": query,
        "top_1_match": top_1_match,
        "top_5_match": top_5_match,
        "top_similarity": similarities[top_1_index]
    })

    # -----------------------------------
    # Display result
    # -----------------------------------

    print(f"\nQuery: {query}")

    print(
        f"Top result: "
        f"{data.iloc[top_1_index]['image']}"
    )

    print(
        f"Similarity: "
        f"{similarities[top_1_index]:.3f}"
    )

    print(
        f"Top-1 match: "
        f"{'YES' if top_1_match else 'NO'}"
    )

    print(
        f"Top-5 match: "
        f"{'YES' if top_5_match else 'NO'}"
    )


# -----------------------------------
# 8. Calculate metrics
# -----------------------------------

total_queries = len(test_queries)

top_1_accuracy = (
    top_1_correct / total_queries
) * 100

top_5_accuracy = (
    top_5_correct / total_queries
) * 100


# -----------------------------------
# 9. Display final results
# -----------------------------------

print("\n" + "=" * 70)
print("                    FINAL RESULTS")
print("=" * 70)

print(
    f"\nTotal queries: {total_queries}"
)

print(
    f"Top-1 Accuracy: "
    f"{top_1_accuracy:.2f}%"
)

print(
    f"Top-5 Accuracy: "
    f"{top_5_accuracy:.2f}%"
)

print(
    "\nEvaluation completed."
)