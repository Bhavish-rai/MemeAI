import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/meme_embeddings.npy"
EVALUATION_FILE = "dataset/evaluation_queries.csv"


print("\n" + "=" * 70)
print("                    MEMEAI SEARCH EVALUATION")
print("=" * 70)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("\nLoading dataset...")

data = pd.read_csv(DATASET_FILE)

print(f"Total memes: {len(data)}")


# --------------------------------------------------
# 2. Load embeddings
# --------------------------------------------------

print("\nLoading meme embeddings...")

meme_embeddings = np.load(EMBEDDINGS_FILE)

print(f"Embedding shape: {meme_embeddings.shape}")


if len(data) != len(meme_embeddings):
    raise ValueError(
        "Dataset and embeddings have different numbers of rows."
    )


# --------------------------------------------------
# 3. Load AI model
# --------------------------------------------------

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("AI model loaded successfully!")


# --------------------------------------------------
# 4. Load evaluation queries
# --------------------------------------------------

print("\nLoading evaluation queries...")

evaluation_data = pd.read_csv(EVALUATION_FILE)

print(
    f"Total evaluation queries: "
    f"{len(evaluation_data)}"
)


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

top1_scores = []
recall5_scores = []
mrr_scores = []


for _, evaluation_row in evaluation_data.iterrows():

    query = evaluation_row["query"]

    keywords = [
        keyword.strip().lower()
        for keyword in evaluation_row["keywords"].split("|")
    ]

    # Create query embedding
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        meme_embeddings
    )[0]

    # Rank all memes
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    # ----------------------------------------------
    # Find first relevant result
    # ----------------------------------------------

    first_relevant_rank = None

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        caption = str(
            data.iloc[index]["caption"]
        ).lower()

        is_relevant = any(
            keyword in caption
            for keyword in keywords
        )

        if is_relevant:
            first_relevant_rank = rank
            break

    # ----------------------------------------------
    # Top-1 Accuracy
    # ----------------------------------------------

    top1 = (
        1
        if first_relevant_rank == 1
        else 0
    )

    top1_scores.append(top1)

    # ----------------------------------------------
    # Recall@5
    # ----------------------------------------------

    recall5 = (
        1
        if first_relevant_rank is not None
        and first_relevant_rank <= 5
        else 0
    )

    recall5_scores.append(recall5)

    # ----------------------------------------------
    # MRR
    # ----------------------------------------------

    if first_relevant_rank is not None:
        mrr = 1 / first_relevant_rank
    else:
        mrr = 0

    mrr_scores.append(mrr)

    # ----------------------------------------------
    # Display query result
    # ----------------------------------------------

    print("\n" + "-" * 70)

    print(f"Query: {query}")

    if first_relevant_rank is not None:
        print(
            f"First relevant result: "
            f"Rank {first_relevant_rank}"
        )
    else:
        print(
            "First relevant result: "
            "Not found"
        )

    print(
        f"Top-1: "
        f"{'PASS' if top1 else 'FAIL'}"
    )

    print(
        f"Recall@5: "
        f"{'PASS' if recall5 else 'FAIL'}"
    )

    print(
        f"MRR: "
        f"{mrr:.3f}"
    )


# --------------------------------------------------
# 6. Calculate final metrics
# --------------------------------------------------

top1_accuracy = (
    sum(top1_scores)
    / len(top1_scores)
)

recall_at_5 = (
    sum(recall5_scores)
    / len(recall5_scores)
)

mean_reciprocal_rank = (
    sum(mrr_scores)
    / len(mrr_scores)
)


# --------------------------------------------------
# 7. Final report
# --------------------------------------------------

print("\n\n" + "=" * 70)
print("                       FINAL RESULTS")
print("=" * 70)

print(
    f"\nTop-1 Accuracy : "
    f"{top1_accuracy * 100:.2f}%"
)

print(
    f"Recall@5       : "
    f"{recall_at_5 * 100:.2f}%"
)

print(
    f"MRR            : "
    f"{mean_reciprocal_rank:.3f}"
)

print(
    f"\nEvaluation Queries: "
    f"{len(evaluation_data)}"
)

print("\n" + "=" * 70)