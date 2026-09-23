import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Define paths
# -----------------------------------

DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/meme_embeddings.npy"


# -----------------------------------
# 2. Load dataset
# -----------------------------------

data = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully!")
print(f"Total memes: {len(data)}")


# -----------------------------------
# 3. Check required column
# -----------------------------------

if "search_text" not in data.columns:
    raise ValueError(
        "search_text column not found. "
        "Run ml/prepare_dataset.py first."
    )


# -----------------------------------
# 4. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 5. Get searchable meme text
# -----------------------------------

search_text = (
    data["search_text"]
    .fillna("")
    .tolist()
)


# -----------------------------------
# 6. Generate embeddings
# -----------------------------------

print("\nCreating meme embeddings...")

embeddings = model.encode(
    search_text,
    show_progress_bar=True
)


# -----------------------------------
# 7. Save embeddings
# -----------------------------------

np.save(
    EMBEDDINGS_FILE,
    embeddings
)


# -----------------------------------
# 8. Display results
# -----------------------------------

print("\n" + "=" * 60)
print("EMBEDDINGS CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nNumber of memes: {len(search_text)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Saved to: {EMBEDDINGS_FILE}")