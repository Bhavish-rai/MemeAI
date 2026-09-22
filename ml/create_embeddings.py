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
# 3. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 4. Get meme captions
# -----------------------------------

captions = data["caption"].fillna("").tolist()


# -----------------------------------
# 5. Generate embeddings
# -----------------------------------

print("\nCreating meme embeddings...")

embeddings = model.encode(
    captions,
    show_progress_bar=True
)


# -----------------------------------
# 6. Save embeddings
# -----------------------------------

np.save(
    EMBEDDINGS_FILE,
    embeddings
)


# -----------------------------------
# 7. Display information
# -----------------------------------

print("\n" + "=" * 60)
print("EMBEDDINGS CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nNumber of memes: {len(captions)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Saved to: {EMBEDDINGS_FILE}")