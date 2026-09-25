import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/image_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("                    MEMEAI IMAGE SEARCH")
print("=" * 70)

print("\nLoading dataset...")

data = pd.read_csv(DATASET_FILE)

print(f"Total memes: {len(data)}")


# ============================================================
# LOAD IMAGE EMBEDDINGS
# ============================================================

print("\nLoading image embeddings...")

image_embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    f"Embedding shape: "
    f"{image_embeddings.shape}"
)


if len(data) != len(image_embeddings):

    raise ValueError(
        "Dataset and image embeddings "
        "have different numbers of rows."
    )


# ============================================================
# SELECT DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")


# ============================================================
# LOAD CLIP
# ============================================================

print("\nLoading CLIP model...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("CLIP model loaded successfully!")


# ============================================================
# GET IMAGE EMBEDDING
# ============================================================

def get_image_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = (
            model.visual_projection(
                pooled_output
            )
        )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1
        )

    return image_features.cpu().numpy()


# ============================================================
# ASK FOR IMAGE
# ============================================================

image_path = input(
    "\nEnter the path of the image you want to search: "
).strip()


if not os.path.exists(image_path):

    print(
        "\nERROR: Image file not found."
    )

    exit()


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

print("\nAnalyzing image...")

query_embedding = get_image_embedding(
    image_path
)

print("Image analyzed successfully!")


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

print("\nFinding visually similar memes...")

similarities = cosine_similarity(
    query_embedding,
    image_embeddings
)[0]


# ============================================================
# GET TOP RESULTS
# ============================================================

top_indices = np.argsort(
    similarities
)[::-1][:5]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("                    SEARCH RESULTS")
print("=" * 70)

for rank, index in enumerate(
    top_indices,
    start=1
):

    row = data.iloc[index]

    print(
        f"\n{rank}. {row['image']}"
    )

    print(
        f"   Caption: "
        f"{row['caption']}"
    )

    print(
        f"   Sentiment: "
        f"{row['sentiment']}"
    )

    print(
        f"   Similarity: "
        f"{similarities[index]:.4f}"
    )

    print("-" * 70)