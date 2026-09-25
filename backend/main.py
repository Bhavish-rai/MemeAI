from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from io import BytesIO

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel


app = FastAPI(
    title="MemeAI API",
    description="AI-powered semantic and image meme search API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "memes_real.csv"
)

TEXT_EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "meme_embeddings.npy"
)

IMAGE_EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "image_embeddings.npy"
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "images",
    "images"
)


# ============================================================
# SERVE IMAGES
# ============================================================

app.mount(
    "/images",
    StaticFiles(directory=IMAGE_FOLDER),
    name="images"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading meme dataset...")

data = pd.read_csv(DATASET_FILE)

print(
    f"Loaded {len(data)} memes"
)


# ============================================================
# LOAD TEXT EMBEDDINGS
# ============================================================

print("\nLoading text embeddings...")

text_embeddings = np.load(
    TEXT_EMBEDDINGS_FILE
)

print(
    f"Text embeddings: "
    f"{text_embeddings.shape}"
)


# ============================================================
# LOAD IMAGE EMBEDDINGS
# ============================================================

print("\nLoading image embeddings...")

image_embeddings = np.load(
    IMAGE_EMBEDDINGS_FILE
)

print(
    f"Image embeddings: "
    f"{image_embeddings.shape}"
)


# ============================================================
# LOAD TEXT MODEL
# ============================================================

print("\nLoading text AI model...")

text_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Text model loaded!")


# ============================================================
# LOAD CLIP
# ============================================================

print("\nLoading CLIP model...")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.to(device)
clip_model.eval()

print(
    f"CLIP loaded on {device}!"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "MemeAI API is running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "memes": len(data),
        "text_embedding_dimensions":
            text_embeddings.shape[1],
        "image_embedding_dimensions":
            image_embeddings.shape[1]
    }


# ============================================================
# TEXT SEARCH
# ============================================================

@app.get("/api/search")
def search_memes(
    query: str = Query(
        ...,
        min_length=1
    ),
    limit: int = Query(
        5,
        ge=1,
        le=20
    )
):

    query_embedding = text_model.encode(
        [query]
    )

    similarities = cosine_similarity(
        query_embedding,
        text_embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:limit]

    results = []

    for index in top_indices:

        row = data.iloc[index]

        image_name = row["image"]

        results.append({

            "image": image_name,

            "image_url":
                f"/images/{image_name}",

            "caption":
                row["caption"],

            "sentiment":
                row["sentiment"],

            "similarity":
                round(
                    float(similarities[index]),
                    4
                )
        })

    return {
        "query": query,
        "count": len(results),
        "results": results
    }


# ============================================================
# IMAGE SEARCH
# ============================================================

@app.post("/api/image-search")
async def image_search(
    file: UploadFile = File(...),
    limit: int = Query(
        5,
        ge=1,
        le=20
    )
):

    contents = await file.read()

    try:

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        return {
            "error": "Invalid image file."
        }


    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)


    # --------------------------------------------------------
    # CLIP image embedding
    # --------------------------------------------------------

    with torch.no_grad():

        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = (
            clip_model.visual_projection(
                pooled_output
            )
        )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1
        )


    query_embedding = (
        image_features
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        image_embeddings
    )[0]


    top_indices = np.argsort(
        similarities
    )[::-1][:limit]


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    for index in top_indices:

        row = data.iloc[index]

        image_name = row["image"]

        results.append({

            "image": image_name,

            "image_url":
                f"/images/{image_name}",

            "caption":
                row["caption"],

            "sentiment":
                row["sentiment"],

            "similarity":
                round(
                    float(similarities[index]),
                    4
                )
        })


    return {
        "filename": file.filename,
        "count": len(results),
        "results": results
    }