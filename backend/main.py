from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import numpy as np
import pandas as pd


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Create FastAPI application
# -----------------------------------

app = FastAPI(
    title="MemeAI API",
    description="AI-powered semantic meme search API",
    version="1.0.0"
)


# -----------------------------------
# 2. Enable CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# 3. Define project paths
# -----------------------------------

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

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "meme_embeddings.npy"
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "images",
    "images"
)
app.mount(
    "/images",
    StaticFiles(directory=IMAGE_FOLDER),
    name="images"
)


# -----------------------------------
# 4. Load dataset
# -----------------------------------

print("\nLoading meme dataset...")

data = pd.read_csv(
    DATASET_FILE
)

print(
    f"Loaded {len(data)} memes"
)


# -----------------------------------
# 5. Load embeddings
# -----------------------------------

print("\nLoading meme embeddings...")

meme_embeddings = np.load(
    EMBEDDINGS_FILE
)

print(
    f"Loaded embeddings: "
    f"{meme_embeddings.shape}"
)


# -----------------------------------
# 6. Validate data
# -----------------------------------

if len(data) != len(meme_embeddings):

    raise ValueError(
        "Dataset and embeddings "
        "have different numbers of rows."
    )


# -----------------------------------
# 7. Load AI model
# -----------------------------------

print("\nLoading AI model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("AI model loaded successfully!")


# -----------------------------------
# 8. Root endpoint
# -----------------------------------

@app.get("/")
def root():

    return {
        "message": "MemeAI API is running",
        "version": "1.0.0"
    }


# -----------------------------------
# 9. Health endpoint
# -----------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "memes": len(data),
        "embedding_dimensions": meme_embeddings.shape[1]
    }


# -----------------------------------
# 10. Search endpoint
# -----------------------------------

@app.get("/api/search")
def search_memes(
    query: str = Query(
        ...,
        min_length=1,
        description="Text describing the meme"
    ),
    limit: int = Query(
        5,
        ge=1,
        le=20
    )
):

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
    # Get top results
    # -----------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:limit]


    # -----------------------------------
    # Build response
    # -----------------------------------

    results = []

    for index in top_indices:

        row = data.iloc[index]

        image_name = row["image"]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )

        results.append({
    "image": image_name,
    "image_url": f"/images/{image_name}",
    "caption": row["caption"],
    "sentiment": row["sentiment"],
    "similarity": round(
        float(similarities[index]),
        4
    ),
    "image_exists": os.path.exists(image_path)
})


    return {
        "query": query,
        "count": len(results),
        "results": results
    }