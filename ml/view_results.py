import os
import webbrowser
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Paths
# -----------------------------------

DATASET_FILE = "dataset/memes_real.csv"
EMBEDDINGS_FILE = "dataset/meme_embeddings.npy"
IMAGE_FOLDER = "images/images"
OUTPUT_FILE = "meme_results.html"


# -----------------------------------
# 2. Load dataset
# -----------------------------------

data = pd.read_csv(DATASET_FILE)

# Load saved embeddings
meme_embeddings = np.load(EMBEDDINGS_FILE)

# Load model
print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# 3. Get query
# -----------------------------------

query = input("\nEnter what you want a meme about: ")


# -----------------------------------
# 4. Create query embedding
# -----------------------------------

query_embedding = model.encode([query])


# -----------------------------------
# 5. Calculate similarity
# -----------------------------------

similarities = cosine_similarity(
    query_embedding,
    meme_embeddings
)[0]

data["similarity"] = similarities


# -----------------------------------
# 6. Get top 5 results
# -----------------------------------

results = data.sort_values(
    by="similarity",
    ascending=False
).head(5)


# -----------------------------------
# 7. Create HTML cards
# -----------------------------------

cards = ""

for index, (_, row) in enumerate(results.iterrows(), start=1):

    image_path = os.path.join(
        IMAGE_FOLDER,
        row["image"]
    )

    image_path = image_path.replace("\\", "/")

    cards += f"""
    <div class="card">

        <div class="rank">
            #{index}
        </div>

        <img src="{image_path}" alt="Meme">

        <div class="content">

            <div class="similarity">
                Similarity: {row["similarity"]:.3f}
            </div>

            <p>
                {row["caption"]}
            </p>

            <span>
                Sentiment: {row["sentiment"]}
            </span>

        </div>

    </div>
    """


# -----------------------------------
# 8. Create HTML page
# -----------------------------------

html = f"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <title>MemeAI Results</title>

    <style>

        body {{
            margin: 0;
            padding: 40px;
            font-family: Arial, sans-serif;
            background: #f8fafc;
            color: #1e293b;
        }}

        .container {{
            max-width: 1100px;
            margin: auto;
        }}

        h1 {{
            margin-bottom: 8px;
        }}

        .query {{
            color: #64748b;
            margin-bottom: 30px;
        }}

        .results {{
            display: grid;
            grid-template-columns: repeat(
                auto-fit,
                minmax(280px, 1fr)
            );
            gap: 24px;
        }}

        .card {{
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}

        .card img {{
            width: 100%;
            height: 300px;
            object-fit: contain;
            background: #e2e8f0;
        }}

        .content {{
            padding: 18px;
        }}

        .rank {{
            position: absolute;
            margin: 12px;
            background: #1e293b;
            color: white;
            padding: 6px 10px;
            border-radius: 8px;
            font-weight: bold;
        }}

        .similarity {{
            font-weight: bold;
            margin-bottom: 10px;
        }}

        p {{
            line-height: 1.5;
        }}

        span {{
            font-size: 14px;
            color: #64748b;
        }}

    </style>

</head>

<body>

<div class="container">

    <h1>🧠 MemeAI</h1>

    <div class="query">
        Search: "{query}"
    </div>

    <div class="results">

        {cards}

    </div>

</div>

</body>

</html>
"""


# -----------------------------------
# 9. Save HTML
# -----------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(html)


# -----------------------------------
# 10. Open browser
# -----------------------------------

output_path = os.path.abspath(OUTPUT_FILE)

print("\nResults generated successfully!")
print(f"Saved to: {output_path}")

webbrowser.open(
    "file://" + output_path
)