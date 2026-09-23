import os
import pandas as pd


# -----------------------------------
# 1. Define paths
# -----------------------------------

LABELS_FILE = "labels.csv"
IMAGE_FOLDER = "images/images"
OUTPUT_FILE = "dataset/memes_real.csv"


# -----------------------------------
# 2. Load original labels
# -----------------------------------

data = pd.read_csv(LABELS_FILE)

print("\nOriginal dataset:")
print(f"Rows: {len(data)}")
print(f"Columns: {list(data.columns)}")


# -----------------------------------
# 3. Keep required columns
# -----------------------------------

data = data[
    [
        "image_name",
        "text_ocr",
        "text_corrected",
        "overall_sentiment"
    ]
].copy()


# -----------------------------------
# 4. Handle missing values
# -----------------------------------

data["text_ocr"] = data["text_ocr"].fillna("")
data["text_corrected"] = data["text_corrected"].fillna("")
data["overall_sentiment"] = data["overall_sentiment"].fillna("unknown")


# -----------------------------------
# 5. Remove rows without text
# -----------------------------------

data = data[
    data["text_corrected"].str.strip() != ""
].copy()


# -----------------------------------
# 6. Check whether images exist
# -----------------------------------

data["image_path"] = data["image_name"].apply(
    lambda name: os.path.join(IMAGE_FOLDER, name)
)

data["image_exists"] = data["image_path"].apply(
    os.path.exists
)


missing_images = (~data["image_exists"]).sum()

print(f"\nMissing images: {missing_images}")


# -----------------------------------
# 7. Remove missing images
# -----------------------------------

data = data[
    data["image_exists"]
].copy()


# -----------------------------------
# 8. Create searchable text
# -----------------------------------

data["search_text"] = (
    "Meme text: "
    + data["text_corrected"]
    + ". OCR text: "
    + data["text_ocr"]
    + ". Sentiment: "
    + data["overall_sentiment"]
)


# -----------------------------------
# 9. Create clean dataset
# -----------------------------------

clean_data = pd.DataFrame({
    "id": range(1, len(data) + 1),
    "image": data["image_name"],
    "caption": data["text_corrected"],
    "ocr_text": data["text_ocr"],
    "sentiment": data["overall_sentiment"],
    "search_text": data["search_text"]
})


# -----------------------------------
# 10. Save dataset
# -----------------------------------

clean_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------------
# 11. Display results
# -----------------------------------

print("\nClean dataset created successfully!")

print(f"Total memes: {len(clean_data)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst 5 rows:")
print(
    clean_data[
        [
            "id",
            "image",
            "caption",
            "sentiment"
        ]
    ].head()
)