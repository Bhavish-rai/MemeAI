import os
import pandas as pd


LABELS_FILE = "labels.csv"
IMAGE_FOLDER = "images/images"
OUTPUT_FILE = "dataset/memes_real.csv"


print("\n" + "=" * 70)
print("                  MEMEAI DATASET PREPARATION")
print("=" * 70)


# --------------------------------------------------
# 1. Load original dataset
# --------------------------------------------------

print("\nLoading labels.csv...")

data = pd.read_csv(LABELS_FILE)

print(f"Original rows: {len(data)}")
print(f"Columns: {list(data.columns)}")


# --------------------------------------------------
# 2. Select required columns
# --------------------------------------------------

required_columns = [
    "image_name",
    "text_ocr",
    "text_corrected",
    "overall_sentiment"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


data = data[required_columns].copy()


# --------------------------------------------------
# 3. Clean missing values
# --------------------------------------------------

data["text_ocr"] = (
    data["text_ocr"]
    .fillna("")
    .astype(str)
)

data["text_corrected"] = (
    data["text_corrected"]
    .fillna("")
    .astype(str)
)

data["overall_sentiment"] = (
    data["overall_sentiment"]
    .fillna("unknown")
    .astype(str)
)


# --------------------------------------------------
# 4. Remove memes without text
# --------------------------------------------------

data = data[
    data["text_corrected"].str.strip() != ""
].copy()


# --------------------------------------------------
# 5. Create image paths
# --------------------------------------------------

data["image_path"] = data["image_name"].apply(
    lambda name: os.path.join(
        IMAGE_FOLDER,
        name
    )
)


data["image_exists"] = data["image_path"].apply(
    os.path.exists
)


missing_images = (
    ~data["image_exists"]
).sum()


print(
    f"\nMissing images: "
    f"{missing_images}"
)


# Remove missing images

data = data[
    data["image_exists"]
].copy()


# --------------------------------------------------
# 6. Create structured search text
# --------------------------------------------------

data["search_text"] = (
    "Caption: "
    + data["text_corrected"].str.strip()
    + ". OCR: "
    + data["text_ocr"].str.strip()
    + ". Sentiment: "
    + data["overall_sentiment"].str.strip()
)


# --------------------------------------------------
# 7. Create final dataset
# --------------------------------------------------

clean_data = pd.DataFrame({

    "id": range(
        1,
        len(data) + 1
    ),

    "image": data["image_name"],

    "caption": data["text_corrected"],

    "ocr_text": data["text_ocr"],

    "sentiment": data["overall_sentiment"],

    "search_text": data["search_text"]

})


# --------------------------------------------------
# 8. Save dataset
# --------------------------------------------------

clean_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 9. Display summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("                  DATASET READY")
print("=" * 70)

print(
    f"\nTotal memes: "
    f"{len(clean_data)}"
)

print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)

print("\nColumns:")

for column in clean_data.columns:
    print(f"- {column}")


print("\nSample search representation:")

print(
    clean_data.iloc[0]["search_text"]
)

print("\n" + "=" * 70)