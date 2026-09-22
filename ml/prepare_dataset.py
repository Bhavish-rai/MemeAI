import os
import pandas as pd


# -----------------------------------
# 1. Define paths
# -----------------------------------

LABELS_FILE = "labels.csv"
IMAGE_FOLDER = "images/images"
OUTPUT_FILE = "dataset/memes_real.csv"


# -----------------------------------
# 2. Load the original labels
# -----------------------------------

data = pd.read_csv(LABELS_FILE)

print("\nOriginal dataset:")
print(f"Rows: {len(data)}")
print(f"Columns: {list(data.columns)}")


# -----------------------------------
# 3. Keep the columns we need
# -----------------------------------

data = data[
    [
        "image_name",
        "text_ocr",
        "text_corrected",
        "overall_sentiment"
    ]
]


# -----------------------------------
# 4. Remove rows without text
# -----------------------------------

data["text_corrected"] = data["text_corrected"].fillna("")

data = data[
    data["text_corrected"].str.strip() != ""
]


# -----------------------------------
# 5. Check whether images exist
# -----------------------------------

data["image_path"] = data["image_name"].apply(
    lambda name: os.path.join(IMAGE_FOLDER, name)
)

data["image_exists"] = data["image_path"].apply(
    os.path.exists
)


# -----------------------------------
# 6. Remove rows whose images
#    are missing
# -----------------------------------

missing_images = (~data["image_exists"]).sum()

print(f"\nMissing images: {missing_images}")

data = data[
    data["image_exists"]
]


# -----------------------------------
# 7. Create a clean dataset
# -----------------------------------

clean_data = pd.DataFrame({
    "id": range(1, len(data) + 1),
    "image": data["image_name"],
    "caption": data["text_corrected"],
    "sentiment": data["overall_sentiment"]
})


# -----------------------------------
# 8. Save the cleaned dataset
# -----------------------------------

clean_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------------
# 9. Show results
# -----------------------------------

print("\nClean dataset created successfully!")

print(f"Total memes: {len(clean_data)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst 5 rows:")
print(clean_data.head())