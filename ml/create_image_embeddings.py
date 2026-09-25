import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "dataset/memes_real.csv"
IMAGE_FOLDER = "images/images"
OUTPUT_FILE = "dataset/image_embeddings.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 16


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("              MEMEAI IMAGE EMBEDDING GENERATOR")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_FILE)

print(f"Total memes: {len(data)}")


# ============================================================
# 2. SELECT DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device}")


# ============================================================
# 3. LOAD CLIP MODEL
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
# 4. FUNCTION TO CREATE IMAGE EMBEDDING
# ============================================================

def create_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    # Run image through CLIP vision encoder
    vision_outputs = model.vision_model(
        pixel_values=pixel_values
    )

    # Get pooled visual representation
    pooled_output = vision_outputs.pooler_output

    # Project into CLIP embedding space
    image_features = model.visual_projection(
        pooled_output
    )

    # Normalize embedding
    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1
    )

    return image_features.cpu().numpy()[0]


# ============================================================
# 5. PROCESS IMAGES
# ============================================================

embeddings = []

failed_images = []

total = len(data)

print("\nCreating image embeddings...")
print(f"Batch size: {BATCH_SIZE}")
print()


with torch.no_grad():

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_images = []
        batch_indices = []

        # ----------------------------------------------------
        # Load batch
        # ----------------------------------------------------

        for index in range(start, end):

            image_name = str(
                data.iloc[index]["image"]
            )

            image_path = os.path.join(
                IMAGE_FOLDER,
                image_name
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                batch_images.append(image)
                batch_indices.append(index)

            except Exception as error:

                print(
                    f"\nWarning: Could not open "
                    f"{image_name}"
                )

                print(
                    f"Reason: {error}"
                )

                failed_images.append(
                    image_name
                )

                # Keep row alignment
                batch_images.append(
                    Image.new(
                        "RGB",
                        (224, 224),
                        "white"
                    )
                )

                batch_indices.append(index)

        # ----------------------------------------------------
        # Process batch
        # ----------------------------------------------------

        try:

            inputs = processor(
                images=batch_images,
                return_tensors="pt"
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(device)

            # CLIP vision encoder
            vision_outputs = model.vision_model(
                pixel_values=pixel_values
            )

            # Pooled visual representation
            pooled_output = (
                vision_outputs.pooler_output
            )

            # CLIP projection
            image_features = (
                model.visual_projection(
                    pooled_output
                )
            )

            # Normalize
            image_features = F.normalize(
                image_features,
                p=2,
                dim=-1
            )

            batch_embeddings = (
                image_features
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            embeddings.extend(
                batch_embeddings
            )

        except Exception as error:

            print(
                f"\nError processing batch "
                f"{start + 1}-{end}: {error}"
            )

            # Add zero vectors to maintain alignment
            for _ in range(len(batch_images)):

                embeddings.append(
                    np.zeros(
                        512,
                        dtype=np.float32
                    )
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        processed = end

        if (
            processed % 100 == 0
            or processed == total
        ):

            print(
                f"Processed "
                f"{processed}/{total}"
            )


# ============================================================
# 6. CONVERT TO NUMPY
# ============================================================

embeddings = np.array(
    embeddings,
    dtype=np.float32
)


# ============================================================
# 7. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("                    EMBEDDING SUMMARY")
print("=" * 70)

print(
    f"\nNumber of images: "
    f"{len(embeddings)}"
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)

print(
    f"Failed images: "
    f"{len(failed_images)}"
)


if len(embeddings) != len(data):

    raise ValueError(
        "Number of embeddings does not "
        "match number of dataset rows."
    )


if embeddings.shape[1] != 512:

    raise ValueError(
        "Unexpected CLIP embedding dimension."
    )


# ============================================================
# 8. SAVE EMBEDDINGS
# ============================================================

np.save(
    OUTPUT_FILE,
    embeddings
)


print(
    f"\nSaved to: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# 9. FAILED IMAGE REPORT
# ============================================================

if failed_images:

    print("\nImages that could not be opened:")

    for image_name in failed_images[:20]:

        print(
            f"- {image_name}"
        )

    if len(failed_images) > 20:

        print(
            f"... and "
            f"{len(failed_images) - 20} more"
        )

else:

    print(
        "\nAll images were processed successfully!"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("          IMAGE EMBEDDINGS CREATED SUCCESSFULLY")
print("=" * 70)

print(
    "\nYou can now use:"
)

print(
    "dataset/image_embeddings.npy"
)

print()