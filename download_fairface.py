from datasets import load_dataset
import os
from PIL import Image
import pandas as pd

# Download FairFace dataset (0.25 padding config)
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")

# Save dataset and images to a local directory
output_dir = "fairface"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)

# Save images and metadata
for split in ["train", "validation"]:
    split_dir = os.path.join(output_dir, split)
    metadata = []
    for idx, example in enumerate(dataset[split]):
        img = example["image"]
        img_path = os.path.join(split_dir, f"image_{idx}.jpg")
        img.save(img_path)
        example["image_path"] = img_path
        metadata.append(example)
    
    # Save metadata as CSV
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, f"fairface_{split}.csv"), index=False)

print(f"Dataset and images saved to {output_dir}")