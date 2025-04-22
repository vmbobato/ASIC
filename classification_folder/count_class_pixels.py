import os
from PIL import Image
import numpy as np
import torch

# Path to dataset
mask_dir = "classification_folder/DeepGlobe_Converted_Dataset/train"
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith("_mask.png")])

num_classes = 7
class_counts = np.zeros(num_classes, dtype=np.int64)

# Count pixels per class
for file in mask_files:
    mask_path = os.path.join(mask_dir, file)
    mask = np.array(Image.open(mask_path))

    for class_id in range(num_classes):
        class_counts[class_id] += np.sum(mask == class_id)

# Compute inverse-frequency weights
total_pixels = class_counts.sum()
class_weights = total_pixels / (num_classes * (class_counts + 1e-6))
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Display weights and counts
print("\nClass-wise Pixel Counts and Weights:\n")
for class_id in range(num_classes):
    print(f"Class {class_id}: Count = {class_counts[class_id]}, Weight = {class_weights[class_id]:.4f}")
