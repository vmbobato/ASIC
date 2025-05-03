import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import os

model_path = "classification_folder/model/b2-ade_30_epochs.pth"
config_path = "classification_folder/model/config.json"
OUTPUT_PATH = "output_segmentation/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper structures
id2label = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown"
}
class_colors = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (255, 0, 255),
    3: (0, 255, 0),
    4: (0, 0, 255),
    5: (255, 255, 255),
    6: (0, 0, 0)
}

# load model
config = SegformerConfig.from_json_file(config_path)
model = SegformerForSemanticSegmentation(config)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

transform = T.Compose([
    T.Resize((240, 240)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def compute_coverage(mask : np.array, num_classes=7) -> dict:
    """
    Calculate the percentage coverage of each land class in a segmentation mask.

    Args:
        mask (np.array): A 2D numpy array containing class IDs for each pixel
        num_classes (int, optional): Number of classes in the segmentation. Defaults to 7.

    Returns:
        dict: A dictionary mapping class names to their percentage coverage in the image.
              Example: {'urban_land': 25.5, 'agriculture_land': 30.2, ...}
    """
    total_pixels = mask.size
    coverage = {}
    for class_id in range(num_classes):
        count = np.sum(mask == class_id)
        percent = count / total_pixels * 100
        coverage[id2label[class_id]] = round(percent, 2)
    return coverage


def get_masks(image_path : str) -> np.array:
    """
    Generate a segmentation mask for an input image using the Segformer model.

    Args:
        image_path (str): Path to the input image file

    Returns:
        np.array: A 2D numpy array containing class IDs for each pixel in the image.
                 The array has the same dimensions as the input image.
    """
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (W, H)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(orig_size[1], orig_size[0]), 
            mode='bilinear',
            align_corners=False
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return pred_mask

def run_analysis(mask : np.array, img_name : str) -> tuple[str, str]:
    """
    Analyze a segmentation mask and generate a visual representation and coverage statistics.

    Args:
        mask (np.array): A 2D numpy array containing class IDs for each pixel
        img_name (str): Original name of the input image file

    Returns:
        tuple[str, str]: A tuple containing:
            - str: A formatted string containing coverage statistics for each class
            - str: The filename of the generated segmented image

    The function:
    1. Computes coverage statistics for each land class
    2. Generates a color-coded visualization of the segmentation
    3. Saves the visualization to the output directory
    4. Returns both the analysis text and the output filename
    """
    file_name_only = os.path.splitext(img_name)[0]
    output_filename = f"{file_name_only}_segmented.png"
    output = ''
    coverage = compute_coverage(mask)
    output += "\n> Class-wise Pixel Coverage:"
    for label, percent in coverage.items():
        if percent != 0:
            label_formatted = label.replace('_', ' ').title()
            output += f"\n> {label_formatted}: {percent:.2f}% - Total Area: {(percent/100 * 1498176):.2f} meter-squared (m2)" # based on 1 px = 0.5 m
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    Image.fromarray(color_mask).save(output_path)
    return output, output_filename
