import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import os

print(os.getcwd())

model_path = "classification_folder/models/saved_models/b2-ade_30_epochs.pth"
config_path = "classification_folder/models/b2-ade-512-512/config.json"
OUTPUT_PATH = "output_segmentation/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

config = SegformerConfig.from_json_file(config_path)
model = SegformerForSemanticSegmentation(config)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def compute_coverage(mask, num_classes=7):
    total_pixels = mask.size
    coverage = {}
    for class_id in range(num_classes):
        count = np.sum(mask == class_id)
        percent = count / total_pixels * 100
        coverage[id2label[class_id]] = round(percent, 2)
    return coverage


def get_masks(image_path):
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

def run_analysis(mask, img_name):
    file_name_only = os.path.splitext(img_name)[0]
    output_filename = f"{file_name_only}_segmented.png"
    output = ''
    coverage = compute_coverage(mask)
    output += "\n> Class-wise Pixel Coverage:"
    for label, percent in coverage.items():
        if percent != 0:
            label_formatted = label.replace('_', ' ').title()
            output += f"\n> {label_formatted}: {percent:.2f}% - Total Area: {(percent/100 * 1498176):.2f} meter-squared (m2)"
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    Image.fromarray(color_mask).save(output_path)
    return output, output_filename