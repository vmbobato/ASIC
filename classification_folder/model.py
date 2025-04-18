import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from safetensors.torch import load_file
from tqdm import trange

MODEL_NAME = 'b2'

"""
This maps each class index (from the mask) to a name. Used for printing during evaluation.
"""
id2label = {
    0: "urban_land",
    1: "agriculture_land",
    2: "rangeland",
    3: "forest_land",
    4: "water",
    5: "barren_land",
    6: "unknown"
}


def rgb_to_class(mask_rgb):
    """
    Converts RGB color-coded segmentation masks (like [0,255,255]) into class index maps (e.g., 0, 1, 2...).
    """
    color_to_class = {
        (0, 255, 255): 0,
        (255, 255, 0): 1,
        (255, 0, 255): 2,
        (0, 255, 0): 3,
        (0, 0, 255): 4,
        (255, 255, 255): 5,
        (0, 0, 0): 6
    }
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.uint8)
    for rgb, class_id in color_to_class.items():
        mask_class[(mask_rgb == rgb).all(axis=2)] = class_id
    return mask_class

class SegmentationDataset(Dataset):
    """
    Loads image/mask pairs for training.
    Used by PyTorch DataLoader for batching.
    """
    def __init__(self, root_dir, image_size=(224, 224), augment=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith("_sat.jpg")])

        if augment:
            self.img_transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=30),
                T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.02),
                T.GaussianBlur(kernel_size=5),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.mask_transform = T.Resize(image_size, interpolation=Image.NEAREST)

    def __len__(self):
        """
        Number of Training samples
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads and transforms a single image-mask pair.
        """
        img_name = self.files[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert model output to probabilities
        inputs = torch.softmax(inputs, dim=1) 

        # Convert targets to one-hot [B, C, H, W]
        one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        # Dice formula
        intersection = (inputs * one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Added focal loss for IoU improvement
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def class_pixel_coverage(mask, num_classes=7):
    """
    Calculates how much of the image belongs to each class, as a percentage.
    """
    total_pixels = mask.size
    coverage = {}
    for class_id in range(num_classes):
        class_pixels = np.sum(mask == class_id)
        coverage[class_id] = class_pixels / total_pixels
    return coverage


def calculate_iou(pred, true, num_classes=7):
    """
    Computes Intersection over Union for each class.
    Used to evaluate how well the model is segmenting each class.
    """
    ious = []
    for class_id in range(num_classes):
        intersection = np.logical_and(pred == class_id, true == class_id).sum()
        union = np.logical_or(pred == class_id, true == class_id).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


def pixel_accuracy(pred, true):
    """
    Calculates how many pixels the model predicted correctly.
    """
    correct = (pred == true).sum()
    total = pred.size
    return correct / total


def compute_class_weights(dataset, num_classes=7):
    """
    Function to calculate inverse frequency-based weights for each class 
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for _, mask in dataset:
        mask_np = mask.numpy()
        for i in range(num_classes):
            class_counts[i] += np.sum(mask_np == i)

    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts + 1e-6)
    class_weights[6] = 1e-3
    return torch.tensor(class_weights, dtype=torch.float32)


def train_segformer(train_dir, valid_dir, num_classes=7, image_size=(224,224), batch_size=4, epochs=10):
    """
    Loads data and model.
    Trains and validates the model.
    Logs the results into a CSV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = SegmentationDataset(train_dir, image_size=image_size, augment=True)
    val_dataset = SegmentationDataset(valid_dir, image_size=image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Load config and weights from safetensors
    config = SegformerConfig.from_json_file("classification_folder/models/b2-ade-512-512/config.json")
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "classification_folder/models/b2-ade-512-512",
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )
    
    ### USED FOR .safetensors file
    #state_dict = load_file("classification_folder/models/mit-b0/deepglobe.safetensors")
    #model = SegformerForSemanticSegmentation(config)
    #model.load_state_dict(state_dict)
    model.to(device)

    class_weights = compute_class_weights(train_dataset, num_classes=num_classes).to(device)
    #class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 0.1]).to(device)

    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, label_smoothing=0.1)
    dice_loss_func = DiceLoss()
    focal_loss_func = FocalLoss(alpha=1.0, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',        # We're maximizing validation mean IoU
        factor=0.5,        # Reduce LR by 50%
        patience=3,        # Wait 3 epochs of no improvement
        verbose=True
    )

    # CSV setup
    stat_folder = 'classification_folder/model_statistics/'
    with open(stat_folder + f"training_metrics_{MODEL_NAME}_{epochs}_epochs.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Epoch", "Pixel_Accuracy"] + [f"IoU_{id2label[i]}" for i in range(num_classes)] + ["Mean_IoU_wo_Unknown", "Loss"]
        writer.writerow(header)

        # Training loop
        best_iou = 0.0
        for epoch in trange(epochs, desc="Training Epoch"):
            model.train()
            total_loss = 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(pixel_values=images)
                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                ce_loss = criterion(upsampled_logits, masks)
                dice_loss = dice_loss_func(upsampled_logits, masks)
                focal_loss = focal_loss_func(upsampled_logits, masks)
                
                if epoch < 10:
                    loss = 0.5 * dice_loss + 0.5 * ce_loss
                else:
                    loss = 0.4 * dice_loss + 0.3 * ce_loss + 0.3 * focal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            all_ious = []
            all_accs = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(pixel_values=images)
                    logits = outputs.logits

                    upsampled_logits = torch.nn.functional.interpolate(
                        logits,
                        size=image_size,
                        mode="bilinear",
                        align_corners=False
                    )

                    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
                    true_mask = masks[0].cpu().numpy()

                    ious = calculate_iou(pred_mask, true_mask, num_classes)
                    acc = pixel_accuracy(pred_mask, true_mask)

                    all_ious.append(ious)
                    all_accs.append(acc)

                    # Pixel coverage printout
                    coverage = class_pixel_coverage(pred_mask, num_classes)
                    print("Class-wise Pixel Coverage:")
                    for i, percent in coverage.items():
                        label = id2label.get(i, f"Class {i}")
                        print(f"{label} (Class {i}): {percent * 100:.2f}%")

            mean_iou = np.nanmean(np.array(all_ious), axis=0)
            mean_acc = np.mean(all_accs)
            print(f"\nEpoch {epoch+1} - Validation Pixel Accuracy: {mean_acc * 100:.2f}%")
            for i, iou in enumerate(mean_iou):
                label = id2label.get(i, f"Class {i}")
                print(f"{label} (Class {i}) IoU: {iou:.4f}")

            valid_classes = [i for i in range(num_classes) if i != 6]
            val_mean_iou = np.nanmean([mean_iou[i] for i in valid_classes])
            print(f"Mean IoU : {val_mean_iou}")
            row = [epoch+1, mean_acc] + list(mean_iou) + [val_mean_iou, avg_loss]
            writer.writerow(row)
            scheduler.step(val_mean_iou)
            if val_mean_iou > best_iou:
                best_iou = val_mean_iou
                torch.save(model.state_dict(), f"best_model_{MODEL_NAME}.pth")
                print(f"\nNew best model saved at epoch {epoch+1} with Mean IoU: {val_mean_iou:.4f} and Validation Pixel Accuracy: {mean_acc * 100:.2f}%")

    # Save model
    torch.save(model.state_dict(), f"{MODEL_NAME}_ade_{epochs}_epochs.pth")
    print(f"Model saved as {MODEL_NAME}_ade_{epochs}_epochs.pth")


if __name__ == "__main__":
    train_segformer(
        train_dir="classification_folder/DeepGlobe_Converted_Dataset/train", 
        valid_dir="classification_folder/DeepGlobe_Converted_Dataset/valid",
        num_classes=7, 
        image_size=(256,256), 
        batch_size=4, 
        epochs=40
    )
