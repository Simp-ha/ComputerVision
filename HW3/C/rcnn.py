import os
import zipfile
for file in ['TrainIJCNN2013','TestIJCNN2013']:
  if os.path.exists(f'{file}.zip'):
    if os.path.isdir(file):
      print('Files already exist...')
    else:
      local_zip = f'{file}.zip'
      zip_ref = zipfile.ZipFile(local_zip, 'r')
      zip_ref.extractall('./')
      zip_ref.close()
      print("Unzip completed...")
  else:
    print(f'{file}.zip not found')
train_dir = 'TrainIJCNN2013'
test_dir = 'TestIJCNN2013'

import os
import torch
import torchvision.transforms as T
from PIL import Image


# Class name to integer mapping
CLASS_NAME_TO_ID = {"RBC": 0, "WBC": 1, "Platelets": 2, "Background": 3}


class BCCDDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.annotations = []

        # Load annotations
        with open(label_file, "r") as f:
            for line_no, line in enumerate(f.readlines()):
                if line_no == 0:  # Skip header row
                    continue
                # Extract columns
                fields = line.strip().split(",")
                if len(fields) != 8:  # Ensure all 8 columns are present
                    print(f"Skipping malformed line {line_no + 1}: {line.strip()}")
                    continue

                filename, width, height, class_name, xmin, ymin, xmax, ymax = fields
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

                # Skip invalid boxes where width or height is zero
                if xmin >= xmax or ymin >= ymax:
                    print(f"Skipping invalid bounding box at line {line_no + 1}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                    continue

                self.annotations.append({
                    "filename": filename,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "label": CLASS_NAME_TO_ID[class_name]
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation["filename"])
        img = Image.open(img_path).convert("RGB")

        # Convert bounding box and label to tensors
        boxes = torch.tensor([annotation["bbox"]], dtype=torch.float32)
        labels = torch.tensor([annotation["label"]], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # Apply transforms if provided
        if self.transforms:
            img = self.transforms(img)

        return img, target


# Define transforms
transforms = T.Compose([
    T.ToTensor(),
    T.Resize((416, 416))
])

# Load the dataset
train_dataset = BCCDDataset("BCCD/train", "BCCD/train/_annotations.csv", transforms=transforms)
val_dataset = BCCDDataset("BCCD/valid", "BCCD/valid/_annotations.csv", transforms=transforms)
test_dataset = BCCDDataset("BCCD/test", "BCCD/valid/_annotations.csv", transforms=transforms)

# Example: Access a sample
img, target = train_dataset[0]
print("Image shape:", img.shape)
print("Bounding boxes:", target["boxes"])
print("Labels:", target["labels"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier for BCCD dataset
num_classes = 4  # RBC, WBC, Platelets + 1 (background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torch
import torch.optim as optim
from torchvision.ops import box_iou
import time

def calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Calculate mAP for a batch of predictions and ground truths.
    """
    tp = 0
    fp = 0
    total_gt = len(gt_boxes)

    # Sort predictions by confidence (highest confidence first)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    matched_gt = set()

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        # Match with ground truth boxes
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou > iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / total_gt
    return precision, recall

def train_model_with_map(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    """
    Train a Faster R-CNN model with mAP calculation and verbose outputs.

    Args:
    - model: PyTorch Faster R-CNN model.
    - train_loader: DataLoader for training set.
    - val_loader: DataLoader for validation set.
    - device: 'cuda' or 'cpu'.
    - num_epochs: Number of epochs.
    - lr: Learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        model.train()
        epoch_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
          # if batch_idx>3:
            # continue
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward and compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}] - Average Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass to get predictions
                outputs = model(images)

                # Note: Skipping loss calculation in eval mode
                for i, output in enumerate(outputs):
                    print(f"Validation Batch - Image {i+1}: Predicted {len(output['boxes'])} boxes")

        print(f"Epoch [{epoch + 1}] - Validation Done.")

    print("\nTraining Complete!")
    return model

    # Initialize the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # RBC, WBC, Platelets + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
trained_model = train_model_with_map(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=1,
    lr=1e-4
)

# Save the trained model
torch.save(trained_model.state_dict(), "faster_rcnn_bccd_with_map.pth")

# Initialize the model architecture
num_classes = 4  # RBC, WBC, Platelets + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("faster_rcnn_bccd_with_map.pth"))
model.eval()  # Set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Map class IDs to class names
ID_TO_CLASS_NAME = {0: "Background", 1: "RBC", 2: "WBC", 3: "Platelets"}

def visualize_predictions(image_path, model, threshold=0.5):
    """
    Visualize predictions on a single image.

    Args:
    - image_path: Path to the input image.
    - model: Trained Faster R-CNN model.
    - threshold: Confidence threshold for displaying predictions.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    # Make predictions
    with torch.no_grad():
        prediction = model([img_tensor])

    # Extract boxes, labels, and scores
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()

    # Draw boxes on the original image
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            class_name = ID_TO_CLASS_NAME[label]

            # Draw rectangle and label
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

import os
import numpy as np
# Test images directory
test_images_dir = "BCCD/test"

# List all test images
test_images = os.listdir(test_images_dir)

# Visualize predictions for a few images
for i in range(5):  # Visualize first 5 images
    image_path = os.path.join(test_images_dir, test_images[i])
    print(f"Visualizing predictions for: {test_images[i]}")
    visualize_predictions(image_path, model, threshold=0.5)
