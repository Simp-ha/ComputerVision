import os
import zipfile
import csv
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

from PIL import Image
import pandas as pd
from ultralytics import YOLO
import shutil
if os.path.isdir(f'./{train_dir}/images') or not os.path.isdir(f'./{test_dir}/images'):  
    print("Converting Dataset...")
    def convert_to_yolo_format(txt_file, image_dir, output_dir):
        """
        Converts dataset annotations from .txt files to YOLO format and saves them in the specified output directory.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        yolo_annotations = []
        with open(txt_file, 'r') as file:
            for line in file:
                parts = line.strip().split(';')
                filename = parts[0]
                image_filename = os.path.splitext(filename)[0] + ".ppm"  
                img_path = os.path.join(image_dir, image_filename)
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(txt_file, 'r') as f:    
                    # Prepare YOLO annotations
                    for line in f:
                        # Each line in the txt file has:  xmin ymin xmax ymax class_id
                        parts = line.strip().split(';')
                        x_min, y_min, x_max, y_max, class_id = map(int, parts[1:])  
                        # Normalize to YOLO format
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # Append the YOLO formatted annotation
                        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Save YOLO annotations in the output directory
                output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
                print(os.path.splitext(filename)[0])
                with open(output_file, "w") as f:
                    f.write("\n".join(yolo_annotations))

    # Convert datasets to YOLO format
    convert_to_yolo_format("TrainIJCNN2013/gt.txt", "TrainIJCNN2013/images", "TrainIJCNN2013/labels")
    convert_to_yolo_format("TestIJCNN2013/gt.txt", "TestIJCNN2013/images", "TestIJCNN2013/labels")
    CLASS_NAME_TO_ID = {    '0':'speed limit 20 (prohibitory)'              ,                  
                            '1':'speed limit 30 (prohibitory)'              ,                  
                            '2':'speed limit 50 (prohibitory)'              ,              
                            '3':'speed limit 60 (prohibitory)'              ,              
                            '4':'speed limit 70 (prohibitory)'              ,              
                            '5':'speed limit 80 (prohibitory)'              ,              
                            '6':'restriction ends 80 (other)'               ,              
                            '7':'speed limit 100 (prohibitory)'             ,
                            '8':'speed limit 120 (prohibitory)'             ,
                            '9':'no overtaking (prohibitory)'               ,
                            '10':'no overtaking (trucks) (prohibitory)'     ,
                            '11':'priority at next intersection (danger)'   ,
                            '12':'priority road (other)'                    ,
                            '13':'give way (other)'                         ,
                            '14':'stop (other)'                             ,
                            '15':'no traffic both ways (prohibitory)'       ,
                            '16':'no trucks (prohibitory)'                  ,
                            '17':'no entry (other)'                         ,
                            '18':'danger (danger)'                          ,
                            '19':'bend left (danger)'                       ,
                            '20':'bend right (danger)'                      ,
                            '22':'bend (danger)'                            ,
                            '21':'uneven road (danger)'                     ,
                            '23':'slippery road (danger)'                   ,
                            '24':'road narrows (danger)'                    ,
                            '25':'construction (danger)'                    ,
                            '26':'traffic signal (danger)'                  ,
                            '27':'pedestrian crossing (danger)'             ,
                            '28':'school crossing (danger)'                 ,
                            '29':'cycles crossing (danger)'                 ,
                            '30':'snow (danger)'                            ,
                            '31':'animals (danger)'                         ,
                            '32':'restriction ends (other)'                 ,
                            '33':'go right (mandatory)'                     ,
                            '34':'go left (mandatory)'                      ,
                            '35':'go straight (mandatory)'                  ,
                            '36':'go right or straight (mandatory)'         ,
                            '37':'go left or straight (mandatory)'          ,
                            '38':'keep right (mandatory)'                   ,
                            '39':'keep left (mandatory)'                    ,
                            '40':'roundabout (mandatory)'                   ,
                            '41' :'restriction ends (overtaking) (other)'   ,
                            '42':'restriction ends (overtaking (trucks)) (other)}'}           
    
    def fix_yolo_labels(label_dir):
        """
        Convert class names to numeric IDs in YOLO label files.
        Args:
        - label_dir: Directory containing YOLO label files.
        """
        for label_file in os.listdir(label_dir):
            if label_file.endswith(".txt"):
                label_path = os.path.join(label_dir, label_file)
                fixed_labels = []

                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        class_name = parts[0]
                        bbox = parts[1:]

                        # Convert class name to numeric ID
                        if class_name in CLASS_NAME_TO_ID:
                            class_id = CLASS_NAME_TO_ID[class_name]
                            fixed_labels.append(f"{class_id} {' '.join(bbox)}")
                        else:
                            print(f"Skipping unknown class '{class_name}' in {label_file}")

                # Overwrite the label file with fixed labels
                with open(label_path, "w") as f:
                    f.write("\n".join(fixed_labels))

    # Fix labels for train, val, and test sets
    fix_yolo_labels("TrainIJCNN2013/labels")
    # fix_yolo_labels("BCCD/valid/labels")
    fix_yolo_labels("TestIJCNN2013/labels")


    def organize_images(image_dir, destination_dir):
        """

        Move all images to the correct 'images/' subdirectory.
        Args:
        - image_dir: Directory containing the images.
        - destination_dir: Target 'images/' directory.
        """
        os.makedirs(destination_dir, exist_ok=True)

        for file in os.listdir(image_dir):
            if file.endswith((".jpg", ".png", ".jpeg", ".ppm")):  # Include valid image formats
                source_path = os.path.join(image_dir, file)
                destination_path = os.path.join(destination_dir, file)
                shutil.move(source_path, destination_path)

    # Organize images into 'images/' subfolders
    organize_images("TrainIJCNN2013", "TrainIJCNN2013/images")
    # organize_images("BCCD/valid", "BCCD/valid/images")
    organize_images("TestIJCNN2013", "TestIJCNN2013/images")
else:
    print("Dataset ready")

# Load YOLOv3 model
model = YOLO("yolov3.yaml")  # Specify the YOLOv3 architecture

# Train the model
model.train(data="data.yaml", epochs=10, imgsz=416, batch=32, device=0)

# Evaluate the model
metrics = model.val()
print(metrics)

# Save the model
model.export(format="torchscript")  # Save trained weights

import cv2
import matplotlib.pyplot as plt


def visualize_yolo_detections(image_path, model, confidence_threshold=0.5):
    """
    Visualize YOLO predictions on an image.

    Args:
    - image_path: Path to the input image.
    - model: YOLO model object.
    - confidence_threshold: Minimum confidence for displaying detections.
    """
    # Perform inference
    results = model(image_path)  # Run YOLO on the input image

    # Convert image for visualization
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process detections
    detections = results[0].boxes  # Access the boxes property
    for det in detections:
        # Extract bounding box, confidence, and class
        box = det.xyxy.cpu().numpy()[0]  # xyxy format
        conf = det.conf.cpu().numpy()[0]  # Confidence score
        cls = int(det.cls.cpu().numpy()[0])  # Class index

        if conf < confidence_threshold:
            continue

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.title("YOLO Detections")
    plt.show()

    # Load the trained YOLO model
model = YOLO("../../runs/detect/train5/weights/best.pt")  # Load the trained YOLOv3 model

# Test on an image from the dataset
image_path = "./TestIJCNN2013Download/00131.ppm"  # Replace with an actual test image path
visualize_yolo_detections(image_path, model, confidence_threshold=0.5)
