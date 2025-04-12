from ultralytics import YOLO

# Load the YOLOv3 model (e.g., pretrained weights)
model = YOLO("yolov3.pt")  # You can use YOLOv3 weights

# Run inference on an image
results = model("path_to_image.jpg")

# Display results
results.show()  # Show the image with detections
