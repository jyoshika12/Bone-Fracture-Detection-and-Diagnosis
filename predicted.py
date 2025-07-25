from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLO model
model = YOLO('bonefracture.pt')  # Example: 'runs/detect/train/weights/best.pt'

# Path to the test image
image_path = 'bf6.jpg'  # Replace with your test image path

# Run inference
results = model(image_path)

# Show detections in a window (opens a window with bounding boxes)
results[0].show()

# Save the output image with bounding boxes
results[0].save(filename='predicted.jpg')

# Optional: Print each detected object
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = model.names[cls_id]
    print(f"Detected: {label} with confidence {conf:.2f}")

