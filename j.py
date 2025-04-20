from ultralytics import YOLO

model = YOLO("bonefracture.pt")  # Replace with your model file
results = model("frac.jpg", conf=0.3, save=True)  # Replace with an actual image

for result in results:
    print(result.boxes)  # Check bounding boxes
    print(result.names)   # Check detected class names
