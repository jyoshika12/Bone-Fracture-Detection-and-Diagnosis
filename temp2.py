from ultralytics import YOLO
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load YOLO model
model = YOLO('bonefracture.pt')

# Ground truth labels
ground_truth = {
    'bf2.jpg': 'fracture',
    'bf3.jpeg': 'fracture',
    'bf4.png': 'fracture',
    'bf5.jpg': 'fracture',
    'bf6.jpg': 'messed_up_angle',
    'bf7.jpg': 'angle',
    'bfn.jpg': 'line',
    'bfn1.jpg': 'fracture'
}

# All known labels (including no_detection)
all_labels = sorted(set(ground_truth.values()) | {"no_detection"})

# Storage
y_true = []
y_pred = []

# Create folder for per-image confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

# Process each image
for filename, true_label in ground_truth.items():
    print(f"\n🖼️ Processing: {filename}")
    image_path = filename

    # YOLO inference
    results = model(image_path)
    result = results[0]

    # Get top predicted class if any box exists
    if result.boxes and len(result.boxes.cls) > 0:
        cls_id = int(result.boxes.cls[0])
        pred_label = model.names[cls_id]
    else:
        pred_label = "no_detection"

    # Append results
    y_true.append(true_label)
    y_pred.append(pred_label)

    # Show in terminal
    print(f"✅ True Label     : {true_label}")
    print(f"🔍 Predicted Label: {pred_label}")

    # Create per-image confusion matrix
    cm = confusion_matrix([true_label], [pred_label], labels=all_labels)

    # Plot
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap='Oranges')
    plt.title(f"Confusion Matrix - {filename}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/{filename}_confusion.png")
    plt.close()

# === Overall Evaluation ===
print("\n📊 === Overall Classification Report ===")
print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

# Overall confusion matrix
overall_cm = confusion_matrix(y_true, y_pred, labels=all_labels)

# Plot overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(overall_cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap='Blues')
plt.title("Overall Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("overall_confusion_matrix.png")
plt.show()