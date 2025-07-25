import os
import cv2
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model_gemini =genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load only Bone Fracture model
bone_model = YOLO("bonefracture.pt")

# Optional: Label map for display
LABEL_MAP = {
    "transverse": "Transverse Fracture",
    "oblique": "Oblique Fracture",
    "spiral": "Spiral Fracture",
    "comminuted": "Comminuted Fracture",
    "greenstick": "Greenstick Fracture",
    "buckle": "Buckle Fracture",
    "avulsion": "Avulsion Fracture",
    "segmental": "Segmental Fracture",
    "impacted": "Impacted Fracture",
    "normal": "Normal"  # If included in training
}

# Disease detection function
def detect_bone_fracture(image):
    results = bone_model(image)
    detected = []
    image_np = np.array(image)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = map(int, box)
            label = names[int(class_id)]
            readable = LABEL_MAP.get(label, label.capitalize() + " Fracture")
            detected.append(readable)

            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image_np, readable, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    processed = Image.fromarray(image_np)
    return detected, processed

# Gemini analysis
def analyze_with_gemini(fracture_type):
    query = (
        f"A patient X-ray shows a suspected {fracture_type}. "
        "Give a detailed medical explanation including: 1. Fracture nature, 2. Common causes, "
        "3. Severity, 4. Bone parts affected, 5. Recommended imaging/tests, "
        "6. Emergency care, and 7. Treatment and recovery plan."
    )

    response = model_gemini.generate_content(query)
    return response.text.strip() if hasattr(response, "text") else "No response from Gemini."

# Encode image to Base64
def encode_image(image):
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode("utf-8")

# API Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("[DEBUG] /predict endpoint called")

    image_bytes = await file.read()
    print(f"[DEBUG] Received file: {file.filename}, size: {len(image_bytes)} bytes")

    if not image_bytes:
        print("[ERROR] Received empty image file.")
        return {"error": "Empty image file received."}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("[DEBUG] Image opened and converted to RGB")
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {str(e)}")
        return {"error": f"Image decoding failed: {str(e)}"}

    try:
        detected_diseases, processed_image = detect_bone_fracture(image)
        print(f"[DEBUG] Detected fractures: {detected_diseases}")
    except Exception as e:
        print(f"[ERROR] YOLO detection failed: {str(e)}")
        return {"error": f"Detection failed: {str(e)}"}

    if not detected_diseases or "Normal" in detected_diseases:
        print("[DEBUG] No specific fracture detected")
        detected_diseases = ["No specific fracture detected"]

    gemini_results = {}
    for label in detected_diseases:
        if label == "No specific fracture detected":
            diagnosis = "No abnormalities found in the bone scan."
        else:
            print(f"[DEBUG] Calling Gemini for: {label}")
            try:
                diagnosis = analyze_with_gemini(label)
            except Exception as e:
                print(f"[ERROR] Gemini API failed: {str(e)}")
                diagnosis = "Gemini API failed to provide analysis."

        gemini_results[label] = {"diagnosis": diagnosis}
        print(f"[DEBUG] Diagnosis for {label}: {diagnosis[:100]}...")  # show first 100 chars

    try:
        encoded_original = encode_image(image)
        encoded_processed = encode_image(processed_image)
        print("[DEBUG] Images encoded successfully")
    except Exception as e:
        print(f"[ERROR] Image encoding failed: {str(e)}")
        return {"error": f"Encoding failed: {str(e)}"}

    response_payload = {
        "detected_diseases": detected_diseases,
        "original_image": encoded_original,
        "processed_image": encoded_processed,
        "gemini_analysis": gemini_results,
        "model_type": "Bone Fracture"
    }

    print("[DEBUG] Sending response back to frontend")
    return response_payload

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
