import os
import cv2
import torch
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form,Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import google.generativeai as genai  # Gemini Import
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Initialize Gemini 1.5
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-pro")

app = FastAPI()
# CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 models
models = {
    "Bone Fracture": YOLO("bonefracture.pt"),
    "Eye Disease": YOLO("eyecancer.pt"),
    "Monkeypox": YOLO("monkeypox.pt"),
}

# Function to detect disease using YOLO
def detect_disease(image, selected_model):
    model = models[selected_model]  # Use the correct YOLO model
    if not model:
        return None, None
    results = model(image)

    detected_diseases = []
    image_np = np.array(image)  # Convert to NumPy for processing

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        names = result.names  # Class names

        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = map(int, box)
            disease_label = names[int(class_id)]  # Get label from class ID
            detected_diseases.append(disease_label)

            # Draw bounding boxes
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image_np, disease_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    processed_image = Image.fromarray(image_np)
    return detected_diseases, processed_image

# Function to analyze disease with Gemini 1.5
# Analyze with Gemini (returns only diagnosis)
def analyze_with_gemini(disease_name, model_type):
    query = (
        f"The patient scan reveals a suspected case of {disease_name} in the category of {model_type}. "
        "Provide a thorough medical analysis including: "
        "1. Condition summary, 2. Severity, 3. Likely cause, "
        "4. Affected anatomical areas, 5. Recommended diagnostic tests (if needed), "
        "6. Immediate care steps, and 7. Long-term treatment options."
    )

    response = model_gemini.generate_content(query)
    diagnosis = response.text.strip() if response and hasattr(response, "text") else "No response from Gemini."

    print(f"[Gemini Diagnosis] {diagnosis}")  # Optional: for backend logs

    return diagnosis

# Encode image to Base64
def encode_image(image):
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode("utf-8")

# API Endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    disease_type: str = Form(...)
):
    image_bytes = await file.read()
    if not image_bytes:
        return {"error": "Received empty image file"}

    # Save received file for debugging
    with open("received_image.jpg", "wb") as f:
        f.write(image_bytes)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image: {str(e)}"}

    # Detect disease using YOLO
    detected_diseases, processed_image = detect_disease(image, disease_type)
    if not detected_diseases or "normal" in detected_diseases:
        detected_diseases = ["No specific disease detected"]

    if detected_diseases is None or processed_image is None:
        return {"error": "Disease detection failed"}

    # Get Gemini diagnosis
    gemini_results = {}
    for disease in detected_diseases:
        diagnosis = analyze_with_gemini(disease, disease_type)
        gemini_results[disease] = {"diagnosis": diagnosis}

    # Custom message for no disease
    if "No specific disease detected" in detected_diseases:
        gemini_results["No disease detected"] = {
            "diagnosis": "No abnormalities were found in the scan."
        }

    return {
        "detected_diseases": detected_diseases,
        "original_image": encode_image(image),
        "processed_image": encode_image(processed_image),
        "gemini_analysis": gemini_results,
        "model_type": disease_type
    }

# Chatbot Endpoint
# New Chatbot Endpoint for interactive Q&A
# @app.post("/ask_gemini")
# async def ask_gemini(request: Request):
#     data = await request.json()
#     question = data.get("question", "")
#
#     if not question:
#         return {"answer": "No question provided."}
#
#     response = model_gemini.generate_content(question)
#
#     answer = response.text if response and hasattr(response, "text") else "No response from Gemini."
#     return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
