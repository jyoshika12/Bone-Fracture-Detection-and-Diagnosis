# Bone Fracture Detection and Diagnosis

A web application that detects types of bone fractures in X-ray images using a custom-trained YOLOv8 model and provides detailed diagnosis using Gemini Pro (Google GenAI).

---

## Features

- Detects various bone fracture types (e.g., transverse, oblique, spiral)
- Highlights fracture areas in uploaded X-ray images
- Generates detailed medical analysis using Gemini AI
- Built with FastAPI backend and Streamlit frontend

---

## Tech Stack

- **Language:** Python
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **ML Model:** YOLOv8 (Ultralytics)
- **AI Integration:** Gemini API (Google GenAI)
- **Tools:** OpenCV, Pillow, Roboflow

---

## Setup Instructions

1. **Clone the repository**
git clone https://github.com/jyoshika12/Bone-Fracture-Detection-and-Diagnosis.git
cd Bone-Fracture-Detection-and-Diagnosis

2.Install dependencies
pip install -r requirements.txt

3.Create a .env file
GEMINI_API_KEY=your_api_key_here

4.Run the backend
uvicorn app:app --reload

5.Run the frontend
streamlit run streamlit_app.py


Note:
1..env file is ignored for security. Use .env.example as a reference.
2.Not intended for medical use without validation.
