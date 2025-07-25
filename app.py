import os
import streamlit as st
import requests
import base64
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Load Custom CSS
def load_css():
    if os.path.exists("styles.css"):
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Title
st.markdown('<h1 class="title">MedAI - Bone Fracture Detection & Diagnosis</h1>', unsafe_allow_html=True)

# File Upload Box
uploaded_file = st.file_uploader("Upload an X-ray scan", type=["png", "jpg", "jpeg"])

# Show the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Diagnosis Button
if uploaded_file:
    if st.button("Detect Fracture"):
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        st.write("File Name:", file_name)
        st.write("File Size:", file_size)

        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            st.error("Error: Uploaded file is empty!")
        else:
            st.write("Analyzing the image...")

        # Send image to backend (no need for disease_type anymore)
        files = {"file": (file_name, image_bytes, uploaded_file.type)}

        try:
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            response.raise_for_status()
            result = response.json()

            # Decode and show images
            if "original_image" in result and "processed_image" in result:
                original_img = Image.open(BytesIO(base64.b64decode(result["original_image"])))
                processed_img = Image.open(BytesIO(base64.b64decode(result["processed_image"])))

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(processed_img, caption="Processed Image (Detected Fractures)", use_container_width=True)
            else:
                st.warning("Processed images are not available.")

            st.markdown("---")

            # Show detected fracture types
            detected_diseases = result.get("detected_diseases", [])
            st.subheader("Detected Fracture Types")
            st.write(", ".join(detected_diseases))

            # Show Gemini medical analysis
            st.subheader("Medical Analysis:")
            medical_analysis = result.get("gemini_analysis", {})
            for disease in detected_diseases:
                analysis_data = medical_analysis.get(disease, {})
                diagnosis = analysis_data.get("diagnosis", "No specific diagnosis available.")

                st.write(f"**{disease} Analysis:**")
                st.write(diagnosis.strip())

            # Show model type
            model_type = result.get("model_type", "Model type information not available.")
            st.write(f"**Model Type:** {model_type}")

            if not detected_diseases or detected_diseases == ["No specific fracture detected"]:
                st.warning("No specific fracture detected. Please consult a radiologist.")

            st.markdown("---")

        except requests.exceptions.RequestException as e:
            st.error(f"Prediction API error: {e}")
            st.stop()
