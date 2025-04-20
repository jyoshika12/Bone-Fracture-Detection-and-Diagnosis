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
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Title
st.markdown('<h1 class="title">MedAI Disease Detection & Diagnosis</h1>', unsafe_allow_html=True)

# File Upload Box
uploaded_file = st.file_uploader("Upload a scan or image", type=["png", "jpg", "jpeg"])

# Disease Selection (Dropdown)
disease_type = st.selectbox(
    "Select Disease Type",
    options=["Bone Fracture", "Eye Disease", "Monkeypox"],
    index=None,
    placeholder="Select Type",
)

# Show the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Diagnosis Button
if uploaded_file and disease_type:
    if st.button("Get Diagnosis"):
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        st.write("File Name:", file_name)
        st.write("File Size:", file_size)  # Debugging step

        # Convert image to bytes
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            st.error("Error: Uploaded file is empty!")
        else:
            st.write("Sending image to backend...")

        # Send image to backend
        files = {"file": (file_name, image_bytes, uploaded_file.type)}
        data = {"disease_type": disease_type}
        try:
            response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
            response.raise_for_status()
            result = response.json()

            # Decode the images from base64
            if "original_image" in result and "processed_image" in result:
                original_img = Image.open(BytesIO(base64.b64decode(result["original_image"])))
                processed_img = Image.open(BytesIO(base64.b64decode(result["processed_image"])))

                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(processed_img, caption="Processed Image (with Detections)", use_container_width=True)
            else:
                st.warning("Processed images are not available.")

            st.markdown("---")

            # Show detected diseases
            detected_diseases = result.get("detected_diseases", [])
            st.subheader("Detected Diseases")
            st.write(", ".join(detected_diseases))

            # AI Medical Analysis and Common Questions
            st.subheader("Medical Analysis:")
            medical_analysis = result.get("gemini_analysis", {})
            if medical_analysis:
                for disease in detected_diseases:
                    analysis_data = medical_analysis.get(disease, {})
                    diagnosis = analysis_data.get("diagnosis", "No specific diagnosis available.")

                    st.write(f"**{disease} Analysis:**")
                    st.write(diagnosis.strip())
                    # Display the model type
            model_type = result.get("model_type", "Model type information not available.")
            st.write(f"**Model Type:** {model_type}")

            # If no disease detected
            if not detected_diseases:
                st.warning("No disease detected. Please consult a doctor.")

            st.markdown("---")

        except requests.exceptions.RequestException as e:
            st.error(f"Prediction API error: {e}")
            st.stop()
