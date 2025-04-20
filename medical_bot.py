from google.generativeai import configure, GenerativeModel
import streamlit as st

def setup_medical_bot(api_key):
    """
    Configures Gemini AI with the provided API key and returns
    an instance of the Gemini GenerativeModel.
    """
    configure(api_key=api_key)
    return GenerativeModel("gemini-1.5-flash")

def run_medical_bot(medical_model):
    """
    Runs the MedicalBot interface.
    This function creates the MedicalBot UI components so that users can ask a question.
    """
    st.subheader("💬 AI MedicalBot")
    st.write("Ask me any medical-related question, and I'll respond like a doctor!")

    user_query = st.text_input("Enter your medical question:", key="medical_bot_input")

    if st.button("Get Medical Advice"):
        if user_query:
            try:
                response = medical_model.generate_content(user_query)
                answer = response.text if hasattr(response, "text") else "I'm sorry, I couldn't process your request."
                st.success(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")
