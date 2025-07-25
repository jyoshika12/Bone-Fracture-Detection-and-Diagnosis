import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file.")
    exit()

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize the Gemini model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")


    prompt = "What is a bone fracture? Explain in simple terms."
    response = model.generate_content(prompt)

    if hasattr(response, "text"):
        print("✅ Gemini API is working! Here's the response:\n")
        print(response.text.strip())
    else:
        print("⚠️ No text in the response. Raw response:")
        print(response)
except Exception as e:
    print("❌ Error calling Gemini API:", str(e))
