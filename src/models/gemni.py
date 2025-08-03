import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import requests

# === Load environment ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# Set your API key as an environment variable
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": API_KEY,
}

data = {"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]}

response = requests.post(URL, headers=headers, json=data)
print(response.json())
