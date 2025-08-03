import os
from rich import print as rprint
from dotenv import load_dotenv
import requests
import warnings

warnings.filterwarnings("ignore", category=Warning, module="urllib3")

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
rprint(response.json())
