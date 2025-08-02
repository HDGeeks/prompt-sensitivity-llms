# src/test_keys.py

import os
import requests
from dotenv import load_dotenv

# Load from .env
load_dotenv("src/.env")


# === OpenAI API Test ===
def test_openai():
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _ = client.models.list()
        return "✅ OpenAI key valid"
    except Exception as e:
        return f"❌ OpenAI key error: {str(e)}"


# === Gemini API Test (cURL-style POST)
def test_gemini():
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]
        }
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 200:
            return "✅ Gemini key valid"
        else:
            return f"❌ Gemini key error: {response.status_code} — {response.text}"
    except Exception as e:
        return f"❌ Gemini key error: {str(e)}"


# === HuggingFace Token Test ===
def test_huggingface():
    try:
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        headers = {"Authorization": f"Bearer {hf_token}"}
        r = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        if r.status_code == 200:
            return "✅ HuggingFace token valid"
        else:
            return f"❌ HuggingFace token error: {r.status_code} — {r.text}"
    except Exception as e:
        return f"❌ HuggingFace token error: {str(e)}"


# === Run All Tests
if __name__ == "__main__":
    print(test_openai())
    print(test_gemini())
    print(test_huggingface())
