import os
from rich import print as rprint
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face Inference API endpoint for LLaMA 3
URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

data = {"inputs": "Ping", "parameters": {"max_length": 10}}


def ping_huggingface_llama3() -> dict:
    try:
        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        return {"status": "success", "response": result[0]["generated_text"].strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = ping_huggingface_llama3()
    rprint(result)
