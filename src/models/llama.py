import os
from rich import print as rprint
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face Inference API endpoint for LLaMA 3.1 (text-only model)
URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

data = {"inputs": "Ping", "parameters": {"max_length": 20, "return_full_text": False}}


def ping_huggingface_llama3() -> dict:
    try:
        if not HF_API_KEY:
            return {
                "status": "error",
                "message": "HUGGING_FACE_API_KEY not set in .env",
            }

        response = requests.post(URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()

        if isinstance(result, list) and result:
            return {
                "status": "success",
                "response": result[0]["generated_text"].strip(),
            }
        else:
            return {
                "status": "error",
                "message": f"Unexpected response format: {result}",
            }

    except requests.exceptions.HTTPError as e:
        return {
            "status": "error",
            "message": f"HTTP Error: {e.response.status_code} - {e.response.text}",
        }
    except Exception as e:
        return {"status": "error", "message": f"General Error: {str(e)}"}


if __name__ == "__main__":
    result = ping_huggingface_llama3()
    rprint(result)
