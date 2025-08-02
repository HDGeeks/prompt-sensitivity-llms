import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# === Load environment ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Placeholder

# === Global Settings ===
CONTROL_SUFFIX = " Respond clearly and concisely in no more than 3 sentences."
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

# === Model IDs ===
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "models/gemini-1.5-pro-latest"

# === Output Setup ===
Path("src/outputs").mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_file = f"src/outputs/responses_{timestamp}.json"

# === Load Prompt Data ===
with open("src/data/prompts.json", "r") as f:
    prompts = json.load(f)

# === Initialize Clients ===
openai_client = OpenAI(api_key=OPENAI_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# === OpenAI Completion ===
def get_openai_response(prompt: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {e}"


# === Gemini Completion ===
def get_gemini_response(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_TOKENS,
            },
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"


# === HuggingFace Placeholder ===
def get_hf_response(prompt: str) -> str:
    return "[Not implemented]"


# === Run Prompt Batch ===
all_results = []

for entry in prompts:
    domain = entry["domain"]
    for variant in ["base", "tone", "formality", "emotion"]:
        raw_prompt = entry[variant]
        full_prompt = raw_prompt + CONTROL_SUFFIX

        result = {
            "domain": domain,
            "variant": variant,
            "prompt": raw_prompt,
            "full_prompt": full_prompt,
            "responses": {
                "openai": get_openai_response(full_prompt),
                "gemini": get_gemini_response(full_prompt),
                "huggingface": get_hf_response(full_prompt),
            },
        }

        all_results.append(result)

# === Write to File ===
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Responses saved to {output_file}")
