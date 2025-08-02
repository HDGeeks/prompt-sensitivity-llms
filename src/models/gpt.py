import os
from openai import OpenAI
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

client = OpenAI(api_key=OPENAI_KEY)


def get_openai_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {e}"
