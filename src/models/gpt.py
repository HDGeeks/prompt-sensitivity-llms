import os
from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

# === Load environment ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

client = OpenAI(api_key=OPENAI_KEY)


def get_openai_response(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        # Format response to match Gemini's JSON structure for consistency
        result = {
            "response": response.choices[0].message.content.strip(),
            "model": OPENAI_MODEL,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        return result
    except Exception as e:
        return {"error": f"[OpenAI Error] {e}"}


if __name__ == "__main__":
    prompt = "Explain how AI works in a few words"
    response = get_openai_response(prompt)
    rprint(response)
