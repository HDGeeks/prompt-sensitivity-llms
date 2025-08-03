import os
from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

# Load environment
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)


def ping_openai() -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=10,
        )
        return {
            "status": "success",
            "response": response.choices[0].message.content.strip(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = ping_openai()
    rprint(result)
