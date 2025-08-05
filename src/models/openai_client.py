import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

MODEL_ID = "gpt-4o-mini"  # change if needed


def query_openai(prompt: str, max_tokens=150, temperature=0.7, top_p=0.9):
    t0 = time.time()
    try:
        resp = client.responses.create(
            model=MODEL_ID,
            input=prompt,
        )
        text = resp.output_text.strip()
        err = ""
    except Exception as e:
        text, err = "", f"{e}"
    latency_ms = int((time.time() - t0) * 1000)
    return text, err, latency_ms
