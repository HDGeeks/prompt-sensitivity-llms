import os
from openai import OpenAI
from dotenv import load_dotenv
from rich import print as rprint

# Load environment
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key=OPENAI_KEY,
)

response = client.responses.create(
    model="gpt-4o-mini",
    input="write a haiku about ai",
    store=True,
)

rprint(response.output_text)
