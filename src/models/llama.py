import os
from rich import print as rprint
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACE_API_KEY environment variable not set. Please add it to your .env file."
    )


client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

result = client.text_generation(
    "Can you please let us know more details about your ",
    model="meta-llama/Llama-3.1-8B",
)
rprint(result)
