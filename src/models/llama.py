import os
from rich import print as rprint
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_KEY not set in .env")

# Create client
client = InferenceClient(api_key=HF_TOKEN)

# Run basic ping test
prompt = "You are a helpful AI assistant.\nAnswer this:\nWhat is 2 + 2?"
response = client.text_generation(
    prompt,
    model="meta-llama/Llama-3.1-8B",
)

rprint(f"[bold green]LLaMA response:[/] {response}")
