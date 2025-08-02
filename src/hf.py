import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B"

client = InferenceClient(model=LLAMA_MODEL, token=HF_TOKEN)


def get_hf_response(prompt):
    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["\n\n", "###", "User:"],
        )
        return response.strip()
    except Exception as e:
        return f"[HuggingFace Error] {e}"
