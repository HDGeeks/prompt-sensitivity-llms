import os, csv, json, random, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from models.openai_client import query_openai
from models.gemini_client import query_gemini
from models.llama_client import query_llama

load_dotenv()

CONTROL_SUFFIX = " Respond clearly and concisely in ~80–120 words."
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9
N_RUNS = 5

OUTDIR = Path("src/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def word_count(s: str) -> int:
    return len(s.split())


def build_variants(base_text: str, variants: dict, paraphrase: str):
    """
    Returns (variant_name, prompt_text) tuples.
    Includes: base, paraphrase_neutral, tone, formality, emotion.
    """
    return [
        ("base", base_text),
        ("paraphrase_neutral", paraphrase),
        ("tone", variants["tone"]),
        ("formality", variants["formality"]),
        ("emotion", variants["emotion"]),
    ]


def load_promptset(pth="src/data/prompts_v2.json"):
    with open(pth, "r") as f:
        return json.load(f)


def all_items(promptset):
    """
    Expand to a flat list of (domain, base_id, variant_name, raw_prompt)
    """
    rows = []
    for dom in promptset:
        domain = dom["domain"]
        variants = dom["variants"]
        for item in dom["items"]:
            base_id = item["base_id"]
            base = item["base"]
            para = item["paraphrase_neutral"]
            for vname, vtext in build_variants(base, variants, para):
                rows.append((domain, base_id, vname, vtext))
    return rows


def query_model(model_name: str, prompt: str):
    if model_name == "openai":
        return query_openai(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
    if model_name == "gemini":
        return query_gemini(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
    if model_name == "llama":
        return query_llama(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
    raise ValueError(f"Unknown model {model_name}")


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    promptset = load_promptset()
    base_rows = all_items(promptset)  # ~ 5 domains * 3 bases * 5 variants = 75 prompts

    models = ["openai", "gemini", "llama"]
    # One combined CSV for all runs (easiest for stats)
    out_csv = OUTDIR / f"responses_all_{stamp}.csv"
    header = [
        "timestamp",
        "run_id",
        "order_idx",
        "model",
        "domain",
        "base_id",
        "variant",
        "prompt",
        "full_prompt",
        "response",
        "err",
        "latency_ms",
        "char_len",
        "word_len",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for run_id in range(1, N_RUNS + 1):
            # randomize order per run with a fixed but different seed
            seed = 1000 + run_id
            random.Random(seed).shuffle(base_rows)

            for order_idx, (domain, base_id, variant, raw_prompt) in enumerate(
                base_rows, start=1
            ):
                full_prompt = f"{raw_prompt}{CONTROL_SUFFIX}"

                for model_name in models:
                    text, err, latency_ms = query_model(model_name, full_prompt)
                    row = [
                        datetime.now().isoformat(timespec="seconds"),
                        run_id,
                        order_idx,
                        model_name,
                        domain,
                        base_id,
                        variant,
                        raw_prompt,
                        full_prompt,
                        text,
                        err,
                        latency_ms,
                        len(text),
                        word_count(text),
                    ]
                    w.writerow(row)
                    # small polite delay to avoid rate spikes
                    time.sleep(0.1)

    print(f"✅ Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()
