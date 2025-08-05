import os
import csv
import json
import random
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests
from models.openai_client import query_openai
from models.gemini_client import query_gemini
from models.llama_client import query_llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("src/outputs/query_log.txt"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Constants
CONTROL_SUFFIX = " Respond clearly and concisely in ~80–120 words."
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9
N_RUNS = 3  # Reduced to 3 runs for 675 queries
RETRIES = 3
RETRY_DELAY = 5  # Seconds
QUERY_DELAY = 1  # Seconds between queries
OUTDIR = Path("src/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


def word_count(s: str) -> int:
    """Count words in a string, handling None/empty inputs."""
    return len(s.split()) if s else 0


def validate_promptset(promptset):
    """Validate prompts_v2.json structure."""
    required_keys = ["domain", "items"]
    item_keys = ["base_id", "base", "paraphrase_neutral"]
    variant_keys = ["tone", "formality", "emotion"]

    for dom in promptset:
        for key in required_keys:
            if key not in dom:
                raise ValueError(f"Missing key {key} in promptset domain")
        for item in dom["items"]:
            for key in item_keys:
                if key not in item:
                    raise ValueError(f"Missing key {key} in item {item.get('base_id')}")
        for key in variant_keys:
            if key not in dom["variants"]:
                raise ValueError(f"Missing variant key {key} in domain {dom['domain']}")
    logger.info("Promptset validated successfully")


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
    """Load and validate promptset."""
    try:
        with open(pth, "r") as f:
            promptset = json.load(f)
        validate_promptset(promptset)
        return promptset
    except Exception as e:
        logger.error(f"Failed to load promptset: {e}")
        raise


def all_items(promptset):
    """Expand to a flat list of (domain, base_id, variant_name, raw_prompt)."""
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
    logger.info(f"Generated {len(rows)} prompts")
    return rows


def query_model(model_name: str, prompt: str):
    """Query a model with retries for transient errors."""
    for attempt in range(RETRIES):
        try:
            if model_name == "openai":
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY not set")
                text, err, latency_ms = query_openai(
                    prompt, MAX_TOKENS, TEMPERATURE, TOP_P
                )
            elif model_name == "gemini":
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY not set")
                text, err, latency_ms = query_gemini(
                    prompt, MAX_TOKENS, TEMPERATURE, TOP_P
                )
            elif model_name == "llama":
                if not HUGGING_FACE_API_KEY:
                    raise ValueError("HUGGING_FACE_API_KEY not set")
                text, err, latency_ms = query_llama(
                    prompt, MAX_TOKENS, TEMPERATURE, TOP_P
                )
            else:
                raise ValueError(f"Unknown model {model_name}")

            if err and any(code in err for code in ["402", "429", "503"]):
                logger.warning(
                    f"Error {err} for {model_name}, attempt {attempt+1}/{RETRIES}"
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return (
                text if text else "",
                err if err else "",
                latency_ms if latency_ms else 0,
            )
        except (requests.exceptions.RequestException, ConnectionError) as e:
            logger.warning(
                f"Network error for {model_name}, attempt {attempt+1}/{RETRIES}: {e}"
            )
            time.sleep(RETRY_DELAY * (attempt + 1))
            continue
    logger.error(f"Failed to query {model_name} after {RETRIES} attempts")
    return "", f"Failed after {RETRIES} retries", 0


def main():
    """Main function to query models and save responses."""
    # Validate API keys
    for key, name in [
        (OPENAI_API_KEY, "OPENAI_API_KEY"),
        (GEMINI_API_KEY, "GEMINI_API_KEY"),
        (HUGGING_FACE_API_KEY, "HUGGING_FACE_API_KEY"),
    ]:
        if not key:
            logger.error(f"{name} not set in .env")
            raise ValueError(f"{name} not set")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    promptset = load_promptset()
    base_rows = all_items(promptset)  # 5 domains * 3 bases * 5 variants = 75 prompts

    models = ["openai", "gemini", "llama"]
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
        f.flush()  # Ensure header is written

        total_queries = len(base_rows) * len(models) * N_RUNS
        query_count = 0

        for run_id in range(1, N_RUNS + 1):
            seed = 1000 + run_id
            random.Random(seed).shuffle(base_rows)

            for order_idx, (domain, base_id, variant, raw_prompt) in enumerate(
                base_rows, start=1
            ):
                full_prompt = f"{raw_prompt}{CONTROL_SUFFIX}"

                for model_name in models:
                    query_count += 1
                    logger.info(
                        f"Query {query_count}/{total_queries}: {model_name}, run {run_id}, {domain}, {base_id}, {variant}"
                    )
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
                    f.flush()  # Write immediately to avoid memory issues
                    time.sleep(QUERY_DELAY)  # Avoid rate limits

    logger.info(f"Completed {query_count} queries. Wrote {out_csv}")
    print(f"✅ Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()

# import os, csv, json, random, time
# from pathlib import Path
# from datetime import datetime
# from dotenv import load_dotenv

# from models.openai_client import query_openai
# from models.gemini_client import query_gemini
# from models.llama_client import query_llama

# load_dotenv()

# CONTROL_SUFFIX = " Respond clearly and concisely in ~80–120 words."
# MAX_TOKENS = 150
# TEMPERATURE = 0.7
# TOP_P = 0.9
# N_RUNS = 5

# OUTDIR = Path("src/outputs")
# OUTDIR.mkdir(parents=True, exist_ok=True)


# def word_count(s: str) -> int:
#     return len(s.split())


# def build_variants(base_text: str, variants: dict, paraphrase: str):
#     """
#     Returns (variant_name, prompt_text) tuples.
#     Includes: base, paraphrase_neutral, tone, formality, emotion.
#     """
#     return [
#         ("base", base_text),
#         ("paraphrase_neutral", paraphrase),
#         ("tone", variants["tone"]),
#         ("formality", variants["formality"]),
#         ("emotion", variants["emotion"]),
#     ]


# def load_promptset(pth="src/data/prompts_v2.json"):
#     with open(pth, "r") as f:
#         return json.load(f)


# def all_items(promptset):
#     """
#     Expand to a flat list of (domain, base_id, variant_name, raw_prompt)
#     """
#     rows = []
#     for dom in promptset:
#         domain = dom["domain"]
#         variants = dom["variants"]
#         for item in dom["items"]:
#             base_id = item["base_id"]
#             base = item["base"]
#             para = item["paraphrase_neutral"]
#             for vname, vtext in build_variants(base, variants, para):
#                 rows.append((domain, base_id, vname, vtext))
#     return rows


# def query_model(model_name: str, prompt: str):
#     if model_name == "openai":
#         return query_openai(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
#     if model_name == "gemini":
#         return query_gemini(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
#     if model_name == "llama":
#         return query_llama(prompt, MAX_TOKENS, TEMPERATURE, TOP_P)
#     raise ValueError(f"Unknown model {model_name}")


# def main():
#     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     promptset = load_promptset()
#     base_rows = all_items(promptset)  # ~ 5 domains * 3 bases * 5 variants = 75 prompts

#     models = ["openai", "gemini", "llama"]
#     # One combined CSV for all runs (easiest for stats)
#     out_csv = OUTDIR / f"responses_all_{stamp}.csv"
#     header = [
#         "timestamp",
#         "run_id",
#         "order_idx",
#         "model",
#         "domain",
#         "base_id",
#         "variant",
#         "prompt",
#         "full_prompt",
#         "response",
#         "err",
#         "latency_ms",
#         "char_len",
#         "word_len",
#     ]

#     with open(out_csv, "w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(header)

#         for run_id in range(1, N_RUNS + 1):
#             # randomize order per run with a fixed but different seed
#             seed = 1000 + run_id
#             random.Random(seed).shuffle(base_rows)

#             for order_idx, (domain, base_id, variant, raw_prompt) in enumerate(
#                 base_rows, start=1
#             ):
#                 full_prompt = f"{raw_prompt}{CONTROL_SUFFIX}"

#                 for model_name in models:
#                     text, err, latency_ms = query_model(model_name, full_prompt)
#                     row = [
#                         datetime.now().isoformat(timespec="seconds"),
#                         run_id,
#                         order_idx,
#                         model_name,
#                         domain,
#                         base_id,
#                         variant,
#                         raw_prompt,
#                         full_prompt,
#                         text,
#                         err,
#                         latency_ms,
#                         len(text),
#                         word_count(text),
#                     ]
#                     w.writerow(row)
#                     # small polite delay to avoid rate spikes
#                     time.sleep(0.1)

#     print(f"✅ Done. Wrote {out_csv}")


# if __name__ == "__main__":
#     main()
