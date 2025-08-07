import csv, json, random, time
from datetime import datetime
from pathlib import Path
from models.gemini_client import query_gemini  # 👈 IMPORTANT

# === CONFIG ===
MODEL_NAME = "gemini"
# MODEL_VERSION = "Gemini-2.0-Flash"
MODEL_VERSION = "Gemini-1.5-Pro"
N_RUNS = 5
CONTROL_SUFFIX = " Respond in exactly 100 words. Be clear and concise."
MAX_TOKENS = 130
TEMPERATURE = 0.5
TOP_P = 0.9
OUTDIR = Path("src/outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)


# === UTILS ===
def load_promptset(path="src/data/prompts_v2.json"):
    with open(path, "r") as f:
        return json.load(f)


def build_variants(base_text, variants, paraphrase):
    return [
        ("base", base_text),
        ("paraphrase_neutral", paraphrase),
        ("tone", variants["tone"]),
        ("formality", variants["formality"]),
        ("emotion", variants["emotion"]),
    ]


def all_items(promptset):
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


def word_count(text):
    return len(text.split())


def estimate_tokens(text):
    return int(len(text) / 4)


# === RUN ===
def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = OUTDIR / f"responses_{MODEL_NAME}_{stamp}.csv"
    promptset = load_promptset()
    base_rows = all_items(promptset)

    header = [
        "timestamp",
        "run_id",
        "order_idx",
        "model",
        "model_version",
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
        "token_count_est",
        "response_id",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for run_id in range(1, N_RUNS + 1):
            seed = 1000 + run_id
            random.Random(seed).shuffle(base_rows)

            for order_idx, (domain, base_id, variant, raw_prompt) in enumerate(
                base_rows, start=1
            ):
                full_prompt = f"{raw_prompt}{CONTROL_SUFFIX}"
                text, err, latency_ms = query_gemini(  # 👈 Swapped here
                    full_prompt, MAX_TOKENS, TEMPERATURE, TOP_P
                )

                row = [
                    datetime.now().isoformat(timespec="seconds"),
                    run_id,
                    order_idx,
                    MODEL_NAME,
                    MODEL_VERSION,
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
                    estimate_tokens(text),
                    f"{MODEL_NAME}_{run_id}_{order_idx}_{variant}",
                ]
                writer.writerow(row)
                time.sleep(0.1)

    print(f"✅ {MODEL_NAME} done. Output: {out_csv}")


if __name__ == "__main__":
    main()
