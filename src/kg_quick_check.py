import os, csv, json, random, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from models.openai_client import query_openai
from models.gemini_client import query_gemini
from models.llama_client import query_llama

# -----------------------------
# config
# -----------------------------
SEED = 7
TEST_CSV = Path("data/kg_test.csv")         # cols: head,relation,tail,allowed_tails (comma-separated)
ALIASES_CSV = Path("data/aliases.csv")      # optional: cols: variant,canonical
RESULTS_RAW = Path("results/kg_quick.jsonl")
RESULTS_HARD = Path("results/kg_quick_hardened.jsonl")

IN_CONTEXT_EXAMPLES = 6                      # few-shot examples per prompt
LEV_SIM_THRESHOLD = 0.85                     # for fuzzy match
JSON_ENFORCE = True

MODELS = {
    "chatgpt": lambda prompt: query_openai(prompt, temperature=0.0, max_tokens=64),
    "gemini":  lambda prompt: query_gemini(prompt, temperature=0.0, max_tokens=64),
    "llama":   lambda prompt: query_llama(prompt, temperature=0.0, max_tokens=64, model="meta-llama/Llama-3.1-8B-Instruct"),
}

# -----------------------------
# utils
# -----------------------------
def load_aliases(path):
    m = {}
    if not path.exists():
        return m
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["variant"].strip().lower()] = row["canonical"].strip()
    return m

def norm_text(s):
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())

def alias_map(s, aliases):
    key = s.lower().strip()
    return aliases.get(key, s)

def exact_match(pred, gold, aliases):
    p = alias_map(pred, aliases)
    g = alias_map(gold, aliases)
    return 1 if norm_text(p) == norm_text(g) else 0

def fuzzy_sim(a, b):
    # simple normalized LCS as a cheap proxy to avoid extra deps in the quick check
    # optional: swap with rapidfuzz if you prefer
    def lcs(x, y):
        dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
        for i in range(1, len(x)+1):
            for j in range(1, len(y)+1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    x, y = norm_text(a), norm_text(b)
    if not x or not y:
        return 0.0
    return lcs(x, y) / max(len(x), len(y))

def parse_json_tail(raw):
    # try parsing strict JSON first, then relax with a crude fallback
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "tail" in obj:
            return str(obj["tail"])
    except Exception:
        pass
    # fallback: look for "tail": "..."
    import re
    m = re.search(r'"tail"\s*:\s*"([^"]+)"', raw)
    return m.group(1) if m else raw.strip()

def build_prompt(item, in_context, allowed_list, hardened=False):
    relation = item["relation"]
    ex_lines = []
    for ex in in_context:
        ex_lines.append(json.dumps({
            "head": ex["head"],
            "relation": ex["relation"],
            "tail": ex["tail"]
        }, ensure_ascii=False))
    rules = [
        "Output a single JSON object with keys head, relation, tail.",
        "Do not add explanations."
    ]
    if JSON_ENFORCE:
        rules.append("If unsure, output an empty JSON object {}.")
    if hardened:
        rules.append("Use only a tail from the Allowed list. Any other value is invalid.")

    prompt = f"""You are completing a knowledge graph.

Relation: {relation}
Allowed tail entities: {", ".join(allowed_list)}

Examples:
{chr(10).join(ex_lines)}

Predict the missing tail entity for:
Head: {item["head"]}
Relation: {relation}

Rules:
- {chr(10).join("- " + r for r in rules)}

Output:
"""
    return prompt

def read_items(csv_path):
    items = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            allowed = [t.strip() for t in row["allowed_tails"].split(",") if t.strip()]
            items.append({
                "head": row["head"].strip(),
                "relation": row["relation"].strip(),
                "tail": row["tail"].strip(),
                "allowed": allowed
            })
    return items

def sample_in_context(all_items, k, exclude_head):
    pool = [x for x in all_items if x["head"] != exclude_head]
    random.shuffle(pool)
    return pool[:k]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def summarize(rows, aliases):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    summary = {}
    for m, rs in by_model.items():
        ems = [exact_match(rr["pred_tail"], rr["gold_tail"], aliases) for rr in rs]
        em = sum(ems) / len(ems) if rs else 0.0
        # closed-world violation
        cwv = []
        for rr in rs:
            pred_norm = alias_map(rr["pred_tail"], aliases)
            allowed_norm = set(norm_text(a) for a in rr["allowed"])
            cwv.append(0 if norm_text(pred_norm) in allowed_norm else 1)
        viol = sum(cwv) / len(cwv) if rs else 0.0
        summary[m] = {
            "n": len(rs),
            "em": round(em, 4),
            "violation_rate": round(viol, 4),
            "all_correct": em == 1.0 and viol == 0.0
        }
    return summary

def harden_allowed_list(item):
    # make allowed list trickier: add lookalikes and country cities from other heads
    allowed = list(item["allowed"])
    # add strings that can trick format and aliasing
    extras = []
    if item["tail"] not in allowed:
        allowed.append(item["tail"])
    # add decorated variants of correct tail
    extras.append(f"{item['tail']} City")
    extras.append(f"The city of {item['tail']}")
    # add two random other tails as distractors
    # note: we cannot access global list here cleanly, so caller will pass extras in
    return allowed, extras

# -----------------------------
# main
# -----------------------------
def run_once(items, aliases, out_path, hardened=False, global_tails=None):
    random.seed(SEED)
    rows = []
    for idx, item in enumerate(items):
        in_context = sample_in_context(items, IN_CONTEXT_EXAMPLES, exclude_head=item["head"])

        allowed = list(item["allowed"])
        if hardened:
            # add two plausible distractors sampled from global tails with same initial letter if possible
            distractors = []
            if global_tails:
                same_initial = [t for t in global_tails if t[0].lower() == item["tail"][0].lower() and t != item["tail"]]
                pool = same_initial or [t for t in global_tails if t != item["tail"]]
                random.shuffle(pool)
                distractors = pool[:2]
            # add lookalike strings
            allowed = list(dict.fromkeys(allowed + distractors + [f"{item['tail']} City", f"The city of {item['tail']}"]))

        prompt = build_prompt(item, in_context, allowed, hardened=hardened)

        for model_name, client in MODELS.items():
            t0 = time.time()
            raw = client(prompt)
            latency_ms = int((time.time() - t0) * 1000)

            pred_tail = parse_json_tail(raw) if JSON_ENFORCE else raw.strip()

            row = {
                "ts": datetime.utcnow().isoformat(),
                "idx": idx,
                "model": model_name,
                "prompt": prompt,
                "raw": raw,
                "pred_tail": pred_tail,
                "gold_tail": item["tail"],
                "head": item["head"],
                "relation": item["relation"],
                "allowed": allowed,
                "latency_ms": latency_ms,
                "hardened": hardened,
            }
            rows.append(row)

    write_jsonl(out_path, rows)
    summary = summarize(rows, aliases)
    return rows, summary

if __name__ == "__main__":
    load_dotenv()
    assert TEST_CSV.exists(), f"Missing {TEST_CSV}"
    items = read_items(TEST_CSV)
    aliases = load_aliases(ALIASES_CSV)

    # gather global tail universe for hardening step
    global_tails = sorted({x["tail"] for x in items})

    # pass 1: quick baseline
    rows1, sum1 = run_once(items, aliases, RESULTS_RAW, hardened=False, global_tails=global_tails)
    print("\n=== QUICK CHECK SUMMARY ===")
    for m, s in sum1.items():
        print(f"{m:10s} | n={s['n']:3d} | EM={s['em']:.3f} | Viol={s['violation_rate']:.3f} | all_correct={s['all_correct']}")

    any_perfect = any(s["all_correct"] for s in sum1.values())
    if not any_perfect:
        print("\nNo model is perfect. You are good to proceed with this dataset.")
    else:
        print("\nPerfect detected for at least one model. Running hardened pass...")

        rows2, sum2 = run_once(items, aliases, RESULTS_HARD, hardened=True, global_tails=global_tails)
        print("\n=== HARDENED CHECK SUMMARY ===")
        for m, s in sum2.items():
            print(f"{m:10s} | n={s['n']:3d} | EM={s['em']:.3f} | Viol={s['violation_rate']:.3f} | all_correct={s['all_correct']}")

        print("\nFiles:")
        print(f"- baseline jsonl: {RESULTS_RAW}")
        print(f"- hardened jsonl: {RESULTS_HARD}")