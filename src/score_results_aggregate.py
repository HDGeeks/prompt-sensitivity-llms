
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from bert_score import score
import logging

# Keep console clean from transformer warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# === CONFIG ===
CSV_PATH = Path(
    "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_openai_gpt-4o-mini_5_20250807_1418.csv"
).expanduser()

MODEL_LABEL = "OpenAI GPT-4o Mini"

# Output folder for the generated LaTeX
OUTPUT_DIR = Path("src/results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD CSV ===
# Only load columns used in this script to reduce memory and avoid surprises
cols_needed = [
    "domain",
    "base_id",
    "variant",
    "run_id",
    "response",
    "word_len",
    "response_id",
]
df = pd.read_csv(CSV_PATH, usecols=cols_needed)


def sentiment(text):
    """
    Return TextBlob sentiment polarity in [-1, 1].
    Positive means more positive tone.
    """
    return TextBlob(text).sentiment.polarity


def metrics_for_base(base_group):
    """
    Compute metrics per variant for one base_id group.

    Reference:
      Concatenate all paraphrase_neutral responses across runs.
      This gives a stronger, more stable reference than a single sample.

    For each variant in {base, tone, formality, emotion}:
      - BERTScore F1 vs the neutral reference
      - Mean sentiment polarity
      - Mean word length
      - One short snippet for quick eyeballing
    """
    # Build the neutral reference text by joining all neutral responses
    ref_texts = base_group.loc[
        base_group["variant"] == "paraphrase_neutral", "response"
    ].tolist()
    if not ref_texts:
        # No neutral reference found for this base_id
        return {}

    ref_concat = " ".join(ref_texts)

    out = {}
    for var in ["base", "tone", "formality", "emotion"]:
        # Collect all responses for this variant across runs
        var_texts = base_group.loc[base_group["variant"] == var, "response"].tolist()
        if not var_texts:
            # This variant not present for this base_id
            continue

        # BERTScore vs the neutral reference
        # Use same reference for each hypothesis to compare consistently
        P, R, F1 = score(
            var_texts, [ref_concat] * len(var_texts), lang="en", verbose=False
        )
        bert_mean = F1.mean().item()

        # Average sentiment for this variant
        sent_mean = sum(sentiment(t) for t in var_texts) / len(var_texts)

        # Average word length for this variant
        lens = base_group.loc[base_group["variant"] == var, "word_len"].tolist()
        len_mean = sum(lens) / len(lens)

        # Short preview snippet to spot obvious drift by eye
        snippet = var_texts[0][:60].replace("\n", " ") + "..."

        out[var] = dict(bert=bert_mean, sent=sent_mean, wlen=len_mean, snippet=snippet)
    return out


def make_tables(df_model, model_label):
    """
    Build LaTeX tables per domain.

    Steps:
      1. For each domain, loop over base_id and compute per-base metrics.
      2. Aggregate per-base results into domain-level averages per variant.
      3. Render a LaTeX table with Variant, BERTScore, Sentiment, Word Len.
    """
    latex_all = []

    # Domain loop
    for dom in df_model["domain"].unique():
        dom_df = df_model[df_model["domain"] == dom]

        # Compute metrics per base_id inside this domain
        per_base = []
        for bid in dom_df["base_id"].unique():
            base_group = dom_df[dom_df["base_id"] == bid]
            m = metrics_for_base(base_group)
            if m:
                per_base.append(m)

        # Aggregate across base_ids into one row per variant
        rows = []
        for var in ["base", "tone", "formality", "emotion"]:
            vals = [b[var] for b in per_base if var in b]
            if not vals:
                # This variant missing in this domain
                continue
            bert_avg = sum(v["bert"] for v in vals) / len(vals)
            sent_avg = sum(v["sent"] for v in vals) / len(vals)
            wlen_avg = sum(v["wlen"] for v in vals) / len(vals)
            # Snippet omitted in table to keep columns tight
            rows.append((var, bert_avg, sent_avg, wlen_avg))

        # Build a compact LaTeX table for this domain
        t = []
        t.append("\\begin{table}[h]")
        t.append("\\centering")
        t.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        t.append(f"\\label{{tab:{model_label}_{dom.replace(' ','_')}}}")
        # 3 numeric columns. p{6cm} was planned for a snippet column, not used now
        t.append("\\begin{tabular}{lccc p{6cm}}")
        t.append("\\hline")
        t.append("Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len  \\\\")
        t.append("\\hline")
        for var, b, s, wl in rows:
            t.append(f"{var} & {b:.3f} & {s:.2f} & {wl:.1f}  \\\\")
        t.append("\\hline")
        t.append("\\end{tabular}")
        t.append("\\end{table}")
        latex_all.append("\n".join(t))

    # Return all domain tables concatenated
    return "\n\n".join(latex_all)


# Build all tables across domains for this model
merged = make_tables(df, MODEL_LABEL)

# Write a full LaTeX document so you can compile directly
out_path = OUTPUT_DIR / f"{MODEL_LABEL}_All_Domains_aggregated_rows.tex"
with open(out_path, "w") as f:
    f.write("\\documentclass[12pt,a4paper]{article}\n")
    f.write("\\usepackage[margin=2.5cm]{geometry}\n")
    f.write("\\usepackage{array}\n")
    f.write("\\usepackage{booktabs}\n")
    f.write(f"\\title{{{MODEL_LABEL} – Variant Sensitivity Across Domains}}\n")
    f.write("\\author{}\n\\date{}\n")
    f.write("\\begin{document}\n\\maketitle\n\n")
    f.write(merged)
    f.write("\n\\end{document}\n")

print(f"Saved {out_path}")




# file: score_results_aggregate.py
# """
# score_results_aggregate.py — Batch scorer for all cleared model CSVs.

# What this script does
# ---------------------
# • Scans src/outputs/cleared/ for all response CSVs (one per model).
# • For each CSV (i.e., per model), computes domain-aggregated metrics:
#     - Build a neutral reference per (domain, base_id) by concatenating all
#       "paraphrase_neutral" responses across runs.
#     - For each variant in {base, tone, formality, emotion}, compute:
#         * BERTScore F1 vs the neutral reference (default model: roberta-large)
#         * Mean TextBlob sentiment polarity (Δ in [-1, 1])
#         * Mean word length
#     - Aggregate these per-base_id metrics into a single domain row (average).
# • Emits a standalone LaTeX file per model with one compact table per domain.

# How this differs from score_results.py
# --------------------------------------
# • score_results.py operates on a **single CSV** (one model) and writes one LaTeX
#   file for that model.
# • THIS script **loops over all cleared CSVs** and writes one LaTeX file **per model**,
#   so you can regenerate the whole paper’s domain tables in one run.

# Methodological notes (kept consistent with the paper)
# -----------------------------------------------------
# • Reference text: concatenation of all "paraphrase_neutral" outputs per base_id,
#   which stabilizes the baseline across runs.
# • Aggregation: metrics are first computed at the base_id level, then averaged
#   to obtain domain-level values, preventing bias from unequal row counts.
# • Metric order is fixed: F1 (BERTScore), Δ (sentiment), Len (word length).
# • Domains are rendered in the paper’s order: PH, HE, PS, SC, EP.
# """

# import logging
# from pathlib import Path

# import pandas as pd
# from bert_score import score  # Default model is 'roberta-large'
# from textblob import TextBlob

# # Quiet transformer noise in console logs
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# # === CONFIG ===
# CLEARED_DIR = Path("src/outputs/cleared")
# RESULTS_DIR = Path("src/results_agg")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# # Columns required for scoring and grouping
# REQ_COLS = [
#     "domain",
#     "base_id",
#     "variant",
#     "run_id",
#     "response",
#     "word_len",
#     "response_id",
# ]

# # Domain display order (paper order)
# DOMAIN_ORDER = [
#     "Public Health",
#     "Historical Events",
#     "Political Systems",
#     "Scientific Consensus",
#     "Environmental Policy",
# ]


# def model_label_from_filename(p: Path) -> str:
#     """
#     Convert a cleared CSV filename into the human-readable model label
#     used throughout the paper.
#     Examples:
#       responses_openai_gpt-4o-mini_5_20250807_1418.csv
#         -> OpenAI GPT-4o-Mini
#       responses_openai_gpt-4o_5_20250807_1620.csv
#         -> OpenAI GPT-4o
#       responses_openai_gpt-3.5-turbo_... -> OpenAI GPT-3.5 Turbo
#       responses_gemini_Gemini-1.5-Pro_... -> Gemini-1.5-Pro
#       responses_gemini_Gemini-2.0-Flash_... -> Gemini-2.0-Flash
#       responses_llama_LLaMA-3.1-8B-Instruct_... -> LLaMA-3.1-8B-Instruct
#       responses_mistralai_Mistral-7B-Instruct-v0.3_... -> Mistral-7B-Instruct-v0.3
#     """
#     stem = p.stem
#     # Example stems:
#     # responses_openai_gpt-4o-mini_5_20250807_1418
#     # responses_gemini_Gemini-1.5-Pro_5_20250808_1114
#     # responses_llama_LLaMA-3.1-8B-Instruct_5_20250808_1256
#     # responses_mistralai_Mistral-7B-Instruct-v0.3_5_20250808_1317
#     parts = stem.split("_")
#     if len(parts) < 3:
#         return stem

#     vendor = parts[1]
#     model_raw = parts[2]

#     # Normalize known vendors
#     if vendor == "openai":
#         # Map common OpenAI slugs to paper labels
#         mapping = {
#             "gpt-4o-mini": "OpenAI GPT-4o-Mini",
#             "gpt-4o": "OpenAI GPT-4o",
#             "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
#         }
#         return mapping.get(model_raw, f"OpenAI {model_raw}")

#     if vendor == "gemini":
#         # Already in good paper form
#         return model_raw

#     if vendor == "llama":
#         return model_raw.replace("_", " ")

#     if vendor == "mistralai":
#         return model_raw

#     # Fallback
#     return model_raw


# def compute_sentiment(text: str) -> float:
#     """Return TextBlob sentiment polarity in [-1, 1]."""
#     return TextBlob(text).sentiment.polarity


# def metrics_for_base(base_df: pd.DataFrame) -> dict:
#     """
#     Compute per-variant metrics for a single base_id group.
#     Reference = concatenated 'paraphrase_neutral' responses.
#     Returns:
#       {variant: {"bert": float, "sent": float, "wlen": float}}
#     """
#     ref_texts = base_df.loc[base_df["variant"] == "paraphrase_neutral", "response"].tolist()
#     if not ref_texts:
#         return {}

#     ref_concat = " ".join(ref_texts)
#     out = {}

#     for var in ["base", "tone", "formality", "emotion"]:
#         var_texts = base_df.loc[base_df["variant"] == var, "response"].tolist()
#         if not var_texts:
#             continue

#         # BERTScore F1 vs the same neutral reference
#         P, R, F1 = score(var_texts, [ref_concat] * len(var_texts), lang="en", verbose=False)
#         bert_mean = F1.mean().item()

#         sent_mean = sum(compute_sentiment(t) for t in var_texts) / len(var_texts)
#         wlens = base_df.loc[base_df["variant"] == var, "word_len"].tolist()
#         wlen_mean = sum(wlens) / len(wlens)

#         out[var] = {"bert": bert_mean, "sent": sent_mean, "wlen": wlen_mean}

#     return out


# def build_domain_tables(df: pd.DataFrame, model_label: str) -> str:
#     """
#     For a single model's DataFrame:
#       • Compute per-base_id metrics, then average to domain-level per variant.
#       • Render one LaTeX table per domain.
#     Returns:
#       A string with all LaTeX tables concatenated.
#     """
#     latex_all = []

#     # Ensure deterministic domain ordering
#     domains_present = [d for d in DOMAIN_ORDER if d in df["domain"].unique()]
#     # Append any unexpected domains (if any)
#     for d in df["domain"].unique():
#         if d not in domains_present:
#             domains_present.append(d)

#     for dom in domains_present:
#         dom_df = df[df["domain"] == dom]
#         if dom_df.empty:
#             continue

#         per_base = []
#         for bid in dom_df["base_id"].unique():
#             base_group = dom_df[dom_df["base_id"] == bid]
#             m = metrics_for_base(base_group)
#             if m:
#                 per_base.append(m)

#         rows = []
#         for var in ["base", "tone", "formality", "emotion"]:
#             vals = [m[var] for m in per_base if var in m]
#             if not vals:
#                 continue
#             bert_avg = sum(v["bert"] for v in vals) / len(vals)
#             sent_avg = sum(v["sent"] for v in vals) / len(vals)
#             wlen_avg = sum(v["wlen"] for v in vals) / len(vals)
#             rows.append((var, bert_avg, sent_avg, wlen_avg))

#         # Render domain table
#         t = []
#         t.append("\\begin{table}[h]")
#         t.append("\\centering")
#         t.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
#         safe_label = f"{model_label}_{dom}".replace(" ", "_").replace("—", "-")
#         t.append(f"\\label{{tab:{safe_label}}}")
#         # Keep the column structure aligned with the paper
#         t.append("\\begin{tabular}{lccc p{6cm}}")
#         t.append("\\hline")
#         t.append("Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len  \\\\")
#         t.append("\\hline")
#         for var, b, s, wl in rows:
#             t.append(f"{var} & {b:.3f} & {s:.2f} & {wl:.1f}  \\\\")
#         t.append("\\hline")
#         t.append("\\end{tabular}")
#         t.append("\\end{table}")
#         latex_all.append("\n".join(t))

#     return "\n\n".join(latex_all)


# def process_one_csv(csv_path: Path) -> None:
#     """Load one cleared CSV, compute tables, and write a standalone LaTeX file."""
#     df = pd.read_csv(csv_path, usecols=REQ_COLS)
#     model_label = model_label_from_filename(csv_path)

#     merged_tables = build_domain_tables(df, model_label)

#     out_path = RESULTS_DIR / f"{model_label}_All_Domains_aggregated_rows.tex"
#     with open(out_path, "w") as f:
#         f.write("\\documentclass[12pt,a4paper]{article}\n")
#         f.write("\\usepackage[margin=2.5cm]{geometry}\n")
#         f.write("\\usepackage{array}\n")
#         f.write("\\usepackage{booktabs}\n")
#         f.write(f"\\title{{{model_label} – Variant Sensitivity Across Domains}}\n")
#         f.write("\\author{}\n\\date{}\n")
#         f.write("\\begin{document}\n\\maketitle\n\n")
#         f.write(merged_tables)
#         f.write("\n\\end{document}\n")

#     print(f"✅ Saved: {out_path}")


# def main():
#     csvs = sorted(CLEARED_DIR.glob("responses_*.csv"))
#     if not csvs:
#         print("No CSVs found in src/outputs/cleared/")
#         return
#     print(f"Found {len(csvs)} cleared CSV(s). Processing…\n")
#     for p in csvs:
#         print(f"-> {p.name}")
#         process_one_csv(p)
#     print("\nAll models processed. Outputs in: src/results_agg/")


# if __name__ == "__main__":
#     main()