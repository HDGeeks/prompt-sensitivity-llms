# file: score_results_all.py
"""
Process ALL cleared CSVs and emit one LaTeX tables doc per file.

Method (matches paper):
- For each domain & base_id, build neutral reference by concatenating ALL
  paraphrase_neutral responses across runs.
- For each variant in {base, tone, formality, emotion}, compute:
    • BERTScore F1 vs that neutral reference (DEFAULT MODEL: roberta-large)
    • Mean TextBlob sentiment polarity (Δ proxy)
    • Mean word length
- Aggregate rows per domain into compact LaTeX tables.

Outputs:
- src/results/<ModelLabel>_All_Domains.tex (one per CSV)
"""
from pathlib import Path
import logging
import pandas as pd
from textblob import TextBlob
from bert_score import score  # default backbone: 'roberta-large'

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

CLEARED_DIR = Path("src/outputs/cleared")
RESULTS_DIR = Path("src/results_test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COLS = [
    "domain",
    "base_id",
    "variant",
    "run_id",
    "response",
    "word_len",
    "response_id",
    # optional, for nice labels if present:
    "model",
    "model_version",
]

# Pretty labels from filename fragments (fallback if cols missing)
PRETTY_MAP = {
    "gpt-4o-mini": "OpenAI GPT-4o-Mini",
    "gpt-4o": "OpenAI GPT-4o",
    "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
    "Gemini-1.5-Pro": "Gemini-1.5-Pro",
    "Gemini-2.0-Flash": "Gemini-2.0-Flash",
    "LLaMA-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
}

def model_label_from(df: pd.DataFrame, csv_name: str) -> str:
    if "model_version" in df.columns and df["model_version"].notna().any():
        # Use the most frequent version string if present
        mv = df["model_version"].dropna().mode()
        if len(mv):
            return mv.iloc[0]
    # Fallback: infer from filename
    for key, pretty in PRETTY_MAP.items():
        if key in csv_name:
            return pretty
    # Last resort
    return csv_name.replace(".csv", "")

def polarity(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def make_tables_for_model(df_model: pd.DataFrame, model_label: str) -> str:
    domains = list(df_model["domain"].dropna().unique())
    all_tables = []

    for dom in domains:
        dom_df = df_model[df_model["domain"] == dom]
        table_rows = []

        for base_id in dom_df["base_id"].dropna().unique():
            base_group = dom_df[dom_df["base_id"] == base_id]

            # Build neutral reference (concat of ALL paraphrase_neutral responses)
            ref_texts = base_group.loc[
                base_group["variant"] == "paraphrase_neutral", "response"
            ].dropna().tolist()
            if not ref_texts:
                continue
            ref_concat = " ".join(ref_texts)

            for var in ["base", "tone", "formality", "emotion"]:
                var_texts = base_group.loc[
                    base_group["variant"] == var, "response"
                ].dropna().tolist()
                if not var_texts:
                    continue

                # BERTScore F1 vs constant reference
                _, _, F1 = score(var_texts, [ref_concat] * len(var_texts),
                                 lang="en", verbose=False)  # roberta-large default
                f1_mean = F1.mean().item()

                sent_mean = sum(polarity(t) for t in var_texts) / len(var_texts)

                lens = base_group.loc[
                    base_group["variant"] == var, "word_len"
                ].dropna().tolist()
                wlen_mean = sum(lens) / len(lens) if lens else float("nan")

                table_rows.append((var, f1_mean, sent_mean, wlen_mean))

        # Render LaTeX for this domain (skip if empty)
        if not table_rows:
            continue

        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        safe_label = model_label.replace(" ", "_").replace(".", "").replace("–", "-")
        lines.append(f"\\label{{tab:{safe_label}_{dom.replace(' ', '_')}}}")
        lines.append("\\begin{tabular}{lccc p{6cm}}")
        lines.append("\\toprule")
        lines.append("Variant & BERTScore F1 (vs. neutral) & Sentiment Polarity & Word Len \\\\")
        lines.append("\\midrule")
        for v, f1m, sm, wl in table_rows:
            lines.append(f"{v} & {f1m:.3f} & {sm:.2f} & {wl:.1f} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        all_tables.append("\n".join(lines))

    return "\n\n".join(all_tables)

def process_csv(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=lambda c: c in COLS, keep_default_na=True)
    label = model_label_from(df, csv_path.name)
    doc = make_tables_for_model(df, label)
    if not doc.strip():
        print(f"⚠️ No tables generated for {csv_path.name} (skipping).")
        return

    out_path = RESULTS_DIR / f"{label.replace(' ', '_').replace('.', '')}_All_Domains.tex"
    with open(out_path, "w") as f:
        f.write("\\documentclass[12pt,a4paper]{article}\n")
        f.write("\\usepackage[margin=2.5cm]{geometry}\n")
        f.write("\\usepackage{array}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{microtype}\n")
        f.write(f"\\title{{{label} – Variant Sensitivity Across Domains}}\n")
        f.write("\\author{}\n\\date{}\n")
        f.write("\\begin{document}\n\\maketitle\n\n")
        f.write(doc)
        f.write("\n\\end{document}\n")
    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    files = sorted(CLEARED_DIR.glob("responses_*.csv"))
    if not files:
        print("No CSVs found in src/outputs/cleared/")
    for p in files:
        print(f"Processing {p.name} ...")
        process_csv(p)
    print("Done.")

# """
# This script processes a CSV file containing LLM-generated responses to prompts with different variants (e.g., base, tone, formality, emotion) across multiple domains. 

# Input:
# - A CSV file with columns: domain, base_id, variant, run_id, response, word_len, response_id.
# - The CSV is expected to be at the path specified by CSV_PATH.

# Operation:
# - For each domain and each base prompt (base_id), it constructs a neutral reference by concatenating all "paraphrase_neutral" responses.
# - For each variant (base, tone, formality, emotion), it computes:
#     - BERTScore F1 (semantic similarity) against the neutral reference.
#     - Mean sentiment polarity (using TextBlob).
#     - Mean word length.
# - Results are aggregated into LaTeX tables per domain.

# Output:
# - A standalone LaTeX file containing tables summarizing the above metrics for each domain and variant, saved to OUTPUT_DIR.
# """
# import pandas as pd
# from bert_score import score # Default model is 'roberta-large'
# from textblob import TextBlob
# from pathlib import Path
# import logging

# # Quiet transformer noise in console logs
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# # === CONFIG ===
# CSV_PATH = Path(
#     "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_openai_gpt-4o-mini_5_20250807_1418.csv"
# ).expanduser()
# MODEL_LABEL = "OpenAI GPT-4o Mini"

# # Where to save the LaTeX file
# OUTPUT_DIR = Path("src/results/")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # === LOAD CSV ===
# # Only the columns we need for scoring and grouping
# cols_needed = [
#     "domain",
#     "base_id",
#     "variant",
#     "run_id",
#     "response",
#     "word_len",
#     "response_id",
# ]
# df = pd.read_csv(CSV_PATH, usecols=cols_needed)


# # === FUNCTIONS ===
# def compute_sentiment(text):
#     """
#     Return TextBlob sentiment polarity in [-1, 1].
#     Positive values mean more positive tone.
#     """
#     return TextBlob(text).sentiment.polarity


# def make_tables_for_model(df_model, model_label):
#     """
#     Build LaTeX tables per domain for one model.

#     For each domain:
#       • For each base_id, create a neutral reference by concatenating all
#         paraphrase_neutral responses across runs.
#       • For each target variant, compute
#           - BERTScore F1 vs the neutral reference
#           - Mean sentiment polarity
#           - Mean word length
#       • Collect rows into a domain table.
#     """
#     domains = df_model["domain"].unique()
#     latex_all = []

#     for dom in domains:
#         dom_df = df_model[df_model["domain"] == dom]
#         table_rows = []

#         # Walk each base_id within this domain
#         for base_id in dom_df["base_id"].unique():
#             base_group = dom_df[dom_df["base_id"] == base_id]

#             # Build the neutral reference by joining all neutral responses
#             base_texts = base_group[base_group["variant"] == "paraphrase_neutral"][
#                 "response"
#             ].tolist()
#             if not base_texts:
#                 # Skip if no neutral reference for this base_id
#                 continue
#             base_concat = " ".join(base_texts)

#             # Score each target variant against the neutral reference
#             for var in ["base", "tone", "formality", "emotion"]:
#                 var_texts = base_group[base_group["variant"] == var][
#                     "response"
#                 ].tolist()
#                 if not var_texts:
#                     # Variant missing for this base_id
#                     continue

#                 # BERTScore F1 vs the same reference for all responses
#                 P, R, F1 = score(
#                     var_texts, [base_concat] * len(var_texts), lang="en", verbose=False
#                 )
#                 bert_mean = F1.mean().item()

#                 # Sentiment mean across responses of this variant
#                 sent_mean = sum(compute_sentiment(t) for t in var_texts) / len(
#                     var_texts
#                 )

#                 # Word length mean for this variant
#                 lengths = base_group[base_group["variant"] == var]["word_len"].tolist()
#                 len_mean = sum(lengths) / len(lengths)

#                 # Add one row for this base_id and variant
#                 table_rows.append((var, bert_mean, sent_mean, len_mean))

#         # Build LaTeX table for this domain
#         latex = []
#         latex.append("\\begin{table}[h]")
#         latex.append("\\centering")
#         latex.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
#         latex.append(f"\\label{{tab:{model_label}_{dom.replace(' ', '_')}}}")
#         # Kept p{6cm} slot in case you later add a snippet column
#         latex.append("\\begin{tabular}{lccc p{6cm}}")
#         latex.append("\\hline")
#         latex.append(
#             "Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len  \\\\"
#         )
#         latex.append("\\hline")

#         for row in table_rows:
#             latex.append(f"{row[0]} & {row[1]:.3f} & {row[2]:.2f} & {row[3]:.1f}  \\\\")

#         latex.append("\\hline")
#         latex.append("\\end{tabular}")
#         latex.append("\\end{table}")
#         latex_all.append("\n".join(latex))

#     # Return all domain tables joined together
#     return "\n\n".join(latex_all)


# # === RUN ===
# merged_tables = make_tables_for_model(df, MODEL_LABEL)

# # Write a standalone LaTeX file so you can compile immediately
# out_path = OUTPUT_DIR / f"{MODEL_LABEL}_All_Domains.tex"
# with open(out_path, "w") as f:
#     f.write("\\documentclass[12pt,a4paper]{article}\n")
#     f.write("\\usepackage[margin=2.5cm]{geometry}\n")
#     f.write("\\usepackage{array}\n")
#     f.write("\\usepackage{booktabs}\n")
#     f.write(f"\\title{{{MODEL_LABEL} – Variant Sensitivity Across Domains}}\n")
#     f.write("\\author{}\n\\date{}\n")
#     f.write("\\begin{document}\n\\maketitle\n\n")
#     f.write(merged_tables)
#     f.write("\n\\end{document}\n")

# print(f"✅ Full LaTeX file saved: {out_path}")


