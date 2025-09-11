"""
This script processes a CSV file containing LLM-generated responses to prompts with different variants (e.g., base, tone, formality, emotion) across multiple domains. 

Input:
- A CSV file with columns: domain, base_id, variant, run_id, response, word_len, response_id.
- The CSV is expected to be at the path specified by CSV_PATH.

Operation:
- For each domain and each base prompt (base_id), it constructs a neutral reference by concatenating all "paraphrase_neutral" responses.
- For each variant (base, tone, formality, emotion), it computes:
    - BERTScore F1 (semantic similarity) against the neutral reference.
    - Mean sentiment polarity (using TextBlob).
    - Mean word length.
- Results are aggregated into LaTeX tables per domain.

Output:
- A standalone LaTeX file containing tables summarizing the above metrics for each domain and variant, saved to OUTPUT_DIR.
"""
import pandas as pd
from bert_score import score
from textblob import TextBlob
from pathlib import Path
import logging

# Quiet transformer noise in console logs
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# === CONFIG ===
CSV_PATH = Path(
    "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_openai_gpt-4o-mini_5_20250807_1418.csv"
).expanduser()
MODEL_LABEL = "OpenAI GPT-4o Mini"

# Where to save the LaTeX file
OUTPUT_DIR = Path("src/results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD CSV ===
# Only the columns we need for scoring and grouping
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


# === FUNCTIONS ===
def compute_sentiment(text):
    """
    Return TextBlob sentiment polarity in [-1, 1].
    Positive values mean more positive tone.
    """
    return TextBlob(text).sentiment.polarity


def make_tables_for_model(df_model, model_label):
    """
    Build LaTeX tables per domain for one model.

    For each domain:
      • For each base_id, create a neutral reference by concatenating all
        paraphrase_neutral responses across runs.
      • For each target variant, compute
          - BERTScore F1 vs the neutral reference
          - Mean sentiment polarity
          - Mean word length
      • Collect rows into a domain table.
    """
    domains = df_model["domain"].unique()
    latex_all = []

    for dom in domains:
        dom_df = df_model[df_model["domain"] == dom]
        table_rows = []

        # Walk each base_id within this domain
        for base_id in dom_df["base_id"].unique():
            base_group = dom_df[dom_df["base_id"] == base_id]

            # Build the neutral reference by joining all neutral responses
            base_texts = base_group[base_group["variant"] == "paraphrase_neutral"][
                "response"
            ].tolist()
            if not base_texts:
                # Skip if no neutral reference for this base_id
                continue
            base_concat = " ".join(base_texts)

            # Score each target variant against the neutral reference
            for var in ["base", "tone", "formality", "emotion"]:
                var_texts = base_group[base_group["variant"] == var][
                    "response"
                ].tolist()
                if not var_texts:
                    # Variant missing for this base_id
                    continue

                # BERTScore F1 vs the same reference for all responses
                P, R, F1 = score(
                    var_texts, [base_concat] * len(var_texts), lang="en", verbose=False
                )
                bert_mean = F1.mean().item()

                # Sentiment mean across responses of this variant
                sent_mean = sum(compute_sentiment(t) for t in var_texts) / len(
                    var_texts
                )

                # Word length mean for this variant
                lengths = base_group[base_group["variant"] == var]["word_len"].tolist()
                len_mean = sum(lengths) / len(lengths)

                # Add one row for this base_id and variant
                table_rows.append((var, bert_mean, sent_mean, len_mean))

        # Build LaTeX table for this domain
        latex = []
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        latex.append(f"\\label{{tab:{model_label}_{dom.replace(' ', '_')}}}")
        # Kept p{6cm} slot in case you later add a snippet column
        latex.append("\\begin{tabular}{lccc p{6cm}}")
        latex.append("\\hline")
        latex.append(
            "Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len  \\\\"
        )
        latex.append("\\hline")

        for row in table_rows:
            latex.append(f"{row[0]} & {row[1]:.3f} & {row[2]:.2f} & {row[3]:.1f}  \\\\")

        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex_all.append("\n".join(latex))

    # Return all domain tables joined together
    return "\n\n".join(latex_all)


# === RUN ===
merged_tables = make_tables_for_model(df, MODEL_LABEL)

# Write a standalone LaTeX file so you can compile immediately
out_path = OUTPUT_DIR / f"{MODEL_LABEL}_All_Domains.tex"
with open(out_path, "w") as f:
    f.write("\\documentclass[12pt,a4paper]{article}\n")
    f.write("\\usepackage[margin=2.5cm]{geometry}\n")
    f.write("\\usepackage{array}\n")
    f.write("\\usepackage{booktabs}\n")
    f.write(f"\\title{{{MODEL_LABEL} – Variant Sensitivity Across Domains}}\n")
    f.write("\\author{}\n\\date{}\n")
    f.write("\\begin{document}\n\\maketitle\n\n")
    f.write(merged_tables)
    f.write("\n\\end{document}\n")

print(f"✅ Full LaTeX file saved: {out_path}")


