
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
