# file: model_tables.py
import pandas as pd
from bert_score import score
from textblob import TextBlob
from pathlib import Path
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# === CONFIG ===
CSV_PATH = Path(
    "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_gemini_Gemini-1.5-Pro_5_20250807_1439.csv"
).expanduser()

MODEL_LABEL = "Gemini-1.5-Pro"

OUTPUT_DIR = Path("src/results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === LOAD CSV ===
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
    return TextBlob(text).sentiment.polarity


def make_tables_for_model(df_model, model_label):
    domains = df_model["domain"].unique()
    latex_all = []

    for dom in domains:
        dom_df = df_model[df_model["domain"] == dom]
        table_rows = []

        for base_id in dom_df["base_id"].unique():
            base_group = dom_df[dom_df["base_id"] == base_id]

            # Baseline = paraphrase_neutral
            base_texts = base_group[base_group["variant"] == "paraphrase_neutral"][
                "response"
            ].tolist()
            if not base_texts:
                continue
            base_concat = " ".join(base_texts)  # reference for BERTScore

            for var in ["base", "tone", "formality", "emotion"]:
                var_texts = base_group[base_group["variant"] == var][
                    "response"
                ].tolist()
                if not var_texts:
                    continue

                # Compute BERTScore vs baseline
                P, R, F1 = score(
                    var_texts, [base_concat] * len(var_texts), lang="en", verbose=False
                )
                bert_mean = F1.mean().item()

                # Sentiment polarity mean
                sent_mean = sum(compute_sentiment(t) for t in var_texts) / len(
                    var_texts
                )

                # Word length mean
                lengths = base_group[base_group["variant"] == var]["word_len"].tolist()
                len_mean = sum(lengths) / len(lengths)

                # Example snippet
                # snippet = var_texts[0][:60].replace("\n", " ") + "..."

                table_rows.append((var, bert_mean, sent_mean, len_mean))

        # Build LaTeX table for domain
        latex = []
        latex.append(f"\\begin{{table}}[h]")
        latex.append(f"\\centering")
        latex.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        latex.append(f"\\label{{tab:{model_label}_{dom.replace(' ', '_')}}}")
        latex.append(f"\\begin{{tabular}}{{lccc p{{6cm}}}}")
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

    return "\n\n".join(latex_all)


# === RUN ===
merged_tables = make_tables_for_model(df, MODEL_LABEL)

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
