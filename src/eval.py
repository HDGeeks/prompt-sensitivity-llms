# file: gemini_tables.py
import pandas as pd
from bert_score import score
from textblob import TextBlob
from pathlib import Path

# === CONFIG ===
CSV_PATH = Path(
    "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_gemini_Gemini-2.0-Flash_5_20250808_1114.csv"
).expanduser()
MODEL_LABEL = "Gemini-2.0-Flash"
OUTPUT_DIR = Path("./tables_out")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    latex_outputs = {}

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

                # Compute BERTScore vs baseline (average over runs)
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

                # Example snippet (shortened)
                snippet = var_texts[0][:60].replace("\n", " ") + "..."

                table_rows.append((var, bert_mean, sent_mean, len_mean, snippet))

        # Build LaTeX table
        latex = []
        latex.append(f"\\begin{{table}}[h]")
        latex.append(f"\\centering")
        latex.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        latex.append(f"\\begin{{tabular}}{{lccc p{{6cm}}}}")
        latex.append("\\hline")
        latex.append(
            "Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len & Example Snippet \\\\"
        )
        latex.append("\\hline")

        for row in table_rows:
            latex.append(
                f"{row[0]} & {row[1]:.3f} & {row[2]:.2f} & {row[3]:.1f} & {row[4]} \\\\"
            )

        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        latex_outputs[dom] = "\n".join(latex)

    return latex_outputs


# === RUN ===
tables = make_tables_for_model(df, MODEL_LABEL)

for dom, tex in tables.items():
    out_path = OUTPUT_DIR / f"{MODEL_LABEL}_{dom.replace(' ', '_')}.tex"
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"✅ Saved {out_path}")
