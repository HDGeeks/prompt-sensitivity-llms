# file: model_tables_agg.py
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from bert_score import score

# === CONFIG ===
CSV_PATH = Path(
    "~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/responses_gemini_Gemini-2.0-Flash_5_20250808_1114.csv"
).expanduser()
MODEL_LABEL = "Gemini-2.0-Flash"
OUTPUT_DIR = Path("~/Desktop/prompt-sensitivity-llms/src/outputs/cleared/results/")
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


def sentiment(text):
    return TextBlob(text).sentiment.polarity


def metrics_for_base(base_group):
    # reference is paraphrase_neutral joined over runs
    ref_texts = base_group.loc[
        base_group["variant"] == "paraphrase_neutral", "response"
    ].tolist()
    if not ref_texts:
        return {}
    ref_concat = " ".join(ref_texts)

    out = {}
    for var in ["base", "tone", "formality", "emotion"]:
        var_texts = base_group.loc[base_group["variant"] == var, "response"].tolist()
        if not var_texts:
            continue
        P, R, F1 = score(
            var_texts, [ref_concat] * len(var_texts), lang="en", verbose=False
        )
        bert_mean = F1.mean().item()
        sent_mean = sum(sentiment(t) for t in var_texts) / len(var_texts)
        lens = base_group.loc[base_group["variant"] == var, "word_len"].tolist()
        len_mean = sum(lens) / len(lens)
        snippet = var_texts[0][:60].replace("\n", " ") + "..."
        out[var] = dict(bert=bert_mean, sent=sent_mean, wlen=len_mean, snippet=snippet)
    return out


def make_tables(df_model, model_label):
    latex_all = []
    for dom in df_model["domain"].unique():
        dom_df = df_model[df_model["domain"] == dom]

        # compute per-base_id metrics
        per_base = []
        for bid in dom_df["base_id"].unique():
            base_group = dom_df[dom_df["base_id"] == bid]
            m = metrics_for_base(base_group)
            if m:
                per_base.append(m)

        # aggregate across base_ids by variant
        rows = []
        for var in ["base", "tone", "formality", "emotion"]:
            vals = [b[var] for b in per_base if var in b]
            if not vals:
                continue
            bert_avg = sum(v["bert"] for v in vals) / len(vals)
            sent_avg = sum(v["sent"] for v in vals) / len(vals)
            wlen_avg = sum(v["wlen"] for v in vals) / len(vals)
            # pick a representative snippet
            # snippet = vals[0]["snippet"]
            rows.append((var, bert_avg, sent_avg, wlen_avg))

        # build table
        t = []
        t.append("\\begin{table}[h]")
        t.append("\\centering")
        t.append(f"\\caption{{{model_label} – {dom} – Variant Sensitivity}}")
        t.append(f"\\label{{tab:{model_label}_{dom.replace(' ','_')}}}")
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

    return "\n\n".join(latex_all)


merged = make_tables(df, MODEL_LABEL)

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
