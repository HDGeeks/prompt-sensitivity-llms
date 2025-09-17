import pandas as pd
from pathlib import Path
from textblob import TextBlob
from bert_score import score
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

CLEARED_DIR = Path("src/outputs/cleared")
OUTPUT_DIR = Path("src/results_agg1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = [
    "domain", "base_id", "variant", "run_id", "response", "word_len", "response_id"
]

DOMAIN_ORDER = [
    "Public Health", "Historical Events", "Political Systems", "Scientific Consensus", "Environmental Policy"
]

def sentiment(text):
    return TextBlob(text).sentiment.polarity

def model_label_from_filename(p: Path) -> str:
    parts = p.stem.split("_")
    if len(parts) < 3:
        return p.stem
    vendor, model_raw = parts[1], parts[2]
    if vendor == "openai":
        return {
            "gpt-4o-mini": "OpenAI GPT-4o-Mini",
            "gpt-4o": "OpenAI GPT-4o",
            "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
        }.get(model_raw, f"OpenAI {model_raw}")
    if vendor == "gemini":
        return model_raw
    if vendor == "llama":
        return model_raw.replace("_", " ")
    if vendor == "mistralai":
        return model_raw
    return model_raw

def metrics_for_base(base_group):
    ref_texts = base_group.loc[base_group["variant"] == "paraphrase_neutral", "response"].tolist()
    if not ref_texts:
        return {}
    ref_concat = " ".join(ref_texts)
    out = {}
    for var in ["base", "tone", "formality", "emotion"]:
        var_texts = base_group.loc[base_group["variant"] == var, "response"].tolist()
        if not var_texts:
            continue
        P, R, F1 = score(var_texts, [ref_concat] * len(var_texts), lang="en", verbose=False)
        bert_mean = F1.mean().item()
        sent_mean = sum(sentiment(t) for t in var_texts) / len(var_texts)
        lens = base_group.loc[base_group["variant"] == var, "word_len"].tolist()
        len_mean = sum(lens) / len(lens)
        snippet = var_texts[0][:60].replace("\n", " ") + "..."
        out[var] = dict(bert=bert_mean, sent=sent_mean, wlen=len_mean, snippet=snippet)
    return out

def make_tables(df_model, model_label):
    latex_all = []
    domains = [d for d in DOMAIN_ORDER if d in df_model["domain"].unique()] + [
        d for d in df_model["domain"].unique() if d not in DOMAIN_ORDER
    ]
    for dom in domains:
        dom_df = df_model[df_model["domain"] == dom]
        per_base = [metrics_for_base(dom_df[dom_df["base_id"] == bid]) for bid in dom_df["base_id"].unique()]
        per_base = [b for b in per_base if b]
        rows = []
        for var in ["base", "tone", "formality", "emotion"]:
            vals = [b[var] for b in per_base if var in b]
            if not vals:
                continue
            bert_avg = sum(v["bert"] for v in vals) / len(vals)
            sent_avg = sum(v["sent"] for v in vals) / len(vals)
            wlen_avg = sum(v["wlen"] for v in vals) / len(vals)
            rows.append((var, bert_avg, sent_avg, wlen_avg))
        t = []
        t.append("\\begin{table}[h]")
        t.append("\\centering")
        t.append(f"\\caption{{{model_label} -- {dom} -- Variant Sensitivity}}")
        label = f"{model_label}_{dom}".replace(" ", "_").replace("--", "-")
        t.append(f"\\label{{tab:{label}}}")
        t.append("\\begin{tabular}{lccc p{6cm}}")
        t.append("\\hline")
        t.append("Variant & BERTScore vs Neutral & Sentiment Polarity & Word Len  \\")
        t.append("\\hline")
        for var, b, s, wl in rows:
            t.append(f"{var} & {b:.3f} & {s:.2f} & {wl:.1f}  \\")
        t.append("\\hline")
        t.append("\\end{tabular}")
        t.append("\\end{table}")
        latex_all.append("\n".join(t))
    return "\n\n".join(latex_all)

def process_all_models():
    files = sorted(CLEARED_DIR.glob("responses_*.csv"))
    for csv_path in files:
        df = pd.read_csv(csv_path, usecols=REQ_COLS)
        model_label = model_label_from_filename(csv_path)
        merged = make_tables(df, model_label)
        out_path = OUTPUT_DIR / f"{model_label}_All_Domains_aggregated_rows.tex"
        with open(out_path, "w") as f:
            f.write("\\documentclass[12pt,a4paper]{article}\n")
            f.write("\\usepackage[margin=2.5cm]{geometry}\n")
            f.write("\\usepackage{array}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write(f"\\title{{{model_label} -- Variant Sensitivity Across Domains}}\n")
            f.write("\\author{}\n\\date{}\n")
            f.write("\\begin{document}\n\\maketitle\n\n")
            f.write(merged)
            f.write("\n\\end{document}\n")
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    process_all_models()
