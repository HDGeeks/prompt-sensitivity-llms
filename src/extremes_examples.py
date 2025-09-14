# file: src/extremes_examples.py
from pathlib import Path
import pandas as pd
from textblob import TextBlob
from bert_score import score

CLEARED_DIR = Path("src/outputs/cleared/")
CSV_FILES = sorted([p for p in CLEARED_DIR.glob("responses_*.csv") if p.is_file()])
OUTPUT_LATEX = Path("src/results/extreme_examples.tex")

VARIANTS = ["tone", "formality", "emotion"]  # we’ll compare each to neutral
SNIPPET_WORDS = 30


def sentiment(text: str) -> float:
    return TextBlob(text or "").sentiment.polarity


def safe_words(s: str, n: int) -> str:
    """ Return first n words of s, safely."""
    w = (s or "").replace("\n", " ").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def compute_pair_metrics(neutral_row: pd.Series, var_row: pd.Series) -> dict:
    # BERTScore F1 (variant vs neutral text)
    cand = [str(var_row["response"])]
    ref = [str(neutral_row["response"])]
    _, _, f1 = score(cand, ref, lang="en", verbose=False)
    f1_val = float(f1.mean().item())

    # sentiment shift and length percent
    s_neu = sentiment(str(neutral_row["response"]))
    s_var = sentiment(str(var_row["response"]))
    delta = s_var - s_neu

    wl_neu = float(neutral_row.get("word_len", 0) or 0)
    wl_var = float(var_row.get("word_len", 0) or 0)
    len_pct = 100.0 * wl_var / wl_neu if wl_neu > 0 else float("nan")

    return {
        "bert_score": f1_val,
        "sentiment_shift": delta,
        "len_percent": len_pct,
    }


def main():
    if not CSV_FILES:
        raise SystemExit(f"No CSVs found in {CLEARED_DIR}")

    frames = []
    for p in CSV_FILES:
        df = pd.read_csv(p, dtype={"variant": str, "response": str})
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Keep only what we need
    cols = [
        "model",
        "model_version",
        "domain",
        "base_id",
        "run_id",
        "variant",
        "response",
        "word_len",
    ]
    all_df = all_df[cols].copy()

    # Group key to pair neutral with each variant
    key_cols = ["model", "model_version", "domain", "base_id", "run_id"]
    results = []

    for key, g in all_df.groupby(key_cols):
        # must have a neutral baseline
        neu = g[g["variant"] == "paraphrase_neutral"]
        if neu.empty:
            continue
        neutral_row = neu.iloc[0]

        for v in VARIANTS:
            var = g[g["variant"] == v]
            if var.empty:
                continue
            var_row = var.iloc[0]
            m = compute_pair_metrics(neutral_row, var_row)

            results.append(
                {
                    "model": key[0],
                    "model_version": key[1],
                    "domain": key[2],
                    "base_id": key[3],
                    "run_id": key[4],
                    "style": v,
                    "bert_score": m["bert_score"],
                    "sentiment_shift": m["sentiment_shift"],
                    "len_percent": m["len_percent"],
                    "response": var_row["response"],
                }
            )

    if not results:
        raise SystemExit(
            "No paired (neutral vs variant) rows found. Check variants/columns."
        )

    met = pd.DataFrame(results)

    # Find extremes:
    #  - Tone: lowest BERTScore
    #  - Formality: largest |sentiment_shift|
    #  - Emotion: largest |len_percent - 100|
    records = []

    # Tone
    tone_df = met[met["style"] == "tone"]
    if not tone_df.empty:
        tone_row = tone_df.loc[tone_df["bert_score"].idxmin()]
        records.append(
            {
                "Style": "Tone (lowest F1)",
                "Model": tone_row["model_version"] or tone_row["model"],
                "Domain": tone_row["domain"],
                "F1": tone_row["bert_score"],
                "Delta": tone_row["sentiment_shift"],
                "Len%": tone_row["len_percent"],
                "Snippet": safe_words(tone_row["response"], SNIPPET_WORDS),
            }
        )

    # Formality
    form_df = met[met["style"] == "formality"].copy()
    if not form_df.empty:
        form_df["abs_delta"] = form_df["sentiment_shift"].abs()
        form_row = form_df.loc[form_df["abs_delta"].idxmax()]
        records.append(
            {
                "Style": "Formality (max |Δ|)",
                "Model": form_row["model_version"] or form_row["model"],
                "Domain": form_row["domain"],
                "F1": form_row["bert_score"],
                "Delta": form_row["sentiment_shift"],
                "Len%": form_row["len_percent"],
                "Snippet": safe_words(form_row["response"], SNIPPET_WORDS),
            }
        )

    # Emotion
    emo_df = met[met["style"] == "emotion"].copy()
    if not emo_df.empty:
        emo_df["abs_len_dev"] = (emo_df["len_percent"] - 100.0).abs()
        emo_row = emo_df.loc[emo_df["abs_len_dev"].idxmax()]
        records.append(
            {
                "Style": "Emotion (max |Len%−100|)",
                "Model": emo_row["model_version"] or emo_row["model"],
                "Domain": emo_row["domain"],
                "F1": emo_row["bert_score"],
                "Delta": emo_row["sentiment_shift"],
                "Len%": emo_row["len_percent"],
                "Snippet": safe_words(emo_row["response"], SNIPPET_WORDS),
            }
        )

    # Console summary
    for r in records:
        print(f"[{r['Style']}] {r['Model']} – {r['Domain']}")
        print(f"  F1={r['F1']:.3f}, Δ={r['Delta']:.2f}, Len%={r['Len%']:.1f}")
        print(f"  \"{r['Snippet']}\"")
        print()

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Extreme examples across all models: lowest tone F1, largest formality sentiment shift, and largest emotion length deviation relative to the neutral baseline.}",
        r"\label{tab:extreme-examples}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l l l c c c p{7cm}}",
        r"\toprule",
        r"Style & Model & Domain & F1 & $\Delta$ & Len\% & Example Snippet \\",
        r"\midrule",
    ]
    for r in records:
        F1 = "--" if pd.isna(r["F1"]) else f"{r['F1']:.3f}"
        D = "--" if pd.isna(r["Delta"]) else f"{r['Delta']:.2f}"
        L = "--" if pd.isna(r["Len%"]) else f"{r['Len%']:.1f}"
        snip = r["Snippet"].replace("&", r"\&")
        lines.append(
            f"{r['Style']} & {r['Model']} & {r['Domain']} & {F1} & {D} & {L} & {snip} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]

    OUTPUT_LATEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LATEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX saved -> {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
