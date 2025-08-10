# file: src/model_summary.py
from pathlib import Path
import re
import pandas as pd
from typing import Tuple, Dict, List

# ---------- Config ----------
AGG_FILES = [
    Path("src/results/Gemini-1.5-Pro_All_Domains_aggregated_rows.tex"),
    Path("src/results/Gemini-2.0-Flash_All_Domains_aggregated_rows.tex"),
    Path("src/results/LLaMA-3.1-8B-Instruct_All_Domains_aggregated_rows.tex"),
    Path("src/results/Mistral-7B-Instruct-v0.3_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-3.5 Turbo_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-4o_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-4o-Mini_All_Domains_aggregated_rows.tex"),
]

DOM_FULL_ORDER = [
    "Public Health",
    "Historical Events",
    "Political Systems",
    "Scientific Consensus",
    "Environmental Policy",
]

STYLES = ["tone", "formality", "emotion"]

OUTPUT_LATEX = Path("src/results/model_comparison_summary.tex")
OUTPUT_CSV = Path("src/results/model_comparison_summary.csv")


# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Normalize spaces and non-breaking spaces."""
    return " ".join(str(s).replace("\u00a0", " ").split()).strip()


def parse_aggregated_tex(tex_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Parse an aggregated *_All_Domains_aggregated_rows.tex file into:
      - df: columns [domain, variant, bert, sent, wlen]
      - model_label: from \\title{<MODEL> – Variant Sensitivity Across Domains}
    """
    text = tex_path.read_text(encoding="utf-8")

    # Model label from \title{...}
    m_title = re.search(r"\\title\{([^}]*)\}", text)
    if not m_title:
        raise ValueError(f"Cannot find \\title{{...}} in {tex_path}")
    title_text = _norm(m_title.group(1))
    # Split model label from the rest on spaced en dash or hyphen
    parts = re.split(r"\s+\u2013\s+|\s+-\s+", title_text)
    model_label = _norm(parts[0]) if parts else title_text

    # Capture current domain from \caption{<MODEL> – <DOMAIN> – Variant Sensitivity}
    cap_re = re.compile(r"\\caption\{([^}]*)\}")
    # Rows like: "tone & 0.827 & -0.04 & 104.0  \\"
    row_re = re.compile(
        r"^(base|tone|formality|emotion)\s*&\s*([0-9.]+)\s*&\s*([\-+0-9.]+)\s*&\s*([0-9.]+)"
    )

    rows: List[Tuple[str, str, float, float, float]] = []
    current_domain = None

    for raw in text.splitlines():
        line = raw.strip()

        m_cap = cap_re.search(line)
        if m_cap:
            cap_text = _norm(m_cap.group(1))
            # Expect: "<MODEL> – <DOMAIN> – Variant Sensitivity"
            parts = re.split(r"\s+\u2013\s+|\s+-\s+", cap_text)
            if len(parts) >= 3:
                current_domain = _norm(parts[1])
            else:
                # Fallback: try to find domain tokens directly
                current_domain = _norm(cap_text)
            continue

        m_row = row_re.match(line)
        if m_row and current_domain:
            variant = m_row.group(1)
            bert = float(m_row.group(2))
            sent = float(m_row.group(3))
            wlen = float(m_row.group(4))
            rows.append((current_domain, variant, bert, sent, wlen))

        if line.startswith(r"\end{tabular}"):
            current_domain = None

    if not rows:
        raise ValueError(f"No rows parsed from {tex_path}")

    df = pd.DataFrame(rows, columns=["domain", "variant", "bert", "sent", "wlen"])
    df["domain"] = df["domain"].map(_norm)
    return df, model_label


def summarize_model(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    For a single model's df (domain/variant/bert/sent/wlen):
      For each style in STYLES, compute across the 5 domains:
        - mean F1 (bert)
        - mean |Δ| where Δ = style.sent - base.sent
        - mean |Len%-100| where Len% = 100 * style.wlen / base.wlen
    Returns:
      { style: {"F1_mean":..., "Delta_abs_mean":..., "LenAbs_mean":...}, ... }
    """
    out: Dict[str, Dict[str, float]] = {}

    # Ensure we have the five domains
    found = sorted(set(df["domain"]))
    missing = [d for d in DOM_FULL_ORDER if d not in found]
    if missing:
        # Not fatal—some files might be partial, but warn
        print(f"[warn] missing domains {missing} in model file")

    for style in STYLES:
        metrics = []
        for dom in DOM_FULL_ORDER:
            sub = df[df["domain"] == dom]
            if sub.empty:
                continue
            base = sub[sub["variant"] == "base"]
            var = sub[sub["variant"] == style]
            if base.empty or var.empty:
                continue
            b_sent = float(base.iloc[0]["sent"])
            v_sent = float(var.iloc[0]["sent"])
            b_wlen = float(base.iloc[0]["wlen"])
            v_wlen = float(var.iloc[0]["wlen"])
            f1 = float(var.iloc[0]["bert"])

            delta = v_sent - b_sent
            len_pct = 100.0 * (v_wlen / b_wlen) if b_wlen != 0 else float("nan")
            metrics.append((f1, abs(delta), abs(len_pct - 100.0)))

        if not metrics:
            out[style] = {
                "F1_mean": float("nan"),
                "Delta_abs_mean": float("nan"),
                "LenAbs_mean": float("nan"),
            }
            continue

        f1_mean = sum(m[0] for m in metrics) / len(metrics)
        d_mean = sum(m[1] for m in metrics) / len(metrics)
        l_mean = sum(m[2] for m in metrics) / len(metrics)

        out[style] = {
            "F1_mean": f1_mean,
            "Delta_abs_mean": d_mean,
            "LenAbs_mean": l_mean,
        }
    return out


def format_summary_latex(df_summary: pd.DataFrame) -> str:
    """
    Create a LaTeX table with multi-column groups per style:
      Model | Tone(F1, |Δ|, |Len|) | Formality(...) | Emotion(...)
    """
    header = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model comparison summary across styles. F1 is BERTScore vs. neutral; $|\Delta|$ is mean absolute sentiment shift; $|{\rm Len}\%-100|$ is mean absolute length deviation in percent.}",
        r"\label{tab:model-comparison-summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Tone} & \multicolumn{3}{c}{Formality} & \multicolumn{3}{c}{Emotion} \\",
        r"Model & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ \\",
        r"\midrule",
    ]

    lines = []
    for _, r in df_summary.iterrows():
        line = (
            f"{r['Model']}"
            f" & {r['Tone_F1']:.3f} & {r['Tone_DeltaAbs']:.2f} & {r['Tone_LenAbs']:.1f}"
            f" & {r['Form_F1']:.3f} & {r['Form_DeltaAbs']:.2f} & {r['Form_LenAbs']:.1f}"
            f" & {r['Emo_F1']:.3f} & {r['Emo_DeltaAbs']:.2f} & {r['Emo_LenAbs']:.1f} \\\\"
        )
        lines.append(line)

    footer = [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(header + lines + footer)


def main():
    summaries = []
    rows_csv = []

    for tex_path in AGG_FILES:
        df, model_label = parse_aggregated_tex(tex_path)
        summary = summarize_model(df)

        row = {
            "Model": model_label,
            "Tone_F1": summary["tone"]["F1_mean"],
            "Tone_DeltaAbs": summary["tone"]["Delta_abs_mean"],
            "Tone_LenAbs": summary["tone"]["LenAbs_mean"],
            "Form_F1": summary["formality"]["F1_mean"],
            "Form_DeltaAbs": summary["formality"]["Delta_abs_mean"],
            "Form_LenAbs": summary["formality"]["LenAbs_mean"],
            "Emo_F1": summary["emotion"]["F1_mean"],
            "Emo_DeltaAbs": summary["emotion"]["Delta_abs_mean"],
            "Emo_LenAbs": summary["emotion"]["LenAbs_mean"],
        }
        summaries.append(row)
        rows_csv.append(row)

    df_summary = pd.DataFrame(summaries)

    # Sort example: best (highest) average F1 over all styles
    df_summary["_F1_allstyles"] = df_summary[["Tone_F1", "Form_F1", "Emo_F1"]].mean(
        axis=1
    )
    df_summary = df_summary.sort_values("_F1_allstyles", ascending=False).drop(
        columns="_F1_allstyles"
    )

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUTPUT_CSV, index=False)

    # Save LaTeX
    latex_str = format_summary_latex(df_summary)
    OUTPUT_LATEX.write_text(latex_str, encoding="utf-8")

    # Print for convenience
    print("=== LaTeX Table ===")
    print(latex_str)
    print(f"\nSaved CSV -> {OUTPUT_CSV}")
    print(f"Saved LaTeX -> {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
