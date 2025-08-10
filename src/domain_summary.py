# file: src/domain_summary.py
from pathlib import Path
import re
import pandas as pd
from typing import Tuple, List, Dict

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

OUTPUT_LATEX = Path("src/results/domain_comparison_summary.tex")
OUTPUT_CSV = Path("src/results/domain_comparison_summary.csv")


# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Normalize spaces and non-breaking spaces."""
    return " ".join(str(s).replace("\u00a0", " ").split()).strip()


def parse_aggregated_tex(tex_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Parse aggregated *_All_Domains_aggregated_rows.tex into:
      - df: columns [domain, variant, bert, sent, wlen]
      - model_label: from \\title{<MODEL> – Variant Sensitivity Across Domains}
    """
    text = tex_path.read_text(encoding="utf-8")

    # Model label from \title{...}
    m_title = re.search(r"\\title\{([^}]*)\}", text)
    if not m_title:
        raise ValueError(f"Cannot find \\title{{...}} in {tex_path}")
    title_text = _norm(m_title.group(1))
    parts = re.split(r"\s+\u2013\s+|\s+-\s+", title_text)
    model_label = _norm(parts[0]) if parts else title_text

    cap_re = re.compile(r"\\caption\{([^}]*)\}")
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
            current_domain = _norm(parts[1]) if len(parts) >= 3 else _norm(cap_text)
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


def aggregate_domains_across_models(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    For each domain and style, average across models:
      - F1_mean = mean(style.bert)
      - |Δ|_mean = mean(abs(style.sent - base.sent))
      - |Len%-100|_mean = mean(abs(100*style.wlen/base.wlen - 100))
    Returns a wide table with one row per domain and 9 numeric columns (3 per style).
    """
    out_rows: List[Dict[str, float]] = []

    for dom in DOM_FULL_ORDER:
        # Collect per-model metrics for this domain
        per_style_vals = {s: [] for s in STYLES}

        for df in dfs:
            sub = df[df["domain"] == dom]
            if sub.empty:
                continue
            base = sub[sub["variant"] == "base"]
            if base.empty:
                continue
            b_sent = float(base.iloc[0]["sent"])
            b_wlen = float(base.iloc[0]["wlen"])

            for style in STYLES:
                var = sub[sub["variant"] == style]
                if var.empty:
                    continue
                v_sent = float(var.iloc[0]["sent"])
                v_wlen = float(var.iloc[0]["wlen"])
                f1 = float(var.iloc[0]["bert"])
                d_abs = abs(v_sent - b_sent)
                l_abs = abs(
                    (100.0 * (v_wlen / b_wlen) if b_wlen else float("nan")) - 100.0
                )
                per_style_vals[style].append((f1, d_abs, l_abs))

        # Compute means across models for this domain
        row = {"Domain": dom}
        for style in STYLES:
            vals = per_style_vals[style]
            if vals:
                f1_mean = sum(v[0] for v in vals) / len(vals)
                d_mean = sum(v[1] for v in vals) / len(vals)
                l_mean = sum(v[2] for v in vals) / len(vals)
            else:
                f1_mean = d_mean = l_mean = float("nan")

            prefix = {"tone": "Tone", "formality": "Form", "emotion": "Emo"}[style]
            row[f"{prefix}_F1"] = f1_mean
            row[f"{prefix}_DeltaAbs"] = d_mean
            row[f"{prefix}_LenAbs"] = l_mean

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def format_domain_latex(df_dom: pd.DataFrame) -> str:
    header = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Domain comparison summary across styles. F1 is BERTScore vs.\ neutral; $|\Delta|$ is mean absolute sentiment shift; $|{\rm Len}\%-100|$ is mean absolute length deviation in percent.}",
        r"\label{tab:domain-comparison-summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Tone} & \multicolumn{3}{c}{Formality} & \multicolumn{3}{c}{Emotion} \\",
        r"Domain & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ & F1 & $|\Delta|$ & $|{\rm Len}\%-100|$ \\",
        r"\midrule",
    ]

    lines = []
    for _, r in df_dom.iterrows():
        line = (
            f"{r['Domain']}"
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
    parsed = []
    for p in AGG_FILES:
        df, model_label = parse_aggregated_tex(p)
        parsed.append(df)

    df_domain = aggregate_domains_across_models(parsed)

    # Order rows by our canonical domain order
    df_domain["__ord"] = df_domain["Domain"].apply(lambda d: DOM_FULL_ORDER.index(d))
    df_domain = df_domain.sort_values("__ord").drop(columns="__ord")

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_domain.to_csv(OUTPUT_CSV, index=False)

    # Save LaTeX
    latex = format_domain_latex(df_domain)
    OUTPUT_LATEX.write_text(latex, encoding="utf-8")

    print("=== LaTeX Table ===")
    print(latex)
    print(f"\nSaved CSV -> {OUTPUT_CSV}")
    print(f"Saved LaTeX -> {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
