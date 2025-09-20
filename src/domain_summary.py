# file: src/domain_summary_v2.py
from pathlib import Path
import re
import math
import argparse
import pandas as pd
from typing import Tuple, List, Dict, Optional

# ---------- Config ----------
DOM_FULL_ORDER = [
    "Public Health",
    "Historical Events",
    "Political Systems",
    "Scientific Consensus",
    "Environmental Policy",
]

STYLES = ["tone", "formality", "emotion"]

DEFAULT_INPUT_DIR = Path("src/results/aggregated_score")
DEFAULT_GLOB = "*_All_Domains_aggregated_rows.tex"

OUTPUT_LATEX = Path("src/results/domain_comparison_summary_v2.tex")
OUTPUT_CSV = Path("src/results/domain_comparison_summary_v2.csv")

# Dash-like separator: en dash, em dash, double hyphen, single hyphen (with spaces around)
DSEP = r"\s+(?:–|—|--|-)\s+"


# ---------- Helpers ----------
def _norm(s: str) -> str:
    """Normalize spaces and non-breaking spaces."""
    return " ".join(str(s).replace("\u00a0", " ").split()).strip()


def _to_float(tok: str) -> float:
    """
    Parse a float from a token that might have % or commas.
    Returns NaN on failure.
    """
    if tok is None:
        return float("nan")
    t = str(tok).strip()
    t = t.replace("%", "")
    t = t.replace(",", ".")  # be tolerant to locales
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _split_title_model(title_text: str) -> str:
    """
    Extract model label from \title{...}.
    - Expected: "<MODEL> – Variant Sensitivity Across Domains"
    - But be tolerant to -- / — / - and extra words.
    Return the part before the first dash group; else whole title.
    """
    tt = _norm(title_text)
    parts = re.split(DSEP, tt)
    return _norm(parts[0]) if parts else tt


def _extract_model_and_domain_from_caption(cap_text: str) -> Tuple[str, Optional[str]]:
    """
    Try to extract (model, domain) from a caption like:
    "<MODEL> – <DOMAIN> – Variant Sensitivity"
    Robust to --, —, -, spacing variations.
    Returns (model, domain or None).
    """
    cap = _norm(cap_text)
    parts = re.split(DSEP, cap)

    # Best case: [MODEL, DOMAIN, ...]
    if len(parts) >= 2:
        model = _norm(parts[0])
        # Try to pick the first token that matches a known domain
        for p in parts[1:]:
            pp = _norm(p)
            if pp in DOM_FULL_ORDER:
                return model, pp
        # If second token looks like a domain (even if not exact), keep it
        if len(parts) >= 2:
            return model, _norm(parts[1])

    # Fallback: search known domains inside the caption
    for dom in DOM_FULL_ORDER:
        if dom in cap:
            before = cap.split(dom, 1)[0].rstrip()
            before = re.sub(r"(?:\s*(?:–|—|--|-)\s*)+$", "", before)
            model = before if before else "UNKNOWN"
            return model, dom

    # Last resort
    return "UNKNOWN", None


def parse_aggregated_tex(tex_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Parse an *_All_Domains_aggregated_rows.tex into:
      - df: columns [domain, variant, bert, sent, wlen]
      - model_label: extracted from \\title{...}
    Robust to different dash types and minor formatting changes.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")

    # Model label from \title{...}
    m_title = re.search(r"\\title\{([^}]*)\}", text, flags=re.DOTALL)
    if not m_title:
        raise ValueError(f"Cannot find \\title{{...}} in {tex_path}")
    model_label = _split_title_model(m_title.group(1))

    cap_re = re.compile(r"\\caption\{([^}]*)\}")
    row_re = re.compile(
        r"^(?P<var>base|tone|formality|emotion)\s*&\s*"
        r"(?P<bert>[0-9.,+-]+)\s*&\s*"
        r"(?P<sent>[0-9.,+-]+)\s*&\s*"
        r"(?P<wlen>[0-9.,+-]+)"
    )

    rows: List[Tuple[str, str, float, float, float]] = []
    current_domain: Optional[str] = None

    for raw in text.splitlines():
        line = raw.strip()

        m_cap = cap_re.search(line)
        if m_cap:
            cap_text = _norm(m_cap.group(1))
            _, dom = _extract_model_and_domain_from_caption(cap_text)
            current_domain = _norm(dom) if dom else None
            continue

        m_row = row_re.match(line)
        if m_row and current_domain:
            variant = m_row.group("var").lower()
            bert = _to_float(m_row.group("bert"))
            sent = _to_float(m_row.group("sent"))
            wlen = _to_float(m_row.group("wlen"))
            rows.append((current_domain, variant, bert, sent, wlen))

        # Reset on end of table
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
    Returns one row per domain and 9 numeric columns (3 per style).
    """
    out_rows: List[Dict[str, float]] = []

    for dom in DOM_FULL_ORDER:
        # Collect per-model metrics for this domain
        per_style_vals: Dict[str, List[Tuple[float, float, float]]] = {s: [] for s in STYLES}

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
                l_abs = abs((100.0 * (v_wlen / b_wlen) if b_wlen else float("nan")) - 100.0)
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
        def f(x, fmt):
            return ("nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else fmt.format(x))
        line = (
            f"{r['Domain']}"
            f" & {f(r['Tone_F1'], '{:.3f}')}"
            f" & {f(r['Tone_DeltaAbs'], '{:.2f}')}"
            f" & {f(r['Tone_LenAbs'], '{:.1f}')}"
            f" & {f(r['Form_F1'], '{:.3f}')}"
            f" & {f(r['Form_DeltaAbs'], '{:.2f}')}"
            f" & {f(r['Form_LenAbs'], '{:.1f}')}"
            f" & {f(r['Emo_F1'], '{:.3f}')}"
            f" & {f(r['Emo_DeltaAbs'], '{:.2f}')}"
            f" & {f(r['Emo_LenAbs'], '{:.1f}')} \\"
        )
        lines.append(line)

    footer = [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(header + lines + footer)


def gather_files(input_dir: Path, glob_pat: str) -> List[Path]:
    files = sorted(input_dir.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No files match: {input_dir}/{glob_pat}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Aggregate domain comparison (v2)")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Directory containing aggregated tex files")
    parser.add_argument("--glob", type=str, default=DEFAULT_GLOB,
                        help="Glob for files (default: *_All_Domains_aggregated_rows.tex)")
    parser.add_argument("--out-csv", type=str, default=str(OUTPUT_CSV),
                        help="Output CSV path")
    parser.add_argument("--out-tex", type=str, default=str(OUTPUT_LATEX),
                        help="Output LaTeX path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = gather_files(input_dir, args.glob)

    parsed_dfs: List[pd.DataFrame] = []
    for p in files:
        df, model_label = parse_aggregated_tex(p)
        parsed_dfs.append(df)

    df_domain = aggregate_domains_across_models(parsed_dfs)
    # Order rows by our canonical domain order
    df_domain["__ord"] = df_domain["Domain"].apply(lambda d: DOM_FULL_ORDER.index(d))
    df_domain = df_domain.sort_values("__ord").drop(columns="__ord")

    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_domain.to_csv(out_csv, index=False)
    latex = format_domain_latex(df_domain)
    out_tex.write_text(latex, encoding="utf-8")

    print("=== LaTeX Table (v2) ===")
    print(latex)
    print(f"\nSaved CSV -> {out_csv}")
    print(f"Saved LaTeX -> {out_tex}")


if __name__ == "__main__":
    main()