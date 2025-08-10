import re
import pandas as pd
from pathlib import Path

DOM_FULL_ORDER = [
    "Public Health",
    "Historical Events",
    "Political Systems",
    "Scientific Consensus",
    "Environmental Policy",
]


def _norm(s: str) -> str:
    return " ".join(str(s).replace("\u00a0", " ").split()).strip()


def parse_aggregated_tex(tex_path: Path) -> pd.DataFrame:
    text = Path(tex_path).read_text(encoding="utf-8")
    cap_re = re.compile(r"\\caption\{([^}]*)\}")
    row_re = re.compile(
        r"^(base|tone|formality|emotion)\s*&\s*([0-9.]+)\s*&\s*([\-0-9.]+)\s*&\s*([0-9.]+)"
    )
    rows, current_domain = [], None
    for raw in text.splitlines():
        line = raw.strip()
        m = cap_re.search(line)
        if m:
            parts = re.split(r"\s+\u2013\s+|\s+-\s+", _norm(m.group(1)))
            if len(parts) >= 3:
                current_domain = _norm(parts[1])
            continue
        m = row_re.match(line)
        if m and current_domain:
            rows.append(
                (
                    current_domain,
                    m.group(1),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                )
            )
        if line.startswith(r"\end{tabular}"):
            current_domain = None
    if not rows:
        raise ValueError("No rows parsed.")
    df = pd.DataFrame(rows, columns=["domain", "variant", "bert", "sent", "wlen"])
    df["domain"] = df["domain"].map(_norm)
    return df


def tone_row_multiline(
    tex_path: Path, model_label="MODEL", baseline_variant="base"
) -> str:
    df = parse_aggregated_tex(tex_path)
    found = {_norm(d) for d in df["domain"].unique()}
    missing = [d for d in DOM_FULL_ORDER if _norm(d) not in found]
    if missing:
        raise ValueError(f"Missing domains {missing}. Found {sorted(found)}")

    lines = [model_label]
    for dom in DOM_FULL_ORDER:
        sub = df[df["domain"] == _norm(dom)]
        base = sub[sub["variant"] == baseline_variant].iloc[0]
        emotion = sub[sub["variant"] == "emotion"].iloc[0]
        f1 = emotion["bert"]
        delta = emotion["sent"] - base["sent"]
        len_pct = 100.0 * emotion["wlen"] / base["wlen"]
        lines.append(f"& {f1:.3f} & {delta:+.2f} & {len_pct:.1f}")
    lines[-1] += " \\\\"
    return "\n".join(lines)


print(
    tone_row_multiline(
        Path("src/results/OpenAI GPT-4o-Mini_All_Domains_aggregated_rows.tex"),
        "OpenAI GPT-4o-Mini",
    )
)

files = [
    Path("src/results/Gemini-2.0-Pro-All_Domains_aggregated_rows.tex"),
    Path("src/results/Gemini-2.0-Flash_All_Domains_aggregated_rows.tex"),
    Path("src/results/LLaMA-3.1-8B-Instruct_All_Domains_aggregated_rows.tex"),
    Path("src/results/Mistral-7B-Instruct-v0.3_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-3.5 Turbo_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-4o_All_Domains_aggregated_rows.tex"),
    Path("src/results/OpenAI GPT-4o-Mini_All_Domains_aggregated_rows.tex"),
]
