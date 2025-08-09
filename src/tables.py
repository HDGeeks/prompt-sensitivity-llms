import pandas as pd
from pathlib import Path


def compute_tone_row_from_file(
    csv_path,
    model_label="MODEL",
    baseline_variant="base",
    domains_order=(
        "Public Health",
        "Historical Events",
        "Political Systems",
        "Scientific Consensus",
        "Environmental Policy",
    ),
    domain_map=None,
):
    df = pd.read_csv(csv_path)
    required_cols = {"domain", "variant", "bert", "sent", "wlen"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Abbreviations
    default_map = {
        "Public Health": "PH",
        "Historical Events": "HE",
        "Political Systems": "PS",
        "Scientific Consensus": "SC",
        "Environmental Policy": "EP",
    }
    name_map = domain_map or default_map
    df = df.copy()
    df["abbr"] = df["domain"].map(name_map).fillna(df["domain"])

    cells = [model_label]
    for dom_full in domains_order:
        dom_abbr = name_map.get(dom_full, dom_full)
        sub = df[df["abbr"] == dom_abbr]
        if sub.empty:
            raise ValueError(f"Domain not found: {dom_full}")

        try:
            base = sub[sub["variant"] == baseline_variant].iloc[0]
            tone = sub[sub["variant"] == "tone"].iloc[0]
        except IndexError:
            raise ValueError(
                f"Missing rows for {dom_full}: need '{baseline_variant}' and 'tone'"
            )

        f1 = float(tone["bert"])
        delta = float(tone["sent"]) - float(base["sent"])
        len_pct = 100.0 * float(tone["wlen"]) / float(base["wlen"])

        cells.append(f"{f1:.3f} & {delta:+.2f} & {len_pct:.1f}")

    return " & ".join(cells) + " \\\\"


# usage
row = compute_tone_row_from_file(
    Path("src/results/Gemini-1.5-Pro_All_Domains_aggregated_rows.csv"),
    model_label="Gemini 1.5 Pro",
    baseline_variant="base",  # or "paraphrase_neutral" if you switch baseline later
)
print(row)
