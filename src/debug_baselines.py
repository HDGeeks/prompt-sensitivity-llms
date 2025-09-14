from pathlib import Path
from tables import parse_aggregated_tex, DOM_FULL_ORDER  # your existing helpers

def extract_tone_row(tex_path: Path, label: str):
    df = parse_aggregated_tex(tex_path)  # columns: domain, variant, bert, sent, wlen
    out = [label]
    for dom in DOM_FULL_ORDER:
        sub = df[df["domain"] == dom]
        base = sub[sub["variant"] == "base"].iloc[0]
        tone = sub[sub["variant"] == "tone"].iloc[0]
        f1 = tone["bert"]
        d_sent = tone["sent"] - base["sent"]
        len_pct = 100.0 * tone["wlen"] / base["wlen"]
        out.append((f1, d_sent, len_pct))
    return out

models = {
    "Gemini 1.5 Pro": "src/results/Gemini-1.5-Pro_All_Domains_aggregated_rows.tex",
    "Gemini 2.0 Flash": "src/results/Gemini-2.0-Flash_All_Domains_aggregated_rows.tex",
    "LLaMA 3.1 8B Instruct": "src/results/LLaMA-3.1-8B-Instruct_All_Domains_aggregated_rows.tex",
    "Mistral 7B Instruct v0.3": "src/results/Mistral-7B-Instruct-v0.3_All_Domains_aggregated_rows.tex",
    "OpenAI GPT-3.5 Turbo": "src/results/OpenAI GPT-3.5 Turbo_All_Domains_aggregated_rows.tex",
    "OpenAI GPT-4o-Mini": "src/results/OpenAI GPT-4o-Mini_All_Domains_aggregated_rows.tex",
    "OpenAI GPT-4o": "src/results/OpenAI GPT-4o_All_Domains_aggregated_rows.tex",
}

for label, path in models.items():
    row = extract_tone_row(Path(path), label)
    print(f"\n=== {label} ===")
    for (dom, (f1, d, L)) in zip(DOM_FULL_ORDER, row[1:]):
        print(f"{dom:20s}  F1={f1:.3f}  Δ={d:+.2f}  Len%={L:.1f}")

# # file: src/debug_baselines.py
# import re
# import pandas as pd
# from pathlib import Path

# DOM_FULL_ORDER = [
#     "Public Health",
#     "Historical Events",
#     "Political Systems",
#     "Scientific Consensus",
#     "Environmental Policy",
# ]

# AGG_GLOB = "src/results/*_All_Domains_aggregated_rows.tex"

# def _norm(s: str) -> str:
#     return " ".join(str(s).replace("\u00a0", " ").split()).strip()

# def parse_aggregated_tex(tex_path: Path) -> pd.DataFrame:
#     text = Path(tex_path).read_text(encoding="utf-8")
#     cap_re = re.compile(r"\\caption\{([^}]*)\}")
#     row_re = re.compile(
#         r"^(base|tone|formality|emotion)\s*&\s*([0-9.]+)\s*&\s*([\-0-9.]+)\s*&\s*([0-9.]+)"
#     )
#     rows, current_domain = [], None
#     for raw in text.splitlines():
#         line = raw.strip()
#         m = cap_re.search(line)
#         if m:
#             parts = re.split(r"\s+\u2013\s+|\s+-\s+", _norm(m.group(1)))
#             if len(parts) >= 3:
#                 current_domain = _norm(parts[1])
#             continue
#         m = row_re.match(line)
#         if m and current_domain:
#             rows.append((
#                 current_domain,
#                 m.group(1),
#                 float(m.group(2)),   # bert (F1 vs neutral)
#                 float(m.group(3)),   # sent
#                 float(m.group(4)),   # wlen
#             ))
#         if line.startswith(r"\end{tabular}"):
#             current_domain = None
#     return pd.DataFrame(rows, columns=["domain","variant","bert","sent","wlen"])

# def debug_baselines(tex_path: Path):
#     df = parse_aggregated_tex(tex_path)
#     model = tex_path.name.replace("_All_Domains_aggregated_rows.tex","").replace("_"," ")
#     print(f"\n=== {model} ===")
#     for dom in DOM_FULL_ORDER:
#         sub = df[df["domain"] == dom]
#         base = sub[sub["variant"]=="base"].iloc[0]
#         neutral = sub[sub["variant"]=="base"].iloc[0]  # NB: aggregated F1 already neutral-ref
#         for style in ["tone","formality","emotion"]:
#             var = sub[sub["variant"]==style].iloc[0]
#             # Δ vs base
#             delta_base = var["sent"] - base["sent"]
#             len_base = 100 * var["wlen"]/base["wlen"]
#             # Δ vs neutral (F1 already vs neutral, so recompute sentiment/len hypothetically)
#             delta_neu = var["sent"] - neutral["sent"]
#             len_neu = 100 * var["wlen"]/neutral["wlen"]
#             print(f"{dom:<22} {style:<10} F1={var['bert']:.3f} "
#                   f"Δ(base)={delta_base:+.2f}, Len%(base)={len_base:.1f} "
#                   f"| Δ(neutral)={delta_neu:+.2f}, Len%(neutral)={len_neu:.1f}")

# if __name__=="__main__":
#     for tex in Path().glob(AGG_GLOB):
#         debug_baselines(tex)