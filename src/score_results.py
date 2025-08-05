import sys
from pathlib import Path
import pandas as pd
from bert_score import score as bert_score
from nltk.sentiment import SentimentIntensityAnalyzer

IN_PATH = Path(sys.argv[1])  # responses_raw_TIMESTAMP.csv
OUT_DIR = Path("src/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_csv(IN_PATH)

# Filter out errors/empties
df = df[~df["response"].fillna("").str.startswith("[ERROR]")]
df["response"] = df["response"].fillna("").astype(str)

# --- Pair each (model, run, domain) variant with its base ---
base = df[df["variant"] == "base"][["model", "run", "domain", "response"]]
base = base.rename(columns={"response": "base_response"})

merged = df.merge(base, on=["model", "run", "domain"], how="inner")
merged = merged[merged["variant"] != "base"].copy()


# --- BERTScore: compare variant response to base_response (same model/run/domain) ---
def bertscore_block(sub):
    cand = sub["response"].tolist()
    ref = sub["base_response"].tolist()
    P, R, F = bert_score(cand, ref, lang="en", rescale_with_baseline=True)
    sub = sub.copy()
    sub["bertscore_f1"] = F.tolist()
    sub["bertscore_p"] = P.tolist()
    sub["bertscore_r"] = R.tolist()
    return sub


scored = merged.groupby(["model", "run", "domain", "variant"], group_keys=False).apply(
    bertscore_block
)

# --- Sentiment (VADER compound) for ALL responses ---
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["response"].apply(lambda t: sia.polarity_scores(t)["compound"])

# For the delta, join base sentiment
base_sent = df[df["variant"] == "base"][["model", "run", "domain", "sentiment"]]
base_sent = base_sent.rename(columns={"sentiment": "base_sentiment"})
sent_M = df.merge(base_sent, on=["model", "run", "domain"], how="left")
sent_M["delta_sentiment"] = sent_M["sentiment"] - sent_M["base_sentiment"]

# Keep only non-base rows for delta reporting
sent_delta = sent_M[sent_M["variant"] != "base"][
    [
        "model",
        "run",
        "domain",
        "variant",
        "delta_sentiment",
        "sentiment",
        "base_sentiment",
    ]
]

# === Write per-run detail CSVs ===
scored_out = (
    OUT_DIR / f"bertscore_perrun_{IN_PATH.stem.split('responses_raw_')[-1]}.csv"
)
sent_out = OUT_DIR / f"sentiment_perrun_{IN_PATH.stem.split('responses_raw_')[-1]}.csv"
scored.to_csv(scored_out, index=False)
sent_delta.to_csv(sent_out, index=False)
print(f"✅ Saved: {scored_out}")
print(f"✅ Saved: {sent_out}")


# === Aggregates for tables (mean ± sd across domains and runs) ===
def agg_berts(dfB):
    g = dfB.groupby(["model", "variant"])["bertscore_f1"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out


def agg_sent(dfS):
    g = dfS.groupby(["model", "variant"])["delta_sentiment"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out


aggB = agg_berts(scored)
aggS = agg_sent(sent_delta)

aggB_out = OUT_DIR / f"bertscore_agg_{IN_PATH.stem.split('responses_raw_')[-1]}.csv"
aggS_out = OUT_DIR / f"sentiment_agg_{IN_PATH.stem.split('responses_raw_')[-1]}.csv"
aggB.to_csv(aggB_out, index=False)
aggS.to_csv(aggS_out, index=False)
print(f"✅ Saved: {aggB_out}")
print(f"✅ Saved: {aggS_out}")
