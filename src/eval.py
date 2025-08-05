import pandas as pd
from bert_score import score
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def evaluate_responses(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["err"].isna()]
    results = []
    for (model, domain, base_id), group in df.groupby(["model", "domain", "base_id"]):
        responses = group.set_index("variant")["response"].to_dict()
        variants = ["base", "paraphrase_neutral", "tone", "formality", "emotion"]
        for v1 in variants:
            if v1 not in responses:
                continue
            for v2 in variants:
                if v2 <= v1 or v2 not in responses:
                    continue
                P, R, F1 = score(
                    [responses[v1]],
                    [responses[v2]],
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                )
                sentiment_v1 = sentiment_analyzer(responses[v1])[0]["score"]
                sentiment_v2 = sentiment_analyzer(responses[v2])[0]["score"]
                results.append(
                    {
                        "model": model,
                        "domain": domain,
                        "base_id": base_id,
                        "variant_pair": f"{v1}_vs_{v2}",
                        "bertscore_f1": F1.item(),
                        "sentiment_v1": sentiment_v1,
                        "sentiment_v2": sentiment_v2,
                    }
                )
    return pd.DataFrame(results)


results_df = evaluate_responses("src/outputs/responses_all_20250805_112049.csv")
print(results_df.groupby(["model", "domain", "variant_pair"])["bertscore_f1"].mean())
print(
    results_df.groupby(["model", "domain", "variant_pair"])[
        ["sentiment_v1", "sentiment_v2"]
    ].mean()
)
