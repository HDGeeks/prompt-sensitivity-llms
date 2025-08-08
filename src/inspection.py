import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIG ===
CSV_PATH = "/Users/hd/Desktop/prompt-sensitivity-llms/src/outputs/responses_openai_gpt-4o-mini_5_20250807_1418.csv"
ERROR_THRESHOLD = 0  # allowed % error rate (0 means perfect)
OUTPUT_REPORT = True  # print details


def inspect_csv(path):
    df = pd.read_csv(path)
    # print path
    print(f"Inspecting CSV: {Path(path).name}")
    # Shape of the DataFrame
    print(df.shape)
    # Columns in the DataFrame
    print(df.columns.tolist())

    # --- Normalize empty strings to NaN in err & response ---
    df["err"] = df["err"].apply(
        lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )
    df["response"] = df["response"].apply(
        lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )

    # --- Error stats ---
    df["has_error"] = df["err"].notna()
    total_rows = len(df)
    error_rows = df["has_error"].sum()
    error_rate = (error_rows / total_rows) * 100

    # --- Structure check ---
    missing_cols = df.isna().sum()

    # --- Decision: FIT or NOT ---
    fit = (error_rate <= ERROR_THRESHOLD) and (
        missing_cols.drop(["err", "response"], errors="ignore") == 0
    ).all()

    if OUTPUT_REPORT:
        print(f"=== Inspection Report for {Path(path).name} ===")
        print(f"Total rows: {total_rows}")
        print(f"Errors: {error_rows} ({error_rate:.2f}%)")
        print(f"Missing values per column:\n{missing_cols}")
        print(f"FIT for cleared/: {fit}")
        print("\n--- Avg deviation from 100 words per domain+variant ---")
        df["word_diff"] = df["word_len"] - 100
        print(df.groupby(["domain", "variant"])["word_diff"].mean().sort_values())

        print("\n--- Latency mean and std per variant ---")
        print(
            df.groupby("variant")["latency_ms"].agg(["mean", "std"]).sort_values("std")
        )

        print("\n--- Std of word length across runs per base_id (top 10) ---")
        print(
            df.groupby("base_id")["word_len"]
            .std()
            .sort_values(ascending=False)
            .head(10)
        )

        print("\n--- Sample duplicate responses ---")
        duplicates = df.groupby("response").filter(lambda x: len(x) > 1)
        print(duplicates[["domain", "variant", "prompt", "response"]].head(10))

        print("\n--- Slowest 5 completions ---")
        print(
            df.sort_values("latency_ms", ascending=False).head(5)[
                ["domain", "variant", "prompt", "latency_ms"]
            ]
        )
        print("\n--- Fastest 5 completions ---")
        print(
            df.sort_values("latency_ms", ascending=True).head(5)[
                ["domain", "variant", "prompt", "latency_ms"]
            ]
        )

    return fit


if __name__ == "__main__":
    result = inspect_csv(CSV_PATH)
    if result:
        print("\n✅ CSV passed inspection. Move to cleared/")
    else:
        print("\n❌ CSV did not pass inspection. Keep in archive/")
