# inspection_all.py

"""
This script inspects CSV files in the 'src/outputs/cleared' directory to validate data quality for LLM prompt sensitivity experiments.

Inputs:
- CSV files in the specified directory, each expected to have columns like 'err', 'response', 'word_len', 'domain', 'variant', 'latency_ms', 'base_id', and 'prompt'.

Operations:
- Reads each CSV file.
- Normalizes empty strings in 'err' and 'response' columns to NaN.
- Calculates error rates and missing values.
- Checks if the file meets the error threshold and has no missing values (except in 'err' and 'response').
- Prints diagnostics: error stats, missing values, word length deviation, latency stats, word length std per base_id, and sample duplicate responses.

Outputs:
- Prints detailed diagnostics for each file if OUTPUT_REPORT is True.
- Prints a summary indicating whether each file fits the quality criteria.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIG ===
CSV_DIR = Path("src/outputs/cleared")   # folder to scan
ERROR_THRESHOLD = 0  # allowed % error rate (0 means perfect)
OUTPUT_REPORT = True  # print detailed diagnostics


def inspect_csv(path):
    df = pd.read_csv(path)
    print(f"\n=== Inspecting {path.name} ===")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

    # Normalize empty strings → NaN
    df["err"] = df["err"].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)
    df["response"] = df["response"].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x)

    # Error stats
    df["has_error"] = df["err"].notna()
    total_rows = len(df)
    error_rows = df["has_error"].sum()
    error_rate = (error_rows / total_rows) * 100

    # Missing check
    missing_cols = df.isna().sum()

    # Decision
    fit = (error_rate <= ERROR_THRESHOLD) and (
        missing_cols.drop(["err", "response"], errors="ignore") == 0
    ).all()

    if OUTPUT_REPORT:
        print(f"Errors: {error_rows} ({error_rate:.2f}%)")
        print(f"Missing values per column:\n{missing_cols}")
        print(f"FIT for cleared/: {fit}")

        # Diagnostics
        print("\n--- Avg deviation from 100 words per domain+variant ---")
        df["word_diff"] = df["word_len"] - 100
        print(df.groupby(["domain", "variant"])["word_diff"].mean().sort_values())

        print("\n--- Latency mean and std per variant ---")
        print(df.groupby("variant")["latency_ms"].agg(["mean", "std"]).sort_values("std"))

        print("\n--- Std of word length across runs per base_id (top 10) ---")
        print(df.groupby("base_id")["word_len"].std().sort_values(ascending=False).head(10))

        print("\n--- Sample duplicate responses ---")
        duplicates = df.groupby("response").filter(lambda x: len(x) > 1)
        print(duplicates[["domain", "variant", "prompt", "response"]].head(10))

    return fit


def main():
    all_csvs = sorted(CSV_DIR.glob("*.csv"))
    summary = {}
    for csv_path in all_csvs:
        fit = inspect_csv(csv_path)
        summary[csv_path.name] = fit

    print("\n=== Final Summary ===")
    for fname, fit in summary.items():
        print(f"{fname}: {'✅ FIT' if fit else '❌ NOT FIT'}")


if __name__ == "__main__":
    main()