import pandas as pd, glob

# 1. Load newest Gemini CSV
csv_path = "/Users/hd/Desktop/prompt-sensitivity-llms/src/outputs/responses_gemini_Gemini-2.0-Flash_5_20250808_1114.csv"
df = pd.read_csv(csv_path)

# Word difference from 100
df["word_diff"] = df["word_len"] - 100

# 1. Average deviation from 100 words per domain and variant
print("\n--- Avg deviation from 100 words per domain+variant ---")
print(df.groupby(["domain", "variant"])["word_diff"].mean().sort_values())

# 2. Latency mean and std per variant
print("\n--- Latency mean and std per variant ---")
print(df.groupby("variant")["latency_ms"].agg(["mean", "std"]).sort_values("std"))

# 3. Word length variation across runs for same base_id
print("\n--- Std of word length across runs per base_id (top 10) ---")
print(df.groupby("base_id")["word_len"].std().sort_values(ascending=False).head(10))

# 4. Duplicate responses across different prompts/variants
print("\n--- Sample duplicate responses ---")
duplicates = df.groupby("response").filter(lambda x: len(x) > 1)
print(duplicates[["domain", "variant", "prompt", "response"]].head(10))

# 5. Slowest and fastest completions
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
