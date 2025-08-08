import pandas as pd, glob

# 1. Load newest Gemini CSV
csv_path = "/Users/hd/Desktop/prompt-sensitivity-llms/src/outputs/responses_gemini_20250807_1401.csv"
df = pd.read_csv(csv_path)

print
