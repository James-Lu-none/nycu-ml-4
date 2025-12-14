import os
import pandas as pd

base_csv = "output/3-shot_Qwen-Qwen2.5-32B-Instruct_2025-12-13T06-46-18_rag.csv"
predictions_dir = "output"

df_base = pd.read_csv(base_csv)
files = os.listdir(predictions_dir)
sorted_files = sorted(files)
sorted_files = sorted(files, key=lambda x: x.split("_")[-1])

for file in sorted_files:
    if file.endswith(".csv"):
        predictions_csv = os.path.join(predictions_dir, file)
        print(f"Evaluating predictions from: {predictions_csv}")
    
    df_pred = pd.read_csv(predictions_csv)
    # column name might be ID or id, Answer or answer
    df_base.columns = df_base.columns.str.lower()
    df_pred.columns = df_pred.columns.str.lower()
    merged = pd.merge(df_base, df_pred, on="id", suffixes=('_base', '_pred'))
    accuracy = (merged['answer_base'].astype(str) == merged['answer_pred'].astype(str)).mean()
    # merged.to_csv("output/merged_results.csv", index=False, encoding="utf-8-sig")
    print(f"Accuracy: {accuracy * 100:.2f}%")