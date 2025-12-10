import argparse
import json
import os
import random
import csv
import pandas as pd
from typing import List, Dict, Any, Optional

DATA_ROOT = "data"
DEFAULT_OUT = os.path.join(DATA_ROOT, "processed")


df = pd.read_csv("data/archive/1001-question-v3.csv")
# print(df.head())
# print(df.columns)
# Index(['ID', '前文', '題幹', '選項1', '選項2', '選項3', '選項4'], dtype='object')
# load in to json and save data/processed/1001-question-v3.json
records = []
for index, row in df.iterrows():
    options = []
    article = row['前文']
    question = row['題幹']
    for i in range(1, 5):
        text = row[f'選項{i}']
        options.append(f"{i}. {text}")
    records.append({
        "id": row['ID'],
        "instruction": "請根據文章回答下列選擇題，請只輸出正確選項的數字。",
        "input": f"文章：\n{article}" + f"\n\n問題：{question}" + "\n\n選項：" + "\n".join(options) 
    })
out_path = "data/1001-question-v3.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved {len(records)} records to {out_path}")