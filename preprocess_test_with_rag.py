import argparse
import json
import os
import random
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from dictionary_rag import *

DATA_ROOT = "data"
DEFAULT_OUT = os.path.join(DATA_ROOT, "processed")
df = pd.read_csv("data/archive/1001-question-v3.csv")
records = []

for index, row in df.iterrows():
    article = row['前文']
    question = row['題幹']

    options_raw = []
    options_with_idx = []

    for i in range(1, 5):
        text = row[f'選項{i}']
        if pd.isna(text):
            continue
        text = str(text).strip()
        options_raw.append(text)
        options_with_idx.append(f"{i}. {text}")

    # print("options_raw:", options_raw)
    dict_block = build_dictionary_block(
        article=article,
        question=question,
        options=options_raw,
        dict_dir="data/dictionaries_cleaned"
    )

    input_text = ""

    input_text += (
        f"文章：\n{article}\n\n"
    )

    if dict_block:
        input_text += dict_block + "\n\n"
        # print("-----------------------------")
        # print("Article:\n", article)
        # print("Question:\n", question)
        # print("Options:\n", options_with_idx)
        # print("Added dictionary block:\n", dict_block)

    input_text += (
        f"問題：\n{question}\n\n"
        f"選項：\n" + "\n".join(options_with_idx) + "\n"
    )
    record = {
        "id": row['ID'],
        "instruction": "請根據文章回答下列選擇題，請只輸出正確選項的數字。",
        "input": input_text
    }
    # print(record)
    records.append(record)
    out_path = "data/1001-question-v3-rag.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved {len(records)} records to {out_path}")


