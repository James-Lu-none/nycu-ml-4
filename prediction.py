import json
import csv
import torch
from models import *
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_choice", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

MODEL_PATH = args.model_path
MODEL_CHOICE = args.model_choice
TEST_FILE = "data/1001-question-v3.jsonl"

timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
OUT_BASE = "output"
parsed_model_name = MODEL_PATH.split("/")[-1]
OUT_CSV = f"{OUT_BASE}/{parsed_model_name}_{timestamp}.csv"

os.makedirs(OUT_BASE, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_fn = globals()[MODEL_CHOICE]
model, tokenizer = model_fn(
    base_model_path=MODEL_PATH,
    load_lora=False
)

model.eval()

CHOICES = ["1", "2", "3", "4"]
choice_token_ids = [
    tokenizer.encode(c, add_special_tokens=False)[0]
    for c in CHOICES
]

results = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        qid = obj["id"]
        prompt = f"{obj['instruction']}\n{obj['input']}"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]

        choice_scores = {
            c: logits[token_id].item()
            for c, token_id in zip(CHOICES, choice_token_ids)
        }

        pred = max(choice_scores, key=choice_scores.get)

        results.append([qid, pred])


with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "answer"])
    writer.writerows(results)

print(f"prediction done, saved to {OUT_CSV}")
