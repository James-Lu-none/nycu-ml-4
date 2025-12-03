import argparse
import json
import os
import random
import csv
import pandas as pd
from typing import List, Dict, Any, Optional

DATA_ROOT = "data"
DEFAULT_OUT = os.path.join(DATA_ROOT, "processed")


def load_records(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return list(data)
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if not row:
                    continue
                article = row.get("前文") or row.get("文章") or row.get("context") or row.get("passage")
                question = row.get("題幹") or row.get("問題") or row.get("question")
                raw_opts = [
                    row.get("選項1"),
                    row.get("選項2"),
                    row.get("選項3"),
                    row.get("選項4"),
                ]
                options = []
                for opt in raw_opts:
                    if opt is None:
                        continue
                    text = str(opt).strip()
                    if not text or text.lower() == "nan":
                        continue
                    options.append(text)

                answer = pick_answer(row, options)
                rows.append({
                    "id": row.get("ID") or row.get("id"),
                    "article": article,
                    "question": question,
                    "options": options,
                    "answer": answer,
                })
            return rows
    raise ValueError(f"Unsupported file type: {ext}")


def pick_answer(row: Dict[str, Any], options: List[str]) -> Optional[str]:
    key_candidates = ["answer", "答案", "Answer", "正確答案"]
    for key in key_candidates:
        if key in row and row[key] not in (None, "", "nan"):
            raw = str(row[key]).strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            if options:
                for opt in options:
                    if opt.strip() == raw:
                        return raw
            return raw
    return None


def as_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if val.strip() == "":
            return []
        return [v.strip() for v in val.split("||") if v.strip()]
    return [str(val)]


def build_prompt(record: Dict[str, Any]) -> Dict[str, str]:
    id = record.get("id") or record.get("ID")
    article = record.get("article") or record.get("context") or record.get("passage") or record.get("input") or record.get("前文")
    question = record.get("question") or record.get("instruction") or record.get("prompt") or record.get("題幹") or record.get("問題")
    answer = record.get("answer") or record.get("output") or record.get("response") or record.get("答案") or record.get("正確答案")
    options = record.get("options") or record.get("choices")

    print (f"Processing ID: {id}, Answer: {answer}")
    if answer is None:
        return {}

    parts = []
    if article:
        parts.append(f"文章：\n{str(article).strip()}")
    if question:
        parts.append(f"問題：{str(question).strip()}")

    option_list = as_list(options)
    if option_list:
        formatted = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(option_list)])
        parts.append(f"選項：\n{formatted}")

    parts.append("答案：")
    prompt = "\n\n".join(parts)
    response = str(answer).strip()

    if not prompt.strip() or not response:
        return {}

    return {"prompt": prompt, "response": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Raw dataset path (json / jsonl / csv).")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.input_path)
    processed = []

    for rec in records:
        item = build_prompt(rec)
        if item:
            processed.append(item)

    if not processed:
        raise RuntimeError("No valid samples were built. Check input format.")

    random.seed(args.seed)
    random.shuffle(processed)

    split_idx = int(len(processed) * (1 - args.val_ratio))
    train, val = processed[:split_idx], processed[split_idx:]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total": len(processed),
        "train": len(train),
        "val": len(val),
    }
    print(stats)
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(train)} train / {len(val)} val samples to {args.out_dir}")


if __name__ == "__main__":
    main()
