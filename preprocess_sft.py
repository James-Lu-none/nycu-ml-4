import argparse
import json
import os
import random
import csv
import pandas as pd
from typing import List, Dict, Any, Optional

DATA_ROOT = "data"


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
            raw_reader = csv.reader(f)
            try:
                headers = next(raw_reader)
            except StopIteration:
                return []

            headers = [h.lstrip("\ufeff") for h in headers]
            rows = []

            def consume_row(fields: List[str]) -> Dict[str, Any]:
                # If columns align, map directly.
                if len(fields) == len(headers):
                    return dict(zip(headers, fields))

                # Handle malformed rows where an article field contains commas without quoting.
                header_set = set(headers)
                has_answer = "正確答案" in header_set or "答案" in header_set
                has_source = "資料來源" in header_set

                # Determine how many trailing fields belong to question/options/(answer/source).
                if has_answer or has_source:
                    tail = 7  # question + 4 options + answer + source
                else:
                    tail = 5  # question + 4 options

                # Fall back to best-effort realignment.
                question = fields[-tail]
                opt1, opt2, opt3, opt4 = fields[-tail + 1 : -tail + 5]
                answer = fields[-2] if (has_answer or has_source) else None
                source = fields[-1] if has_source else None

                article_parts = fields[1 : len(fields) - tail]
                article = ",".join(article_parts).strip()

                return {
                    "ID": fields[0],
                    "文章": article,
                    "問題": question,
                    "選項1": opt1,
                    "選項2": opt2,
                    "選項3": opt3,
                    "選項4": opt4,
                    "正確答案": answer,
                    "資料來源": source,
                }

            for fields in raw_reader:
                if not fields or all(not str(x).strip() for x in fields):
                    continue
                row = consume_row(fields)

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

                answer = row.get("正確答案")
                rows.append({
                    "id": row.get("ID") or row.get("id"),
                    "article": article,
                    "question": question,
                    "options": options,
                    "answer": answer,
                })
            return rows
    raise ValueError(f"Unsupported file type: {ext}")

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
    article = record.get("article") 
    question = record.get("question")
    answer = record.get("answer")
    options = record.get("options")
    # print (f"Processing ID: {id}, Answer: {answer}")
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

    return {"id": id,"instruction": "","input": prompt, "response": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/archive/AI_conv.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=DATA_ROOT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.input_path)
    processed = []

    for rec in records:
        item = build_prompt(rec)
        if item:
            processed.append(item)
        else:
            print(f"Skipping invalid record ID: {rec.get('id') or rec.get('ID')}")

    if not processed:
        raise RuntimeError("No valid samples were built. Check input format.")

    random.seed(args.seed)
    random.shuffle(processed)

    train_path = os.path.join(args.out_dir, "sft.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(processed)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
