import argparse
import csv
import json
import os
import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

OUTPUT_ROOT = "output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)


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
        # Handle the competition test.csv layout:
        # ID starts the line; 前文 may span multiple lines; then 題幹, 選項1-4.
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if not row:
                    continue
                article = row.get("前文") or row.get("context")
                question = row.get("題幹") or row.get("question")
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
                rows.append({
                    "id": row.get("ID"),
                    "article": article,
                    "question": question,
                    "options": options,
                })
            return rows
    raise ValueError(f"Unsupported file type: {ext}")


def build_prompt(record: Dict[str, Any]) -> str:
    article = (
        record.get("article")
        or record.get("context")
        or record.get("passage")
        or record.get("input")
        or record.get("前文")
    )
    question = (
        record.get("question")
        or record.get("instruction")
        or record.get("prompt")
        or record.get("題幹")
    )
    options = record.get("options") or record.get("choices")

    parts = []
    if article:
        parts.append(f"文章：\n{str(article).strip()}")
    if question:
        parts.append(f"問題：{str(question).strip()}")
    if options:
        if isinstance(options, list):
            opts = options
        else:
            opts = [o.strip() for o in str(options).split("||") if o.strip()]
        formatted = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(opts)])
        parts.append(f"選項：\n{formatted}")
    parts.append("答案：")
    return "\n\n".join(parts)


class Predictor:
    def __init__(self, model_dir, max_new_tokens=128, temperature=0.7, top_p=0.9):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()

    def run(self, input_path: str):
        self.load_model()
        records = load_records(input_path)
        rows = []

        for idx, rec in enumerate(records):
            prompt = build_prompt(rec)
            answer = self.generate_answer(prompt)
            rows.append({
                "ID": rec.get("ID", idx),
                "prompt": prompt,
                "Answer": answer,
            })

        df = pd.DataFrame(rows)
        timestamp = np.datetime64("now").astype("str").replace(":", "-").replace(" ", "_")
        model_tag = os.path.basename(self.model_dir.rstrip("/"))
        out_path = os.path.join(OUTPUT_ROOT, f"{model_tag}_{timestamp}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    predictor = Predictor(
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    predictor.run(args.input_path)
