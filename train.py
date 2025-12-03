import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
from models import *

DATA_DIR = "data/processed"
MODEL_ROOT = "model"

os.makedirs(MODEL_ROOT, exist_ok=True)


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]


class QADataset(Dataset):
    def __init__(self, path: str):
        self.df = pd.read_json(path, lines=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "prompt": row["prompt"],
            "response": row["response"],
        }


@dataclass
class SFTCollator:
    tokenizer: any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        labels = []

        for item in features:
            prompt_ids = self.tokenizer.encode(
                item["prompt"],
                add_special_tokens=False,
            )
            response_ids = self.tokenizer.encode(
                item["response"] + self.tokenizer.eos_token,
                add_special_tokens=False,
            )

            ids = (prompt_ids + response_ids)[:self.max_length]
            lbl = ([-100] * len(prompt_ids) + response_ids)[:self.max_length]

            if len(lbl) < len(ids):
                lbl.extend([-100] * (len(ids) - len(lbl)))

            input_ids.append(ids)
            labels.append(lbl)

        max_len = min(self.max_length, max(len(ids) for ids in input_ids))

        padded_inputs, attn_mask, padded_labels = [], [], []
        for ids, lbl in zip(input_ids, labels):
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            pad_len = max_len - len(ids)

            padded_inputs.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attn_mask.append([1] * len(ids) + [0] * pad_len)
            padded_labels.append(lbl + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch


class Train:
    def __init__(self, dataset_dir, model_choice, model_state_path=None, max_length=1024):
        self.dataset_dir = dataset_dir
        self.model_choice = model_choice
        self.model_state_path = model_state_path
        self.max_length = max_length

        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None

        os.makedirs(os.path.join(MODEL_ROOT, self.model_choice), exist_ok=True)

    def load_model(self):
        if self.model_state_path is not None:
            print(f"Loading custom model: {self.model_state_path}")
            self.tokenizer, self.model = custom(self.model_state_path)
        else:
            print(f"Loading model: {self.model_choice}")
            fn = globals()[self.model_choice]
            self.tokenizer, self.model = fn()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_data(self):
        train_path = os.path.join(self.dataset_dir, "train.jsonl")
        val_path = os.path.join(self.dataset_dir, "val.jsonl")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError("Expect train.jsonl and val.jsonl in dataset_dir.")

        print(f"Loading train set: {train_path}")
        print(f"Loading val set: {val_path}")
        self.train_dataset = QADataset(train_path)
        self.val_dataset = QADataset(val_path)
        print(f"Samples -> train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")

    def setup_trainer(self):
        output_dir = f"{MODEL_ROOT}/{self.model_choice}"
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            logging_steps=25,
            dataloader_num_workers=2,
            predict_with_generate=True,
            generation_max_length=256,
            report_to="none",
            gradient_checkpointing=False,
        )

        data_collator = SFTCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
            pred_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            scores = []
            for p, r in zip(pred_str, label_str):
                p = p.strip()
                r = r.strip()
                dist = levenshtein(p, r)
                scores.append(dist / max(1, len(r)))

            return {"lev": float(sum(scores) / len(scores))}

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def run(self):
        self.load_model()
        self.load_data()
        self.setup_trainer()

        print("Training started...")
        self.trainer.train()

        timestamp = np.datetime64("now").astype("str").replace(":", "-").replace(" ", "_")
        best_metric = self.trainer.state.best_metric
        metric_str = f"{best_metric:.4f}" if best_metric is not None else "NA"
        save_dir = f"{MODEL_ROOT}/{self.model_choice}/{timestamp}_{metric_str}"

        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving model to {save_dir}")
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        log_history = self.trainer.state.log_history
        json_path = os.path.join(save_dir, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2)

        df = pd.DataFrame(log_history)
        csv_path = os.path.join(save_dir, "metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to: {csv_path}")

        if "loss" in df.columns and "eval_loss" in df.columns:
            plt.figure()
            df["loss"] = df["loss"].interpolate()
            df["eval_loss"] = df["eval_loss"].interpolate()
            plt.plot(df["step"], df["loss"], label="train_loss")
            plt.plot(df["step"], df["eval_loss"], label="eval_loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training / Eval Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=150)
            plt.close()

        eval_key = "eval_lev"
        if eval_key in df.columns:
            plt.figure()
            plt.plot(df["step"], df[eval_key], marker="o")
            plt.xlabel("Step")
            plt.ylabel("Normalized Levenshtein")
            plt.title("Eval Levenshtein Distance")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "lev_plot.png"), dpi=150)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=DATA_DIR)
    parser.add_argument("--model_choice", type=str, required=True)
    parser.add_argument("--model_state_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    trainer = Train(
        dataset_dir=args.dataset_dir,
        model_choice=args.model_choice,
        model_state_path=args.model_state_path,
        max_length=args.max_length,
    )
    trainer.run()
