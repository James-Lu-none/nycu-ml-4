from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from models import *
import os
import numpy as np
import argparse
from plot_loss import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_choice", type=str, required=True)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

if args.model_path is None:
    base_dir = os.path.join("model_cpt", args.model_choice)
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_model_dir = max(model_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)) if "checkpoint" not in d else 0)
    MODEL_NAME = os.path.join(base_dir, latest_model_dir)
else:
    MODEL_NAME = args.model_path

MODEL_CHOICE = args.model_choice
DATA_PATH = "data/cpt.jsonl"
OUT_DIR = f"model_cpt/{MODEL_CHOICE}"

print("Using model:", MODEL_NAME)

os.makedirs(OUT_DIR, exist_ok=True)

model_fn = globals()[MODEL_CHOICE]
model, tokenizer = model_fn(
    base_model_path=MODEL_NAME,
    load_lora=True
)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
    )

tokenized_train_dataset = train_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

tokenized_val_dataset = val_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=2,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    eval_strategy="steps",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator
)

trainer.train()
log_history = trainer.state.log_history

loss = trainer.state.log_history[-1]["loss"]
print("Final training loss:", loss)
timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')

OUT_DIR = OUT_DIR + f"/{timestamp}_{loss:.4f}"
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
plot_and_save_loss(log_history, OUT_DIR, title="CPT Loss")

print(f"CPT done, model saved to {OUT_DIR}")
