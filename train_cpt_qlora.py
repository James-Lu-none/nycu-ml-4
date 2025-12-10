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

parser = argparse.ArgumentParser()
parser.add_argument("--model_choice", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_CHOICE = args.model_choice
DATA_PATH = "data/cpt.jsonl"
OUT_DIR = "model/qwen-cpt-qlora"

os.makedirs(OUT_DIR, exist_ok=True)

model_fn = globals()[MODEL_CHOICE]
model, tokenizer = model_fn(
    base_model_path=MODEL_NAME,
    load_lora=True
)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
    )

tokenized_dataset = dataset.map(
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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
loss = trainer.state.log_history[-1]["loss"]
print("Final training loss:", loss)
timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')

OUT_DIR = OUT_DIR + f"/{timestamp}_{loss:.4f}"
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"CPT done, model saved to {OUT_DIR}")
