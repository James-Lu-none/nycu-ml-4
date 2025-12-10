import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from models import *
import os
import numpy as np

MODEL_PATH = "model"
BASE_MODEL_PATH = "model/qwen-cpt-qlora"
MODEL_CHOICE = "qwen2_5_1_5b_qlora4bit"
TRAIN_FILE = "data/sft.jsonl"
OUT_DIR = "model/qwen-sft-qlora"

os.makedirs(OUT_DIR, exist_ok=True)

model_fn = globals()[MODEL_CHOICE]
model, tokenizer = model_fn(
    base_model_path=BASE_MODEL_PATH,
    load_lora=True
)

dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

def format_prompt(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    answer = example["response"]
    full_text = prompt + answer

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
    )

    labels = tokenized["input_ids"].copy()

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    labels[:-len(answer_ids)] = [-100] * (len(labels) - len(answer_ids))

    tokenized["labels"] = labels
    return tokenized


tokenized_dataset = dataset.map(
    format_prompt,
    remove_columns=dataset.column_names,
)


training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
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
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
loss = trainer.state.log_history[-1]["loss"]
print("Final training loss:", loss)
timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')

OUT_DIR = OUT_DIR + f"/{timestamp}_{loss:.4f}"
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("SFT done, model saved to", OUT_DIR)