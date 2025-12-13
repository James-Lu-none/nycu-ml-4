import json
import csv
import faiss
import torch
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--test_file", type=str, default="data/1001-question-v3.jsonl")
args = parser.parse_args()

TRAIN_QA = "data/sft.jsonl"
TEST_FILE = args.test_file
MODEL_NAME = args.model_name
K = args.k
timestamp = np.datetime64('now').astype('str').replace(':', '-').replace(' ', '_')
MODEL_NAME_SAFE = MODEL_NAME.replace("/", "-")
OUT_CSV = f"output/{K}-shot_{MODEL_NAME_SAFE}_{timestamp}.csv"
print("few-shot inference will be saved to:", OUT_CSV)

device = "cuda" if torch.cuda.is_available() else "cpu"

qa_inputs = []
qa_answers = []

with open(TRAIN_QA, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj["instruction"] + "\n" + obj["input"]
        qa_inputs.append(text)
        qa_answers.append(obj["response"])

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

qa_emb = embedder.encode(
    qa_inputs,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

index = faiss.IndexFlatIP(qa_emb.shape[1])
index.add(qa_emb)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    # load_in_4bit=True
)
model.eval()

results = []

with open(TEST_FILE, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qid = item["id"]
        query_text = item["instruction"] + "\n" + item["input"]
        pred = ""
        if K == 0:
            prompt = (
                "你是一個台灣閩南語的閱讀理解選擇題專家，請根據使用者提供的文章內容，選出最適合的答案。\n"
                "請根據以下文章內容，選出最適合的答案，只輸出正確選項的數字（1、2、3、4），不要輸出任何解釋。\n\n"
                f"文章內容：\n{item['input']}\n\n答案："
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    temperature=0.0
                )
            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            pred = next((c for c in ["1", "2", "3", "4"] if c in gen), "1")
        else:
            q_emb = embedder.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            _, idxs = index.search(q_emb, K)

            exemplars = []
            for i in idxs[0]:
                exemplars.append(
                    f"{qa_inputs[i]}{qa_answers[i]}"
                )

            prompt = (
                "你是一個台灣閩南語的閱讀理解選擇題專家，請根據使用者提供的文章內容，選出最適合的答案。\n"
                "以下是台文閱讀理解選擇題範例，請依照範例作答，只輸出正確選項的數字（1、2、3、4），不要輸出任何解釋。\n\n"
            )
            for ex in exemplars:
                prompt += ex + "\n\n"

            prompt += (
                "【以上為範例，現在請回答以下題目】\n"
                f"{item['instruction']}\n{item['input']}\n答案："
            )
            print("Prompt:\n", prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    temperature=0.0
                )

            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            pred = next((c for c in ["1", "2", "3", "4"] if c in gen), "1")

        results.append([qid, pred])
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "answer"])
            writer.writerows(results)

print("few-shot inference done:", OUT_CSV)