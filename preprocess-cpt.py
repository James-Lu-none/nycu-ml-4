import os
import json

INPUT_DIR = "data/IMA-Taiwan-taigi-literature"
OUTPUT_FILE = "data/cpt.jsonl"

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(INPUT_DIR, fname)
        print(f"processing: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Skipping non-list format: {fname}")
            continue

        for item in data:
            title = str(item.get("title", "")).strip()
            author = str(item.get("author", "")).strip()
            text = str(item.get("text", "")).strip()

            full_text = f"{title} {author} {text}".strip()

            if len(full_text) < 20:
                continue  # 太短的直接丟掉

            out.write(json.dumps({"text": full_text}, ensure_ascii=False) + "\n")
            count += 1

print(f"Complete: total {count} records saved to {OUTPUT_FILE}")
