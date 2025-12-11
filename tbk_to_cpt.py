import os
import json
import re
import argparse


def extract_hl_text(content: str) -> str:
    """
    從 <HL> ... </HL> 中擷取正文
    """
    match = re.search(r"<HL>(.*?)</HL>", content, re.S)
    if not match:
        return ""

    text = match.group(1)

    # 基本清理
    text = text.replace("\u3000", " ")   # 全形空白
    text = re.sub(r"\n{2,}", "\n", text) # 多餘空行
    text = text.strip()

    return text


def split_long_text(text, max_chars=800):
    """
    將長文本切成多段（適配你目前 max_length=1024）
    800 字是台文混寫下非常安全的長度
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end].strip()

        if len(chunk) > 50:   # 過短的切片直接丟掉
            chunks.append(chunk)

        start = end

    return chunks


def process_tbk_file(path: str):
    """
    讀取單一 tbk（Big5）並抽取 + 切分 CPT 用文本
    回傳：list[str]
    """
    try:
        with open(path, "r", encoding="big5", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return []

    text = extract_hl_text(content)
    if len(text) < 50:
        return []

    chunks = split_long_text(text, max_chars=800)
    return chunks


def collect_tbk(root_dir: str, output_jsonl: str):
    samples = []
    total_files = 0
    used_files = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".tbk"):
                total_files += 1
                path = os.path.join(root, fname)

                chunks = process_tbk_file(path)
                if not chunks:
                    continue

                for chunk in chunks:
                    samples.append({
                        "text": chunk
                    })

                used_files += 1

    print(f"Total .tbk files found : {total_files}")
    print(f"Files used as CPT     : {used_files}")
    print(f"Total CPT samples    : {len(samples)}")

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"CPT jsonl saved to: {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="nmtl_2006_dadwt/原始文字檔/ss",
        help="TBK root directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nmtl_2006_dadwt_cpt.jsonl"
    )
    args = parser.parse_args()

    collect_tbk(args.input_dir, args.output)
