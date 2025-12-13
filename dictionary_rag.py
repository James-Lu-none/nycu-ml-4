import os
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple

# -----------------------------
# 可調參數
# -----------------------------
MIN_TERM_LEN = 2

TERM_COLUMNS = [
    "漢字",
    "詞目漢字",
    "對應詞目漢字",
]

EXPLANATION_COLUMNS_PRIORITY = [
    "華語",
    "解說",
    "對應解說",
    "說明",
    "釋義",
]

EXAMPLE_EXPLANATION_COLUMNS = [
    "華語",
    "解說",
    "對應解說",
    "說明",
]
def is_rare_two_char_term(term: str, dict_dfs) -> bool:
    if len(term) != 2:
        return False

    # 必須都是漢字
    if not all('\u4e00' <= ch <= '\U0002FFFF' for ch in term):
        return False

    # 至少一個是生僻字
    if not any(is_rare_han_char(ch) for ch in term):
        return False

    # 如果本身就是正式詞條 → 不算生僻
    for df in dict_dfs:
        if term_in_df_exact(df, term):
            return False

    return True

def term_in_df_exact(df: pd.DataFrame, term: str) -> bool:
    for col in TERM_COLUMNS:
        if df[col].dtype == object:
            if not df[df[col] == term].empty:
                return True
    return False


def is_rare_han_char(ch: str) -> bool:
    if len(ch) != 1:
        return False

    cp = ord(ch)

    # CJK Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True

    # CJK Extension B ~ F
    if cp >= 0x20000:
        return True

    # CJK Compatibility Ideographs
    if 0xF900 <= cp <= 0xFAFF:
        return True

    # Private Use Area
    if 0xE000 <= cp <= 0xF8FF:
        return True

    return False

def contains_rare_chars(term: str) -> bool:
    for ch in term:
        if is_rare_han_char(ch):
            return True
    return False

def extract_rare_single_chars(text: str) -> List[str]:
    chars = set()
    for ch in text:
        if is_rare_han_char(ch):
            chars.add(ch)
    return list(chars)

def load_dictionary_csvs(dict_dir: str) -> List[pd.DataFrame]:
    dfs = []
    for fname in os.listdir(dict_dir):
        if fname.lower().endswith(".csv"):
            path = os.path.join(dict_dir, fname)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
    return dfs

def extract_candidate_terms_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> List[str]:
    chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    out = set()
    for chunk in chunks:
        L = len(chunk)
        for n in range(n_min, n_max + 1):
            for i in range(L - n + 1):
                out.add(chunk[i:i+n])
    return list(out)

def extract_candidate_terms(text: str) -> List[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{%d,}" % MIN_TERM_LEN, text)
    return list(set(terms))


def find_explanation(df: pd.DataFrame, term: str) -> Optional[str]:
    matched_row = None
    for col in TERM_COLUMNS:
        if col in df.columns:
            matched = df[df[col] == term]
            if not matched.empty:
                matched_row = matched.iloc[0]
                break

    if matched_row is None:
        return None

    for exp_col in EXPLANATION_COLUMNS_PRIORITY:
        if exp_col in df.columns:
            val = matched_row.get(exp_col)
            if isinstance(val, str) and val.strip():
                return val.strip()

    for col in df.columns:
        if col in TERM_COLUMNS:
            continue
        val = matched_row.get(col)
        if isinstance(val, str) and len(val) > 2:
            return val.strip()

    return None

def find_example_fallback(df: pd.DataFrame, term: str) -> Optional[str]:
    """
    find example sentences containing the term
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue

        # 嘗試把這個欄位當「例句欄」
        for _, row in df.iterrows():
            val = row.get(col)
            if not isinstance(val, str):
                continue

            if term in val:
                example_text = val.strip()

                # 找對應的例句解釋
                explanation = None
                for exp_col in EXAMPLE_EXPLANATION_COLUMNS:
                    if exp_col in df.columns:
                        exp_val = row.get(exp_col)
                        if isinstance(exp_val, str) and exp_val.strip():
                            explanation = exp_val.strip()
                            break

                if explanation:
                    return (
                        f"{term}：可參考下列例句用法。\n"
                        f"例：{example_text}\n"
                        f"（例句解釋：{explanation}）"
                    )
                else:
                    return (
                        f"{term}：可參考下列例句用法。\n"
                        f"例：{example_text}"
                    )

    return None
def is_trivial_explanation(term: str, exp: str) -> bool:
    exp_clean = (
        exp.replace("、", "")
           .replace("，", "")
           .replace("。", "")
           .replace(" ", "")
           .strip()
    )
    return exp_clean == term

def lookup_term_in_dicts(
    term: str,
    dict_dfs: List[pd.DataFrame],
    fallback: bool = True
) -> Optional[str]:
    # lookup for term in multiple dataframes
    for df in dict_dfs:
        exp = find_explanation(df, term)
        if exp:
            return exp
    if not fallback:
        return None
    # fallback: try to find example sentences
    for df in dict_dfs:
        example = find_example_fallback(df, term)
        if example:
            return example
    # print(f"[WARN] Term '{term}' not found in any dictionary.")
    return None

# main function
def build_dictionary_block(
    article: str,
    question: str,
    options: List[str],
    dict_dir: str,
) -> str:
    dict_dfs = load_dictionary_csvs(dict_dir)
    # originally i use MAX_TERMS_PER_QUESTION to limit number of terms so there is priority here
    # but now we just extract all possible terms
    priority_text = question + "\n" + "\n".join(options)
    priority_terms = extract_candidate_terms_ngrams(priority_text, 3, 4)
    article_terms  = extract_candidate_terms_ngrams(article, 3, 4)

    multi_terms = []
    for t in priority_terms + article_terms:
        if t not in multi_terms:
            multi_terms.append(t)
    # print("Candidate multi-char terms:", multi_terms)
    found: List[Tuple[str, str]] = []

    for term in multi_terms:
        # if contains_rare_chars(term):
        exp = lookup_term_in_dicts(term, dict_dfs, fallback=False)
        if exp and not is_trivial_explanation(term, exp):
            found.append((term, exp))

    rare_chars = extract_rare_single_chars(priority_text)
    for ch in rare_chars:
        # avoid duplicates
        if any(ch == t for t, _ in found):
            continue

        exp = lookup_term_in_dicts(ch, dict_dfs, fallback=True)
        if exp and not is_trivial_explanation(ch, exp):
            found.append((ch, exp))

    if not found:
        return ""

    block = "【詞彙補充說明（僅供理解）】\n"
    for term, exp in found:
        block += f"{term}：{exp}\n"
    print(block)

    return block.strip()