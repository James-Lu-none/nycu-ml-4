import os
import pandas as pd
from typing import List
DICT_DIR = "data/dictionaries"
CLEANED_DIR = "data/dictionaries_cleaned"
REMOVE_COLUMNS = ["義項id","對應義項id","詞目id", "對應詞目id", "例句順序", "音檔檔名", "羅馬字", "詞目編號"]

os.makedirs(CLEANED_DIR, exist_ok=True)
for fname in os.listdir(DICT_DIR):
    if fname.lower().endswith(".csv"):
        fpath = os.path.join(DICT_DIR, fname)
        df = pd.read_csv(fpath)

        # remove unwanted columns
        for col in REMOVE_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        cleaned_fpath = os.path.join(CLEANED_DIR, fname)
        df.to_csv(cleaned_fpath, index=False)
        print(f"Cleaned and saved: {cleaned_fpath}")