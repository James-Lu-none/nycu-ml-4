Description
Title: Taibun (Taiwanese) Reading Comprehension Test for AI
Questions
Given the advancements in Large Language Models, what level of improvement can we anticipate in computer Taibun (Taiwanese) reading comprehension capabilities?

Objective
Computer Taiwanese Reading Comprehension Test (Multiple Choice): Aim to train a large language model to comprehend articles written in Taibun (Taiwanese) and respond accurately to associated questions.

Example
尹雪艷誠迷人，出名了後，難免予人嫉妒。有的仝徒的姊妹看了紅目，就四界放話：尹雪艷的八字帶重煞，犯白虎，人若磕著，較輕的敗家，若重的人亡。若是這點顛倒加添伊的神秘感，予伊的名聲愈響亮。請問根據這段文章，後壁佗一个選項佮「尹雪豔八字犯重煞」無關係？1. 增加尹雪豔神秘感, 2. 同行因為怨妒來放話, 3. 會予接近伊的人家敗人亡, 4. 忠孝仁愛信義和平。

Background: Formosa Speech Grand Challenge for Mandarin
Between 2018 and 2020, the Ministry of Science and Technology hosted two iterations of the "Formosa Speech Grand Challenge" competition. The overarching theme spotlighted the rapid transformations that Artificial Intelligence (AI) instills in global industries, economies, and social paradigms. Within the expansive domain of AI research and development, voice applications emerge as critically significant. This is attributed to the fact that voice interactions offer the most intuitive and human-oriented channel for human-machine interactions. Moreover, the cornerstone of AI's intelligence is embedded in semantic understanding technologies. Acknowledging these insights, and to drive innovation while navigating the intricacies of voice AI, the Ministry unveiled Taiwan's inaugural "Formosa Speech Grand Challenge" for Mandarin.

Fact
In the final multiple-choice segment of the 2020 competition, participants achieved a pinnacle accuracy rate of 53.7% for the Traditional Chinese Reading Comprehension Test.

References
Yi-Chih Huang, Yu-Lun Hsieh, Yi-Yu Lin, Teo Lin Hui, Hung-Ying Chu, Wen-Lian Hsu, FLUD: Expert-curated large-scale machine comprehension dataset with advanced reasoning strategies, Asian Research Policy, Volume 12, Issue 1, 2021, pp. 30-37, URL: <https://www.kistep.re.kr/arpIssue.es?act=content_view&list_no=200&act=content_view&mid=a20802000000>
建議做法：

先用IMA Taiwan Tongues 語料庫的台文文章做CPT，讓模型可以懂台語用字遣詞，再用SFT訓練模型回答台語問題：

1. CPT（Continual Pre-Training）
訓練資料：
Taiwan Tongues 語料庫說明： <https://tt.ima.org.tw/>
此語料庫請在Huggingface上直接線上申請授權： <https://huggingface.co/collections/IMA-Taiwan/taigi-datasets> 。 取得授權後可直接在程式中以datasets.load_dataset()下載。
程式範例： Gemma 3 微調與推論（Unsloth） <https://unsloth.ai/blog/gemma3>
2. SFT（Supervised Fine-Tuning）
訓練資料： Kaggle「Taiwanese Reading Comprehension Test for LLMs」 （從競賽頁面 Data 分頁下載）
程式範例： Gemma 3 微調與推論（Unsloth） <https://unsloth.ai/blog/gemma3>
競賽規則：

訓練模型時不能使用任何測試集資訊。
不能使用任何雲端大語言模型進行推論，或是產生測試集答案。
競賽成績：

以Private Leaderboard排名取倒數後打成績。
Private Leaderboard前三名需上台報告做法，每人至少報告15分鐘，做法合理才承認其成績。
報告繳交項目（至Github Classroom）：

程式碼，腳本，設定檔，模型，訓練資料，訓練log檔與最終測試集推論結果。其中模型與訓練資料若太大，請改放到Hugging Face（一樣用學號開Hugging Face帳號或是暱稱，報告中要有連結）。
最好在訓練時，把wandb服務開起來，他會記錄log檔，其中很多資訊可以查看。
技術，實驗，訓練過程與心得報告。
報告成績視內容完整度而定。
*參考資料：

<https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl>
<https://huggingface.co/blog/lmassaron/gemma-grpo>
<https://anukriti-ranjan.medium.com/preference-tuning-llms-ppo-dpo-grpo-a-simple-guide-135765c87090>
參考做法：

CPT: Mistral_v0.3_(7B)-CPT.ipynb
SFT: Llama3.1_(8B)-Alpaca.ipynb
GRPO: Llama3.1_(8B)-GRPO.ipynb


## Quick Start Workflow

### data preprocessing
```bash
# QA pair extraction (train/validation set)
python3 preprocess_sft.py
# QA pair extraction (test set)
python3 preprocess_test.py
# CPT data preprocessing
python3 preprocess_cpt.py

# model continual pre-training
python3 train_cpt_qlora.py --model_choice qwen2_5_1_5b_qlora4bit --model_path Qwen/Qwen2.5-1.5B
# model continual pre-training (resume from latest cpt model)
python3 train_cpt_qlora.py --model_choice qwen2_5_1_5b_qlora4bit

# model supervised fine-tuning
python3 train_sft_qlora.py --model_choice qwen2_5_1_5b_qlora4bit --model_path model_cpt/qwen2_5_1_5b_qlora4bit/{timestamp}
# model supervised fine-tuning (resume from latest cpt model)
python3 train_sft_qlora.py --model_choice qwen2_5_1_5b_qlora4bit

# model prediction
python3 prediction.py --model_choice qwen2_5_1_5b_qlora4bit --model_path model_sft/qwen2_5_1_5b_qlora4bit/{timestamp}
# model prediction (use latest sft model)
python3 prediction.py --model_choice qwen2_5_1_5b_qlora4bit
```

## data source

- 語料匯入專案 <https://github.com/Taiwanese-Corpus/hue7jip8>

### Continual Pre-Training (CPT) Corpora

- 台語文數位典藏資料庫 <https://github.com/Taiwanese-Corpus/nmtl_2006_dadwt> 劇本、歌詩、散文、小說
- 台語文語料庫蒐集及語料庫為本台語書面語音節詞頻統計 <https://taiwanese-corpus.github.io/Ungian_2005_guliau-supin>
- 教育部臺灣閩南語字詞頻調查工作 <https://taiwanese-corpus.github.io/Ungian_2009_KIPsupin>
- 臺語國校仔課本 https://taiwanese-corpus.github.io/kok4hau7-kho3pun2

```bash
git clone --depth 1 https://github.com/Taiwanese-Corpus/nmtl_2006_dadwt
```

### Supervised Fine-Tuning (SFT) Dataset

- 

### Qwen-Qwen2.5-7B-Instruct n-shot Results
[0.78666](history/0-shot_Qwen-Qwen2.5-7B-Instruct_2025-12-13T00-15-33.csv)
[0.79666](history/3-shot_Qwen-Qwen2.5-7B-Instruct_2025-12-13T00-32-32.csv)
[0.80333](history/5-shot_Qwen-Qwen2.5-7B-Instruct_2025-12-13T00-51-19.csv)
[0.79666](history/7-shot_Qwen-Qwen2.5-7B-Instruct_2025-12-13T01-11-00.csv)
[0.80666](history/9-shot_Qwen-Qwen2.5-7B-Instruct_2025-12-13T01-32-32.csv)