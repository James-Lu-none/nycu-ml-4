python3 train_cpt_qlora.py --model_choice qwen2_5_1_5b_qlora4bit --model_path Qwen/Qwen2.5-1.5B
python3 train_sft_qlora.py --model_choice qwen2_5_1_5b_qlora4bit
python3 prediction.py --model_choice qwen2_5_1_5b_qlora4bit