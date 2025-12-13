#!/bin/bash
set -e

model=Qwen/Qwen2.5-7B-Instruct

# python3 prediction_few_shot.py --model_name $model --k 0
# python3 prediction_few_shot.py --model_name $model --k 3
# python3 prediction_few_shot.py --model_name $model --k 5
# python3 prediction_few_shot.py --model_name $model --k 7
# python3 prediction_few_shot.py --model_name $model --k 9

python3 prediction_few_shot.py --model_name $model --k 0 --test_file data/1001-question-v3-rag.jsonl
python3 prediction_few_shot.py --model_name $model --k 3 --test_file data/1001-question-v3-rag.jsonl
python3 prediction_few_shot.py --model_name $model --k 5 --test_file data/1001-question-v3-rag.jsonl
python3 prediction_few_shot.py --model_name $model --k 7 --test_file data/1001-question-v3-rag.jsonl
python3 prediction_few_shot.py --model_name $model --k 9 --test_file data/1001-question-v3-rag.jsonl