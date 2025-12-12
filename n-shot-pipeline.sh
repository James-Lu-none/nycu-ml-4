# exit if any command fails
set -e

python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 0
python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 3
python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 5
python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 7
python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 9
python3 prediction_few_shot.py --model_name Qwen/Qwen2.5-7B-Instruct --k 12