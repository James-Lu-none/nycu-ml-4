from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch


def _prepare_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def llama3_8b():
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = _prepare_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return tokenizer, model


def mistral_7b():
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = _prepare_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return tokenizer, model


def llama3_8b_4bit_lora(r=16, alpha=32, dropout=0.05):
    model_name = "meta-llama/Meta-Llama-3-8B"
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = _prepare_tokenizer(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        use_cache=False,
        torch_dtype="auto",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model
