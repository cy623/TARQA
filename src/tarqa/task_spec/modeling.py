from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType as PeftTaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def runtime_max_memory(model_cfg):
    value = os.environ.get('TARQA_LLM_MAX_MEMORY')
    memory = json.loads(value) if value else model_cfg.get('max_memory')
    if memory is None:
        return None
    return {int(k) if str(k).isdigit() else k: v for k, v in memory.items()}


def torch_dtype_from_name(name: str | None) -> torch.dtype | str:
    if not name or name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_tokenizer(model_cfg: dict[str, Any]):
    source = model_cfg.get("tokenizer_path") or model_cfg["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_trainable_model(model_cfg: dict[str, Any], lora_cfg: dict[str, Any], adapter_path=None):
    quantization_config = None
    if bool(model_cfg.get("load_in_4bit", False)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16")),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        torch_dtype=torch_dtype_from_name(model_cfg.get("torch_dtype", "auto")),
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None,
        max_memory=runtime_max_memory(model_cfg),
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if adapter_path:
        return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    peft_config = LoraConfig(
        task_type=PeftTaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model


def load_lora_inference_model(
    model_cfg: dict[str, Any],
    adapter_path: str | Path,
):
    quantization_config = None
    if bool(model_cfg.get("load_in_4bit", False)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16")),
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        torch_dtype=torch_dtype_from_name(model_cfg.get("torch_dtype", "auto")),
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=runtime_max_memory(model_cfg),
    )
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model
