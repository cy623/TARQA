from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tarqa.task_spec.modeling import torch_dtype_from_name
from tarqa.types import CanonicalAnswer


class TemplateVerbalizer:
    def verbalize(self, question: str, answer: CanonicalAnswer) -> str:
        if answer.answer_type == "Boolean":
            prefix = "Yes" if bool(answer.value) else "No"
            if answer.probability is not None:
                return f"{prefix}. The predicted probability is {answer.probability:.4f}."
            return f"{prefix}."
        if answer.answer_type == "Scalar":
            value = answer.value
            if isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            return f"The answer is {rendered}{(' ' + answer.unit) if answer.unit else ''}."
        if answer.answer_type == "Set":
            return "The result is: " + str(answer.value)
        if answer.answer_type == "Ranking":
            return "The ranked result is: " + str(answer.value)
        return str(answer.value)


class LocalLoRAVerbalizer:
    def __init__(self, config: dict[str, Any]) -> None:
        source = config.get("tokenizer_path") or config["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=bool(config.get("local_files_only", True)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            local_files_only=bool(config.get("local_files_only", True)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=torch_dtype_from_name(config.get("torch_dtype", "auto")),
            device_map="auto",
        )
        adapter = config.get("adapter_path")
        if adapter:
            model = PeftModel.from_pretrained(model, str(Path(adapter)), is_trainable=False)
        self.model = model.eval()
        self.max_new_tokens = int(config.get("max_new_tokens", 128))

    @torch.inference_mode()
    def verbalize(self, question: str, answer: CanonicalAnswer) -> str:
        system = (
            "Convert the canonical database answer into one concise natural-language sentence. "
            "Do not change, round, omit, or invent any value."
        )
        user = f"Question: {question}\nCanonical answer: {answer.model_dump_json()}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
