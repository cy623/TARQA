from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tarqa.task_spec.anonymization import TableAnonymization
from tarqa.task_spec.modeling import load_lora_inference_model, load_tokenizer
from tarqa.task_spec.sft import render_chat_prompt
from tarqa.types import TaskSpec
from tarqa.utils.text import extract_json_object


@dataclass
class GeneratedTaskSpec:
    spec: TaskSpec
    raw_text: str
    dependency_errors: list[str]


class TaskSpecOutputError(ValueError):
    """Invalid model text, distinct from a model/runtime failure."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


class LocalTaskSpecGenerator:
    def __init__(self, model_cfg: dict[str, Any], adapter_path: str | Path) -> None:
        self.model_cfg = dict(model_cfg)
        self.adapter_path = Path(adapter_path).resolve()
        self.tokenizer = load_tokenizer(model_cfg)
        self.model = load_lora_inference_model(model_cfg, adapter_path)
        self.max_new_tokens = int(model_cfg.get("max_new_tokens", 256))
        self.max_input_length = int(model_cfg.get("max_length", 8192))

    @classmethod
    def from_loaded_model(cls, model, tokenizer, *, max_length=8192, max_new_tokens=256):
        """Reuse the Trainer's current weights; do not load another base model."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.max_input_length = int(max_length)
        instance.max_new_tokens = int(max_new_tokens)
        return instance

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        anonymization: TableAnonymization | None = None,
        max_new_tokens: int | None = None,
    ) -> GeneratedTaskSpec:
        prompt = render_chat_prompt(self.tokenizer, system_prompt, user_prompt)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False, truncation=False
        )
        if inputs["input_ids"].shape[1] > self.max_input_length:
            raise ValueError("Task Spec prompt exceeds max_length; refusing to truncate schema/question.")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        try:
            payload = extract_json_object(raw)
            if set(payload) != {"relevant_tables", "reference_time", "task_type", "track"}:
                raise ValueError("Generated Task Spec must contain exactly the four declared fields")
            spec = TaskSpec.model_validate(payload)
        except (ValueError, TypeError) as exc:
            raise TaskSpecOutputError(str(exc), raw) from exc
        if anonymization:
            spec = anonymization.restore_spec(spec)
        return GeneratedTaskSpec(spec=spec, raw_text=raw, dependency_errors=spec.validate_dependencies())
