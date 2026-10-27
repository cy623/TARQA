from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tarqa.task_spec.modeling import torch_dtype_from_name


SQL_SYSTEM_PROMPT = """You are a precise Text-to-SQL model.
Return exactly one read-only DuckDB SQL query and no explanation.
Use only tables and columns listed in the schema. Preserve the full semantics of the question.
"""


def _extract_sql(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    start = re.search(r"\b(WITH|SELECT)\b", text, flags=re.I)
    if start:
        text = text[start.start() :]
    if ";" in text:
        text = text[: text.find(";") + 1]
    return text.strip()


class LocalTextToSQLModel:
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
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.max_input_length = int(config.get("max_input_length", 8192))
        self.num_beams = int(config.get("num_beams", 1))

    def _prompt(self, question: str, schema_text: str, error: str | None = None, sql: str | None = None) -> str:
        user = f"Schema:\n{schema_text}\n\nQuestion:\n{question}"
        if error and sql:
            user += f"\n\nPrevious SQL:\n{sql}\n\nDatabase error:\n{error}\n\nReturn a corrected SQL query."
        messages = [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System:\n{SQL_SYSTEM_PROMPT}\n\nUser:\n{user}\n\nAssistant:\n"

    @torch.inference_mode()
    def generate(self, question: str, schema_text: str, *, error: str | None = None, sql: str | None = None) -> str:
        prompt = self._prompt(question, schema_text, error=error, sql=sql)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return _extract_sql(self.tokenizer.decode(generated, skip_special_tokens=True))
