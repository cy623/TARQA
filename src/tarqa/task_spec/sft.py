from __future__ import annotations

from typing import Any


def render_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def _encode_completion_example(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int], list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_text = completion + (tokenizer.eos_token or "")
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if len(completion_ids) >= max_length:
        raise ValueError("Task Spec completion exceeds max_length")
    max_prompt = max(1, max_length - len(completion_ids))
    if len(prompt_ids) > max_prompt:
        raise ValueError("Task Spec prompt exceeds max_length; increase model.max_length rather than dropping schema/question")
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels, len(prompt_ids)


def tokenize_task_spec_example(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    domain_to_id: dict[str, int],
) -> dict[str, Any]:
    prompt = render_chat_prompt(tokenizer, row["system_prompt"], row["user_prompt"])
    pair_prompt = render_chat_prompt(
        tokenizer, row["system_prompt"], row.get("pair_user_prompt", row["user_prompt"])
    )
    input_ids, attention_mask, labels, prompt_length = _encode_completion_example(
        tokenizer, prompt, row["assistant_text"], max_length
    )
    pair_ids = tokenizer(pair_prompt, add_special_tokens=False)["input_ids"]
    if len(pair_ids) > max_length:
        raise ValueError("Paraphrase prompt exceeds max_length")
    pair_attention = [1] * len(pair_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": prompt_length,
        "pair_input_ids": pair_ids,
        "pair_attention_mask": pair_attention,
        "pair_prompt_length": len(pair_ids),
        "pair_mask": float(pair_prompt != prompt),
        "domain_id": domain_to_id[row["domain"]],
    }
