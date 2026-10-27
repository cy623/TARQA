from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def _pad(self, values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(x) for x in values)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        result = torch.full((len(values), max_len), pad_value, dtype=torch.long)
        for row, seq in enumerate(values):
            result[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return result

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = self._pad([f["input_ids"] for f in features], self.tokenizer.pad_token_id)
        attention_mask = self._pad([f["attention_mask"] for f in features], 0)
        labels = self._pad([f["labels"] for f in features], -100)
        pair_input_ids = self._pad(
            [f["pair_input_ids"] for f in features], self.tokenizer.pad_token_id
        )
        pair_attention_mask = self._pad([f["pair_attention_mask"] for f in features], 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": torch.tensor([f["prompt_length"] for f in features], dtype=torch.long),
            "pair_input_ids": pair_input_ids,
            "pair_attention_mask": pair_attention_mask,
            "pair_prompt_length": torch.tensor(
                [f["pair_prompt_length"] for f in features], dtype=torch.long
            ),
            "pair_mask": torch.tensor([f["pair_mask"] for f in features], dtype=torch.float32),
            "domain_id": torch.tensor([f["domain_id"] for f in features], dtype=torch.long),
        }
