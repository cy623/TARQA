from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import random
import time
from typing import Any, Iterator, Sized

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer

from tarqa.eval.task_spec_eval import evaluate_task_spec_generator
from tarqa.task_spec.inference import LocalTaskSpecGenerator
from tarqa.utils.io import read_jsonl, write_json


@contextmanager
def _generation_state(model):
    """Restore module modes and RNG states even if generation fails."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            with torch.inference_mode():
                yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class DistributedWeightedSampler(Sampler[int]):
    """Domain-balanced replacement sampler that also works under DDP."""

    def __init__(self, weights: torch.Tensor, seed: int = 0) -> None:
        if len(weights) == 0:
            raise ValueError("Sampler weights cannot be empty.")
        self.weights = weights.double()
        self.seed = seed
        self.epoch = 0
        self.rank = 0
        self.world_size = 1
        self.num_samples = math.ceil(len(weights) / self.world_size)
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DomainBalancedTaskSpecTrainer(Trainer):
    def __init__(
        self, *args: Any, paraphrase_weight: float = 0.0, domain_balanced: bool = True,
        generation_eval_path=None, generation_max_length=8192,
        generation_max_new_tokens=256, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.paraphrase_weight = paraphrase_weight
        self.domain_balanced = domain_balanced
        # Optional so existing Trainer callers and smoke tests remain valid.
        self.generation_eval_path = Path(generation_eval_path) if generation_eval_path is not None else None
        self.generation_max_length = int(generation_max_length)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        if self.generation_eval_path is not None:
            if (
                self.accelerator.num_processes != 1
                or getattr(self, "is_fsdp_enabled", False)
                or getattr(self, "is_deepspeed_enabled", False)
            ):
                raise ValueError("Epoch generation currently supports a single training process.")
            if self.generation_max_length <= 0 or self.generation_max_new_tokens <= 0:
                raise ValueError("Generation length limits must be positive.")
            rows = list(read_jsonl(self.generation_eval_path))
            if not rows or self.eval_dataset is None or len(rows) != len(self.eval_dataset):
                raise ValueError("Generation and loss validation sets must have the same nonzero size.")
            if any(row.get("qa_split", "validation") != "validation" for row in rows):
                raise ValueError("Epoch generation must use validation QA only.")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.generation_eval_path is None:
            return super().evaluate(
                eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            raise ValueError("Generation evaluation is bound to its configured validation dataset.")
        metrics = super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        model = self.accelerator.unwrap_model(self.model)
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            tokenizer = self.tokenizer
        generator = LocalTaskSpecGenerator.from_loaded_model(
            model, tokenizer, max_length=self.generation_max_length,
            max_new_tokens=self.generation_max_new_tokens,
        )
        output = Path(self.args.output_dir) / "epoch_validation" / f"step_{self.state.global_step:06d}"
        start = time.perf_counter()
        with _generation_state(model):
            scores = evaluate_task_spec_generator(
                generator, self.generation_eval_path, output,
                show_progress=not self.args.disable_tqdm,
            )
        scores["generation_runtime_seconds"] = time.perf_counter() - start
        metrics.update({
            f"{metric_key_prefix}_{key}": value
            for key, value in scores.items() if isinstance(value, (int, float))
        })
        self.log(dict(metrics))
        write_json(output / "metrics.json", {
            **scores,
            "eval_loss": metrics.get(f"{metric_key_prefix}_loss"),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "validation_file": str(self.generation_eval_path.resolve()),
        })
        return metrics

    def _domain_weights(self) -> torch.Tensor:
        domain_ids = list(self.train_dataset["domain_id"])
        counts: dict[int, int] = {}
        for item in domain_ids:
            counts[int(item)] = counts.get(int(item), 0) + 1
        return torch.tensor([1.0 / counts[int(item)] for item in domain_ids], dtype=torch.double)

    def get_train_dataloader(self) -> DataLoader:
        if not self.domain_balanced:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")
        sampler = DistributedWeightedSampler(self._domain_weights(), seed=int(self.args.seed))
        loader = DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=(
                self.args.dataloader_persistent_workers
                and self.args.dataloader_num_workers > 0
            ),
        )

        return self.accelerator.prepare(loader)

    @staticmethod
    def _pool_prompt(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        indices = torch.clamp(lengths - 1, min=0, max=hidden.shape[1] - 1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, indices]

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        pair_input_ids = inputs.pop("pair_input_ids")
        pair_attention_mask = inputs.pop("pair_attention_mask")
        pair_prompt_length = inputs.pop("pair_prompt_length")
        pair_mask = inputs.pop("pair_mask")
        prompt_length = inputs.pop("prompt_length")
        inputs.pop("domain_id", None)

        outputs = model(**inputs, output_hidden_states=self.paraphrase_weight > 0)
        loss = outputs.loss

        if self.paraphrase_weight > 0 and torch.any(pair_mask > 0):
            pair_outputs = model(
                input_ids=pair_input_ids,
                attention_mask=pair_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            main_repr = self._pool_prompt(outputs.hidden_states[-1], prompt_length)
            pair_repr = self._pool_prompt(pair_outputs.hidden_states[-1], pair_prompt_length)
            per_row = 1.0 - F.cosine_similarity(main_repr.float(), pair_repr.float(), dim=-1)
            consistency = (per_row * pair_mask.to(per_row.device)).sum() / pair_mask.sum().clamp_min(1)
            loss = loss + self.paraphrase_weight * consistency

        return (loss, outputs) if return_outputs else loss
