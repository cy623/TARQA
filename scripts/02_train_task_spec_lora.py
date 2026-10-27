#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

# from datasets import load_dataset
from datasets import Dataset, DatasetDict, Features, Value
from transformers import TrainingArguments

from tarqa.config import load_config, resolve_path
from tarqa.task_spec.collator import CompletionOnlyCollator
from tarqa.task_spec.modeling import load_lora_trainable_model, load_tokenizer
from tarqa.task_spec.sft import tokenize_task_spec_example
from tarqa.task_spec.trainer import DomainBalancedTaskSpecTrainer
# from tarqa.utils.io import read_json, write_json
from tarqa.utils.io import read_json, read_jsonl, write_json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task_spec.yaml")
    parser.add_argument("--holdout", required=True, type=str.lower)
    parser.add_argument("--stage", choices=["pretrain", "full", "few"], default="pretrain")
    parser.add_argument("--k", type=int)
    parser.add_argument("--init-adapter")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    strategy = f"few_{args.k}" if args.stage == "few" else args.stage
    processed = resolve_path(cfg["data"]["processed_root"], cfg["_config_path"])
    data_dir = (
        resolve_path(args.data_dir, cfg["_config_path"])
        if args.data_dir
        else processed / "task_spec" / f"holdout_{args.holdout}" / strategy
    )
    output_dir = (
        resolve_path(args.output_dir, cfg["_config_path"])
        if args.output_dir
        else resolve_path(train_cfg["output_root"], cfg["_config_path"])
        / f"holdout_{args.holdout}"
        / strategy
    )
    if ((output_dir / 'final' / 'adapter_model.safetensors').exists()
            or (output_dir / 'adapter_model.safetensors').exists()
            or any(output_dir.glob('checkpoint-*'))):
        raise FileExistsError('Use a fresh output directory; existing adapters/checkpoints are preserved and are not implicitly resumed')
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(data_dir / "manifest.json")
    if manifest["holdout_domain"] != args.holdout or manifest["stage"] != args.stage:
        raise ValueError("Prepared TSG data stage/holdout mismatch")
    if args.stage == "pretrain" and args.init_adapter:
        raise ValueError("Source pretraining must not initialize from a target adapter")
    if args.stage != "pretrain" and not args.init_adapter:
        raise ValueError("Target adaptation requires --init-adapter from source pretraining")
    if args.init_adapter:
        init_path = resolve_path(args.init_adapter, cfg["_config_path"])
        provenance = read_json(init_path / "training_manifest.json")
        if provenance["stage"] != "pretrain" or provenance["holdout_domain"] != args.holdout:
            raise ValueError("Wrong source TSG adapter for this held-out domain")
    else:
        init_path = None
    # Seed before fresh LoRA initialization as well as Trainer sampling.
    from tarqa.utils.seed import set_seed
    set_seed(int(train_cfg.get("seed", 42)))
    tokenizer = load_tokenizer(cfg["model"])
    model = load_lora_trainable_model(cfg["model"], cfg["lora"], adapter_path=init_path)
    # raw = load_dataset(
    #     "json",
    #     data_files={
    #         "train": str(data_dir / "train.jsonl"),
    #         "validation": str(data_dir / "validation.jsonl"),
    #     },
    # )
    features = Features({
        name: Value("string")
        for name in (
            "domain",
            "system_prompt",
            "user_prompt",
            "assistant_text",
            "pair_user_prompt",
        )
    })

    raw = DatasetDict()

    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        rows = []

        for row in read_jsonl(path):
            rows.append({
                "domain": row["domain"],
                "system_prompt": row["system_prompt"],
                "user_prompt": row["user_prompt"],
                "assistant_text": row["assistant_text"],
                "pair_user_prompt": row.get(
                    "pair_user_prompt", row["user_prompt"]
                ),
            })

        if not rows:
            raise ValueError(f"数据文件为空：{path}")

        raw[split] = Dataset.from_list(rows, features=features)
    domains = sorted(set(raw["train"]["domain"]) | set(raw["validation"]["domain"]))
    domain_to_id = {domain: idx for idx, domain in enumerate(domains)}

    def preprocess(row):
        return tokenize_task_spec_example(
            row,
            tokenizer,
            max_length=int(cfg["model"].get("max_length", 8192)),
            domain_to_id=domain_to_id,
        )

    tokenized = raw.map(preprocess, remove_columns=raw["train"].column_names)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("epochs", 3)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        seed=int(train_cfg.get("seed", 42)),
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
    )
    trainer = DomainBalancedTaskSpecTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CompletionOnlyCollator(tokenizer),
        processing_class=tokenizer,
        paraphrase_weight=float(train_cfg.get("paraphrase_consistency_weight", 0.0)),
        generation_eval_path=data_dir / "validation.jsonl",
        generation_max_length=int(cfg["model"].get("max_length", 8192)),
        generation_max_new_tokens=int(train_cfg.get("max_new_tokens", 256)),
        domain_balanced=bool(train_cfg.get("domain_balanced_sampling", True)),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    manifest["init_adapter"] = str(init_path) if init_path else None
    write_json(output_dir / "final" / "training_manifest.json", manifest)


if __name__ == "__main__":
    main()


