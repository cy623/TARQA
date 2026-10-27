"""Two-stage supervised adaptation with validation-only checkpoint selection.

This replaces the upstream training drivers, not the RESDSQL architecture.
Ranker: focal loss, best validation loss. T5: skeleton+SQL teacher forcing,
best validation execution accuracy (validation loss breaks ties). Tests are
not loaded in this module. All optimizer steps use real PyTorch autograd.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from tarqa.reasoning.executor import DuckDBReadOnlyExecutor
from tarqa.reasoning.factory import resolve_reasoning_config
from tarqa.reasoning.resdsql.data import load_training_data
from tarqa.reasoning.resdsql.evaluation import evaluate_ranked
from tarqa.reasoning.resdsql.formatting import ranked_input
from tarqa.reasoning.resdsql.model import RESDSQLModel, encode_texts
from tarqa.reasoning.resdsql.protocol import validate_checkpoint
from tarqa.reasoning.resdsql.ranker import focal_loss, save_ranker
from tarqa.utils.io import write_json


def _item_metrics(labels, probs):
    if not labels:
        return {"f1": None, "auc": None}
    return dict(f1=float(f1_score(labels, np.asarray(probs) >= 0.5, zero_division=0)),
                auc=float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None)


def _chunk_loss(outputs, row, selection, cfg):
    ts = [row["table_labels"][tid] for tid, _ in selection]
    cs = [row["column_labels"][tid][cid] for tid, cids in selection for cid in cids]
    tl, cl = outputs["batch_table_name_cls_logits"][0], outputs["batch_column_info_cls_logits"][0]
    return (focal_loss(tl, ts, cfg.get("alpha", 0.75), cfg.get("gamma", 2)) +
            focal_loss(cl, cs, cfg.get("alpha", 0.75), cfg.get("gamma", 2))), (ts, cs, tl, cl)


@torch.inference_mode()
def _validate_ranker(ranker, rows, chunks, cfg):
    ranker.model.eval()
    losses, table_y, table_p, column_y, column_p = [], [], [], [], []
    for row, selections in zip(rows, chunks):
        per_question = []
        for selection in selections:
            out = ranker.forward_chunk(row["payload"], selection)
            loss, (ty, cy, tl, cl) = _chunk_loss(out, row, selection, cfg)
            per_question.append(float(loss))
            table_y.extend(ty); column_y.extend(cy)
            table_p.extend(torch.softmax(tl.float(), -1)[:, 1].cpu().tolist())
            column_p.extend(torch.softmax(cl.float(), -1)[:, 1].cpu().tolist())
        losses.append(float(np.mean(per_question)))
    return dict(validation_loss=float(np.mean(losses)), tables=_item_metrics(table_y, table_p),
                columns=_item_metrics(column_y, column_p),
                metric_scope="schema items per ranker chunk; wide-table names may recur")


def _optimizer(model, cfg, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-5)),
                                  weight_decay=float(cfg.get("weight_decay", 0.01)))
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(steps*float(cfg.get("warmup_ratio", 0.05))), num_training_steps=steps)
    return optimizer, scheduler


def fit_ranker(ranker, train, validation, cfg, output, *, seed=42):
    output = Path(output)
    epochs, accumulation = int(cfg.get("epochs", 3)), int(cfg.get("gradient_accumulation_steps", 8))
    if epochs < 1 or accumulation < 1:
        raise ValueError("Ranker epochs and accumulation must be positive")
    train_chunks = [ranker.chunks(r["payload"]) for r in train]
    val_chunks = [ranker.chunks(r["payload"]) for r in validation]
    optimizer, scheduler = _optimizer(ranker.model, cfg, epochs*math.ceil(len(train)/accumulation))
    best, history = math.inf, []
    for epoch in range(1, epochs+1):
        ranker.model.train()
        order = list(range(len(train)))
        random.Random(seed+epoch).shuffle(order)
        losses = []
        progress = tqdm(range(0, len(order), accumulation), desc=f"Ranker epoch {epoch}")
        for start in progress:
            group = order[start:start+accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                row, selections = train[index], train_chunks[index]
                question_loss = 0.
                for selection in selections:
                    out = ranker.forward_chunk(row["payload"], selection)
                    loss = _chunk_loss(out, row, selection, cfg)[0]
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite ranker loss: {row['uid']}")
                    (loss / (len(group)*len(selections))).backward()
                    question_loss += float(loss.detach())/len(selections)
                losses.append(question_loss)
            torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step(); scheduler.step()
            progress.set_postfix(loss=float(np.mean(losses[-accumulation:])))
        metrics = dict(epoch=epoch, train_loss=float(np.mean(losses)),
                       **_validate_ranker(ranker, validation, val_chunks, cfg))
        history.append(metrics)
        if metrics["validation_loss"] < best:
            best = metrics["validation_loss"]
            save_ranker(ranker.model, ranker.tokenizer, output / "best/ranker")
            write_json(output / "best/ranker_selection.json", metrics)
        write_json(output / "ranker_history.json", history)
        print("[resdsql/ranker/validation] " + json.dumps(metrics), flush=True)
    from safetensors.torch import load_file
    ranker.model.load_state_dict(load_file(str(output / "best/ranker/model.safetensors")), strict=True)
    ranker.model.eval()
    return history


def _autocast(model, enabled):
    device = next(model.parameters()).device
    if enabled and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16; set training.generator.bf16=false")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _generator_batch(tokenizer, rows, texts, indices, model_cfg, device):
    inputs = encode_texts(tokenizer, [texts[i] for i in indices], int(model_cfg.get("max_input_length", 512)))
    targets = encode_texts(tokenizer, [rows[i]["output_sequence"] for i in indices],
                           int(model_cfg.get("max_new_tokens", 512)), kind="supervised target")
    labels = targets["input_ids"].masked_fill(targets["attention_mask"] == 0, -100)
    return {**{k: v.to(device) for k, v in inputs.items()}, "labels": labels.to(device)}


@torch.inference_mode()
def _validation_loss(model, rows, texts, cfg):
    model.model.eval()
    bs = int(cfg.get("batch_size", 2))
    device = next(model.model.parameters()).device
    total, tokens = 0., 0
    for start in range(0, len(rows), bs):
        indices = list(range(start, min(start+bs, len(rows))))
        batch = _generator_batch(model.tokenizer, rows, texts, indices, model.config, device)
        n = int((batch["labels"] != -100).sum())
        with _autocast(model.model, cfg.get("bf16", True)):
            loss = model.model(**batch).loss
        total += float(loss)*n; tokens += n
    return total/max(tokens, 1)


def fit_generator(model, train, validation, cfg, manifest, output, *, seed=42):
    output = Path(output)
    epochs, bs = int(cfg.get("epochs", 3)), int(cfg.get("batch_size", 2))
    accumulation = int(cfg.get("gradient_accumulation_steps", 8))
    if min(epochs, bs, accumulation) < 1:
        raise ValueError("Generator epochs, batch size and accumulation must be positive")
    texts = {}
    for role, rows in (("train", train), ("validation", validation)):
        texts[role] = []
        for row in tqdm(rows, desc=f"RESDSQL rank {role}"):
            scores = model.ranker.score(row["payload"])
            text, _ = ranked_input(row["payload"], scores,
                model.config.get("topk_tables", 4), model.config.get("topk_columns", 5),
                gold=row if role == "train" else None)
            # Check lengths before any optimizer updates, never silently drop QA.
            encode_texts(model.tokenizer, [text], int(model.config.get("max_input_length", 512)))
            encode_texts(model.tokenizer, [row["output_sequence"]],
                         int(model.config.get("max_new_tokens", 512)), kind="supervised target")
            texts[role].append(text)
    paths = {d: p["database"] for d, p in manifest["source_paths"].items()}
    timeout = float(cfg.get("execution_timeout_seconds", 120))
    gold = {r["uid"]: DuckDBReadOnlyExecutor(paths[r["domain"]], timeout_seconds=timeout).execute(r["golden_sql"])[1]
            for r in validation}
    optimizer, scheduler = _optimizer(model.model, cfg, epochs*math.ceil(math.ceil(len(train)/bs)/accumulation))
    device = next(model.model.parameters()).device
    if cfg.get("gradient_checkpointing", True):
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    best_key, history = (-math.inf, -math.inf), []
    for epoch in range(1, epochs+1):
        model.model.train(); model.model.config.use_cache = False
        order = list(range(len(train)))
        random.Random(seed+epoch).shuffle(order)
        batches = [order[start:start+bs] for start in range(0, len(order), bs)]
        train_loss, train_tokens = 0., 0
        progress = tqdm(range(0, len(batches), accumulation), desc=f"T5 epoch {epoch}")
        for start in progress:
            group = batches[start:start+accumulation]
            optimizer.zero_grad(set_to_none=True)
            for indices in group:
                batch = _generator_batch(model.tokenizer, train, texts["train"], indices, model.config, device)
                with _autocast(model.model, cfg.get("bf16", True)):
                    loss = model.model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite T5 loss")
                (loss/len(group)).backward()
                n = int((batch["labels"] != -100).sum())
                train_loss += float(loss.detach())*n; train_tokens += n
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step(); scheduler.step()
            progress.set_postfix(loss=train_loss/max(train_tokens, 1))
        model.model.eval(); model.model.config.use_cache = True
        val_loss = _validation_loss(model, validation, texts["validation"], cfg)
        metrics, predictions = evaluate_ranked(validation, texts["validation"], model, paths,
                                               timeout_seconds=timeout, gold_cache=gold)
        metrics.update(epoch=epoch, train_loss=train_loss/max(train_tokens, 1), validation_loss=val_loss)
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite T5 validation loss")
        history.append(metrics)
        key = (metrics["execution_accuracy"], -val_loss)
        if key > best_key:
            best_key = key
            model.model.save_pretrained(output / "best/generator", safe_serialization=True)
            model.tokenizer.save_pretrained(output / "best/generator")
            write_json(output / "best/generator_selection.json", metrics)
            write_json(output / "best/validation_predictions.json", predictions)
        write_json(output / "generator_history.json", history)
        print("[resdsql/generator/validation] " + json.dumps(metrics), flush=True)
    return history


def train_fold(config, data_dir, output, holdout, *, stage="pretrain", source_checkpoint=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}; choose a new --output-dir")
    train, validation, manifest = load_training_data(data_dir, config, holdout, stage)
    if not train and stage != "few":
        raise ValueError("Empty RESDSQL training data")
    if train and not validation:
        raise ValueError("Validation data is required")
    if stage in {"full", "few"}:
        if source_checkpoint is None:
            raise ValueError("Target adaptation requires --source-checkpoint from six-source pretraining")
        source = validate_checkpoint(source_checkpoint, holdout, "pretrain")
        for domain in manifest["train_domains"]:
            if source.get("release_reports", {}).get(domain) != manifest["release_reports"].get(domain):
                raise ValueError("Adaptation release differs from the source fold snapshot")
        runtime = resolve_reasoning_config(config, source_checkpoint)
        # Reuse both learned modules while honoring this adaptation run's
        # explicit input/generation budgets. The resulting contract is saved.
        runtime["model"].update({key: value for key, value in config["model"].items()
            if key not in {"model_path", "ranker_path", "tokenizer_path", "adapter_path", "backend"}})
    else:
        if source_checkpoint is not None:
            raise ValueError("Use configured official checkpoints to start source pretraining")
        runtime = resolve_reasoning_config(config)
    # Training labels are normalized DuckDB SQL, irrespective of official SQLite initialization.
    runtime["model"]["sql_dialect"] = "duckdb"
    runtime["model"]["torch_dtype"] = "float32"
    training = config["training"]
    seed = int(training.get("seed", 42)); set_seed(seed)
    model = RESDSQLModel(runtime["model"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", config)
    result = deepcopy(manifest)
    result.update(source_checkpoint=str(source_checkpoint) if source_checkpoint else None,
                  selection_policy="ranker: min validation focal loss; generator: max validation execution accuracy, then min validation loss",
                  training_complete=False, training_skipped_no_reasoning_questions=not bool(train))
    write_json(output / "run_manifest.json", result)
    try:
        if train:
            fit_ranker(model.ranker, train, validation, training.get("ranker", {}), output, seed=seed)
            fit_generator(model, train, validation, training.get("generator", {}), manifest, output, seed=seed)
        else:
            # k is a shared QA budget; do not silently draw more Reasoning examples.
            save_ranker(model.ranker.model, model.ranker.tokenizer, output / "best/ranker")
            model.model.save_pretrained(output / "best/generator", safe_serialization=True)
            model.tokenizer.save_pretrained(output / "best/generator")
            write_json(output / "ranker_history.json", [])
            write_json(output / "generator_history.json", [])
        saved = dict(runtime["model"], backend="resdsql", model_path="generator", ranker_path="ranker", device="auto")
        saved.pop("adapter_path", None); saved.pop("tokenizer_path", None)
        write_json(output / "best/executor_config.json", saved)
        result["training_complete"] = True
        result["best_checkpoint"] = "best"
        write_json(output / "best/training_manifest.json", result)
        write_json(output / "run_manifest.json", result)
    finally:
        model.close()
    return result
