#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import json

from tarqa.config import load_config, resolve_path
from tarqa.data.discovery import discover_domains
from tarqa.eval.task_spec_eval import evaluate_task_spec_generator, save_task_spec_results
from tarqa.task_spec.dataset import TaskSpecDatasetBuilder
from tarqa.task_spec.inference import LocalTaskSpecGenerator
from tarqa.utils.io import read_json, read_jsonl, write_json


def validate_test_file(path, domains, *, stage, trained_uids):
    """Catch wrong-split/domain inputs before loading the language model."""
    counts = defaultdict(int)
    for row in read_jsonl(path):
        if row.get("qa_split") != "test" or row.get("domain") not in domains:
            raise ValueError(f"Wrong test split/domain in {path}: {row.get('uid')}")
        if row.get("stage") != stage:
            raise ValueError(f"Prepared test stage mismatch in {path}")
        if row["uid"] in trained_uids:
            raise ValueError(f"Test QA also appears in adapter training: {row['uid']}")
        if stage == "pretrain" and (row.get("n_shots", 0) or row.get("demonstration_uids")):
            raise ValueError("Pretraining test inputs must not contain demonstrations")
        counts[row["domain"]] += 1
    if set(counts) != set(domains):
        raise ValueError(f"Missing test domains in {path}: {sorted(set(domains) - set(counts))}")
    return dict(counts)


def prepare_source_test(cfg, holdout, output_path):
    domains = [d for d in cfg["data"]["domains"] if d != holdout]
    paths = discover_domains(
        resolve_path(cfg["data"]["dataset_root"], cfg["_config_path"]),
        resolve_path(cfg["data"]["db_root"], cfg["_config_path"]),
        domains,
    )
    training = cfg["training"]
    builder = TaskSpecDatasetBuilder(
        paths,
        generation_strategy=training.get("generation_strategy", "track_last"),
        anonymize_schema=training.get("anonymize_schema", True),
        train_shot_choices=[0], validation_shots=0,
        test_shots=training.get("test_shots", 0),
        validation_ratio=0, seed=training.get("seed", 42),
        include_variants=training.get("include_variants", False),
    )
    return builder.build_source_test(holdout, output_path)


def print_metrics(name, metrics):
    print(f"\n[{name}]", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task_spec.yaml")
    parser.add_argument("--holdout", required=True, type=str.lower)
    parser.add_argument("--stage", choices=["pretrain", "full", "few"], default="pretrain")
    parser.add_argument("--k", type=int)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--scope", choices=["source", "target", "both"], default="both",
        help="source: source qa/test; target: held-out qa/test (default); both: same adapter on both",
    )
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.scope != "target" and args.stage != "pretrain":
        parser.error("--scope source/both currently applies to --stage pretrain")
    if args.test_file and args.scope != "target":
        parser.error("--test-file is an explicit single-file override; omit it for --scope source/both")
    if args.stage == "few" and (args.k is None or args.k < 1):
        parser.error("--stage few requires a positive --k")

    cfg = load_config(args.config)
    domains = cfg["data"]["domains"]
    if args.holdout not in domains:
        parser.error(f"Unknown holdout: {args.holdout}")
    strategy = f"few_{args.k}" if args.stage == "few" else args.stage
    processed = resolve_path(cfg["data"]["processed_root"], cfg["_config_path"])
    data_dir = processed / "task_spec" / f"holdout_{args.holdout}" / strategy
    test_file = (
        resolve_path(args.test_file, cfg["_config_path"])
        if args.test_file
        else data_dir / "test.jsonl"
    )
    output = (
        resolve_path(args.output_dir, cfg["_config_path"])
        if args.output_dir
        else resolve_path(cfg["training"]["output_root"], cfg["_config_path"])
        / f"holdout_{args.holdout}"
        / strategy
        / "evaluation"
    )
    adapter = resolve_path(args.adapter, cfg["_config_path"])
    manifest_path = adapter / "training_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if manifest and (manifest.get("holdout_domain") != args.holdout or manifest.get("stage") != args.stage):
        raise ValueError("Adapter training_manifest has a different holdout/stage")
    trained_uids = {
        uid for items in manifest.get("selected_train_uids", {}).values() for uid in items
    }

    # Preserve explicit single-file evaluation, including the grid script's validation.jsonl.
    jobs = []
    if args.test_file:
        if not test_file.is_file():
            raise FileNotFoundError(test_file)
        jobs.append(("custom", test_file, output))
    else:
        if args.scope in {"target", "both"}:
            validate_test_file(test_file, [args.holdout], stage=args.stage, trained_uids=trained_uids)
        if args.scope in {"source", "both"}:
            source_domains = [d for d in domains if d != args.holdout]
            if manifest.get("train_by_domain") and set(manifest["train_by_domain"]) != set(source_domains):
                raise ValueError("Configured source domains differ from adapter training domains")
            source_file = data_dir / "source_test.jsonl"
            if not source_file.is_file():
                # Compatibility with data prepared before 01 exported source tests.
                prepare_source_test(cfg, args.holdout, source_file)
            counts = validate_test_file(
                source_file, source_domains, stage="pretrain", trained_uids=trained_uids
            )
            print(f"Source qa/test examples: {counts}", flush=True)
            jobs.append(("source_test", source_file, output / "source_test"))
        if args.scope in {"target", "both"}:
            target_output = output / "target_test" / args.holdout if args.scope == "both" else output
            jobs.append(("target_test", test_file, target_output))

    model_cfg = dict(cfg["model"])
    model_cfg["max_new_tokens"] = int(cfg["training"].get("max_new_tokens", 256))
    generator = LocalTaskSpecGenerator(model_cfg, adapter)  # Load exactly once; no training here.
    summary = {
        "adapter": str(adapter), "holdout_domain": args.holdout, "stage": args.stage,
        "scope": "custom" if args.test_file else args.scope,
        "adapter_manifest_available": bool(manifest),
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "results": {},
    }
    for name, input_file, destination in jobs:
        print(f"\nEvaluating {name}: {input_file}", flush=True)
        metrics = evaluate_task_spec_generator(generator, input_file, destination)
        print_metrics(name, metrics)
        result = {"input_file": str(input_file), "output_dir": str(destination), "metrics": metrics}
        if name == "source_test":
            # Re-score the same generations; never run inference twice for per-domain results.
            groups = defaultdict(list)
            for row in read_jsonl(destination / "predictions.jsonl"):
                groups[row["domain"]].append(row)
            by_domain = {}
            for domain, rows in groups.items():
                by_domain[domain] = save_task_spec_results(rows, destination / domain)
                print_metrics(f"source_test/{domain}", by_domain[domain])
            write_json(destination / "metrics_by_domain.json", by_domain)
            result["metrics_by_domain"] = by_domain
        summary["results"][name] = result
    write_json(output / "evaluation_summary.json", summary)
    print(f"\nResults saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
