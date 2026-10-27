from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from tarqa.data.discovery import DomainPaths
from tarqa.data.metadata import SchemaCatalog
from tarqa.data.qa_loader import QARecord, load_qa_records
from tarqa.data.releases import ReleaseDataset, SPLITS
from tarqa.data.task_spec_labels import build_gold_task_spec
from tarqa.task_spec.anonymization import TableAnonymization
from tarqa.task_spec.prompts import Demonstration, SYSTEM_PROMPT, render_user_prompt
from tarqa.task_spec.serialization import serialize_schema
from tarqa.types import TaskSpec
from tarqa.utils.io import write_json, write_jsonl
from tarqa.utils.seed import stable_int


class TaskSpecDatasetBuilder:
    def __init__(
        self,
        paths: dict[str, DomainPaths],
        *,
        generation_strategy: str,
        anonymize_schema: bool,
        train_shot_choices: list[int],
        validation_shots: int,
        test_shots: int,
        validation_ratio: float,
        seed: int,
        unsupported_policy: str = "error",
        include_variants: bool = False,
    ) -> None:
        self.paths = paths
        self.generation_strategy = generation_strategy
        self.anonymize_schema = anonymize_schema
        self.train_shot_choices = train_shot_choices
        self.validation_shots = validation_shots
        self.test_shots = test_shots
        self.validation_ratio = validation_ratio
        self.seed = seed
        self.unsupported_policy = unsupported_policy
        self.include_variants = include_variants

        self.releases = {domain: ReleaseDataset(item) for domain, item in paths.items()}
        self.records = {domain: release.qa("train") for domain, release in self.releases.items()}
        self.catalogs: dict[str, SchemaCatalog] = {
            domain: SchemaCatalog.from_files(domain, item.metadata, item.profiling, database_path=item.database)
            for domain, item in paths.items()
        }

    def _safe_spec(self, record: QARecord) -> TaskSpec | None:
        try:
            return build_gold_task_spec(record)
        except ValueError:
            if self.unsupported_policy == "skip" and record.query_type == "Predictive":
                return None
            raise

    def _pool(self, domain: str) -> list[tuple[QARecord, TaskSpec]]:
        out: list[tuple[QARecord, TaskSpec]] = []
        for record in self.records[domain]:
            spec = self._safe_spec(record)
            if spec is not None:
                out.append((record, spec))
        return out

    def _make_example(
        self,
        record: QARecord,
        spec: TaskSpec,
        question: str,
        pool: list[tuple[QARecord, TaskSpec]],
        n_shots: int,
    ) -> dict[str, Any]:
        catalog = self.catalogs[record.domain]
        anonymization: TableAnonymization | None = None
        if self.anonymize_schema:
            anonymization = TableAnonymization.build(catalog.table_names, record.uid)
            rendered_catalog = anonymization.apply_catalog(catalog)
            rendered_spec = anonymization.apply_spec(spec)
            rendered_question = anonymization.apply_text(question)
        else:
            rendered_catalog = catalog
            rendered_spec = spec
            rendered_question = question

        rng = random.Random(stable_int(f"{self.seed}:{record.uid}:{question}:{n_shots}"))
        candidates = [(r, s) for r, s in pool if r.uid != record.uid]
        picked = rng.sample(candidates, min(n_shots, len(candidates))) if n_shots else []
        demos: list[Demonstration] = []
        for demo_record, demo_spec in picked:
            if anonymization:
                demo_spec = anonymization.apply_spec(demo_spec)
            demo_question = (
                anonymization.apply_text(demo_record.primary_question)
                if anonymization
                else demo_record.primary_question
            )
            demos.append(Demonstration(demo_question, demo_spec))

        schema_text = serialize_schema(rendered_catalog)
        if record.domain == "ncaa":
            schema_text += "\nTime convention: an observation day denotes the NEXT day's midnight cutoff; game facts become available at next-day midnight."

        user_prompt = render_user_prompt(
            schema_text,
            rendered_question,
            demonstrations=demos,
            generation_strategy=self.generation_strategy,
        )
        assistant_text = json.dumps(
            rendered_spec.ordered_dict(self.generation_strategy),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        variants = [q for q in record.all_questions if q != question] if self.include_variants else []
        pair_question = variants[stable_int(question, max(1, len(variants)))] if variants else question
        if anonymization:
            pair_question = anonymization.apply_text(pair_question)
        pair_prompt = render_user_prompt(
            schema_text,
            pair_question,
            demonstrations=demos,
            generation_strategy=self.generation_strategy,
        )

        return {
            "uid": record.uid,
            "domain": record.domain,
            "instance_id": record.instance_id,
            "question": question,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "assistant_text": assistant_text,
            "pair_user_prompt": pair_prompt,
            "gold_spec": rendered_spec.ordered_dict(self.generation_strategy),
            "real_gold_spec": spec.ordered_dict(self.generation_strategy),
            "table_map": anonymization.forward if anonymization else {},
            "reverse_table_map": anonymization.reverse if anonymization else {},
            "n_shots": len(picked),
            "demonstration_uids": [r.uid for r, _ in picked],
        }

    def build_source_test(self, holdout_domain: str, output_path: str | Path) -> dict[str, int]:
        """Export only source qa/test; never rewrite train/validation or resplit QA."""
        if self.test_shots:
            raise ValueError("Pretraining test evaluation requires test_shots=0")
        rows: list[dict[str, Any]] = []
        for domain, release in self.releases.items():
            if domain == holdout_domain.lower():
                continue
            for record in release.qa("test"):
                spec = self._safe_spec(record)
                if spec is None:
                    continue
                questions = record.all_questions if self.include_variants else [record.primary_question]
                for question in questions:
                    example = self._make_example(record, spec, question, [], 0)
                    example.update(qa_split="test", stage="pretrain")
                    rows.append(example)
        if not rows:
            raise ValueError("No source-domain test examples found")
        write_jsonl(output_path, rows)
        return _count_domains(rows)

    def build(self, holdout_domain, output_dir, *, stage="pretrain", few_shot_k=None):
        holdout_domain = holdout_domain.lower()
        if holdout_domain not in self.releases:
            raise ValueError(f"Unknown holdout domain: {holdout_domain}")
        if stage not in {"pretrain", "full", "few"}:
            raise ValueError(stage)
        if stage == "pretrain" and self.test_shots:
            raise ValueError("Zero-shot TSG testing cannot use target-domain demonstrations")
        output_dir = Path(output_dir)
        rows = {split: [] for split in SPLITS}
        selected = {}
        for domain, release in self.releases.items():
            if stage != "pretrain" and domain != holdout_domain:
                continue
            train = release.qa("train")
            if stage == "few":
                if few_shot_k is None or few_shot_k < 1:
                    raise ValueError("TSG few-shot requires a positive original-QA budget")
                train = sorted(train, key=lambda r: stable_int(f"{self.seed}:{r.uid}"))[:few_shot_k]
            allowed_train = train if domain != holdout_domain or stage != "pretrain" else []
            selected[domain] = [r.uid for r in allowed_train]
            pool = [(r, self._safe_spec(r)) for r in allowed_train]
            pool = [(r, spec) for r, spec in pool if spec is not None]
            for split in SPLITS:
                if split == "test" and domain != holdout_domain:
                    continue
                if split != "test" and domain == holdout_domain and stage == "pretrain":
                    continue
                records = allowed_train if split == "train" else release.qa(split)
                for record in records:
                    spec = self._safe_spec(record)
                    if spec is None:
                        continue
                    questions = record.all_questions if self.include_variants else [record.primary_question]
                    for question in questions:
                        if split == "train":
                            choices = self.train_shot_choices or [0]
                            n = choices[stable_int(f"{record.uid}:{question}", len(choices))]
                        else:
                            n = self.validation_shots if split == "validation" else self.test_shots
                        example = self._make_example(record, spec, question, pool, n)
                        example.update(qa_split=split, stage=stage)
                        rows[split].append(example)
        for split, items in rows.items():
            write_jsonl(output_dir / f"{split}.jsonl", items)
        manifest = dict(holdout_domain=holdout_domain, stage=stage,
            generation_strategy=self.generation_strategy, selected_train_uids=selected,
            train_examples=len(rows["train"]), validation_examples=len(rows["validation"]),
            test_examples=len(rows["test"]), include_variants=self.include_variants,
            tsg_few_shot_unit="original QA records", requested_k=few_shot_k,
            train_by_domain=_count_domains(rows["train"]),
            validation_by_domain=_count_domains(rows["validation"]),
            test_by_domain=_count_domains(rows["test"]))
        if stage == "pretrain":
            source_counts = self.build_source_test(holdout_domain, output_dir / "source_test.jsonl")
            manifest["source_test_examples"] = sum(source_counts.values())
            manifest["source_test_by_domain"] = source_counts
        write_json(output_dir / "manifest.json", manifest)
        return manifest


def _count_domains(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["domain"]] += 1
    return dict(sorted(counts.items()))
