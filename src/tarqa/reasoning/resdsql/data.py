"""Build RESDSQL supervision from existing QA releases; no new split/supports."""
from collections import Counter
from pathlib import Path

from tarqa.data.discovery import discover_domains
from tarqa.data.metadata import SchemaCatalog
from tarqa.data.releases import ReleaseDataset, file_hash
from tarqa.reasoning.executor import DuckDBReadOnlyExecutor
from tarqa.reasoning.resdsql import PROTOCOL, UPSTREAM_COMMIT
from tarqa.reasoning.resdsql.content import DuckDBContents
from tarqa.reasoning.resdsql.formatting import public_payload, sql_supervision
from tarqa.reasoning.resdsql.protocol import select_training_qa
from tarqa.config import resolve_path
from tarqa.utils.io import read_json, write_json

INPUT_OPTIONS = ("use_contents", "content_max_distinct", "content_max_matches", "content_max_value_chars")


def build_data(config, holdout, output, *, stage="pretrain", k=None, selection_manifest=None):
    paths = discover_domains(resolve_path(config["data"]["dataset_root"], config["_config_path"]),
        resolve_path(config["data"]["db_root"], config["_config_path"]), config["data"]["domains"])
    if holdout not in paths:
        raise ValueError(f"Unknown holdout: {holdout}")
    releases = {d: ReleaseDataset(p) for d, p in paths.items()}
    reports = {d: release.validate() for d, release in releases.items()}
    train = {d: release.qa("train") for d, release in releases.items()}
    selection = read_json(selection_manifest) if selection_manifest else None
    selected, all_ids = select_training_qa(train, holdout, stage, k=k, selection=selection)
    roles = {"train": [q for rows in selected.values() for q in rows],
             "validation": [q for d in selected for q in releases[d].qa("validation")
                            if q.query_type != "Predictive"]}
    if not roles["train"] and stage != "few":
        raise ValueError("No Reasoning training QA in the selected domains")
    if roles["train"] and not roles["validation"]:
        raise ValueError("A nonempty validation split is required for checkpoint selection")
    catalogs = {d: SchemaCatalog.from_files(d, paths[d].metadata, paths[d].profiling,
                                           database_path=paths[d].database) for d in selected}
    model_cfg = config["model"]
    contents = {d: DuckDBContents(paths[d].database, catalogs[d],
        max_distinct=model_cfg.get("content_max_distinct", 1000),
        max_matches=model_cfg.get("content_max_matches", 2),
        max_value_chars=model_cfg.get("content_max_value_chars", 80),
        timeout_seconds=model_cfg.get("content_timeout_seconds", 120)) for d in selected} if model_cfg.get("use_contents", True) else {}
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared data already exists: {output}; use a new --output-dir")
    count = {}
    try:
        for role, records in roles.items():
            prepared = []
            for record in records:
                catalog = catalogs[record.domain]
                supervision = sql_supervision(record.golden_sql, catalog)
                # Never trust SQL that cannot actually execute on this release.
                DuckDBReadOnlyExecutor(paths[record.domain].database,
                    timeout_seconds=config["execution"].get("timeout_seconds", 120)).execute(record.golden_sql)
                prepared.append(dict(uid=record.uid, domain=record.domain,
                    parent_question_id=record.instance_id, qa_split=role,
                    question=record.primary_question, golden_sql=record.golden_sql,
                    answer_nature=record.answer_nature,
                    payload=public_payload(record.primary_question, catalog, contents.get(record.domain)),
                    **supervision))
            write_json(output / f"{role}.json", prepared)
            count[role] = dict(Counter(r["domain"] for r in prepared))
        # Tests remain original QA; do not run test SQL during data preparation.
        for role, domains in (("source_test", [d for d in paths if d != holdout]), ("target_test", [holdout])):
            refs = [dict(uid=q.uid, domain=d, parent_question_id=q.instance_id, qa_split="test")
                    for d in domains for q in releases[d].qa("test") if q.query_type != "Predictive"]
            write_json(output / f"{role}.json", refs)
            count[role] = dict(Counter(r["domain"] for r in refs))
        manifest = dict(protocol=PROTOCOL, upstream_commit=UPSTREAM_COMMIT,
            backend="resdsql", holdout_domain=holdout, stage=stage,
            requested_k=k, selected_train_uids=all_ids,
            selected_reasoning_uids={d: [q.uid for q in rows] for d, rows in selected.items()},
            train_domains=list(selected), validation_domains=list(selected), counts=count,
            input_options={key: model_cfg.get(key) for key in INPUT_OPTIONS},
            release_reports=reports,
            source_paths={d: dict(database=str(p.database), metadata=str(p.metadata)) for d, p in paths.items()},
            files_sha256={f"{role}.json": file_hash(output / f"{role}.json")
                          for role in ["train", "validation", "source_test", "target_test"]},
            few_shot_unit="original QA shared with TSG; only selected Reasoning QA train RESDSQL",
            validation_budget="all Reasoning QA in the allowed validation domains; additional labelled budget")
        write_json(output / "manifest.json", manifest)
    finally:
        for store in contents.values():
            store.close()
    return manifest


def load_training_data(path, config, holdout, stage):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("protocol"), manifest.get("holdout_domain"), manifest.get("stage")) != (PROTOCOL, holdout, stage):
        raise ValueError("Prepared RESDSQL data belongs to another protocol/fold/stage")
    expected_options = {key: config["model"].get(key) for key in INPUT_OPTIONS}
    if manifest["input_options"] != expected_options:
        raise ValueError("DB-content preprocessing configuration differs from prepared data")
    rows = {}
    expected_domains = set(config["data"]["domains"]) - {holdout} if stage == "pretrain" else {holdout}
    if set(manifest["train_domains"]) != expected_domains or set(manifest["validation_domains"]) != expected_domains:
        raise ValueError("Training/validation domains violate the leave-one-domain-out protocol")
    for role in ("train", "validation"):
        source = path / f"{role}.json"
        if file_hash(source) != manifest["files_sha256"][source.name]:
            raise ValueError(f"Prepared data changed: {source}")
        rows[role] = read_json(source)
        if any(r["domain"] not in expected_domains or r["qa_split"] != role for r in rows[role]):
            raise ValueError("Unexpected domain/split in RESDSQL training data")
    train_ids = {r["uid"] for r in rows["train"]}
    val_ids = {r["uid"] for r in rows["validation"]}
    if train_ids & val_ids or len(train_ids) != len(rows["train"]) or len(val_ids) != len(rows["validation"]):
        raise ValueError("Duplicate/overlapping QA IDs in RESDSQL training and validation")
    selected = {uid for ids in manifest["selected_reasoning_uids"].values() for uid in ids}
    if selected != train_ids:
        raise ValueError("Prepared training data disagrees with selected QA IDs")
    # Only permitted validation DBs are opened by the trainer, never target tests.
    for domain in expected_domains:
        actual = file_hash(manifest["source_paths"][domain]["database"])
        if actual != manifest["release_reports"][domain]["database_sha256"]:
            raise ValueError("Database changed since RESDSQL data preparation")
    return rows["train"], rows["validation"], manifest
