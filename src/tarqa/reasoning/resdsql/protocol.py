"""Source/target split and checkpoint provenance for the SQL executor."""
from pathlib import Path

from tarqa.utils.io import read_json


def select_training_qa(records_by_domain, holdout, stage, *, k=None, selection=None):
    if stage not in {"pretrain", "full", "few"} or holdout not in records_by_domain:
        raise ValueError("Invalid RESDSQL fold/stage")
    if stage == "pretrain":
        selected = {d: list(rows) for d, rows in records_by_domain.items() if d != holdout}
    elif stage == "full":
        selected = {holdout: list(records_by_domain[holdout])}
    else:
        if not k or k < 1 or selection is None:
            raise ValueError("Few-shot requires --k and the shared TSG selection manifest")
        if (selection.get("holdout_domain"), selection.get("stage"), selection.get("requested_k")) != (holdout, "few", k):
            raise ValueError("TSG few-shot selection belongs to another fold/stage/budget")
        ids = selection.get("selected_train_uids", {}).get(holdout, [])
        allowed = {r.uid: r for r in records_by_domain[holdout]}
        if len(set(ids)) != len(ids) or len(ids) > k or not set(ids) <= set(allowed):
            raise ValueError("Few-shot IDs must be distinct target qa/train questions")
        if not ids:
            raise ValueError("Empty TSG few-shot selection")
        selected = {holdout: [allowed[uid] for uid in ids]}
    all_ids = {d: [r.uid for r in rows] for d, rows in selected.items()}
    reasoning = {d: [r for r in rows if r.query_type != "Predictive"] for d, rows in selected.items()}
    return reasoning, all_ids


def validate_checkpoint(path, holdout, stage=None, *, selection=None):
    manifest = read_json(Path(path) / "training_manifest.json")
    if manifest.get("training_complete") is not True:
        raise ValueError("RESDSQL checkpoint is incomplete")
    if manifest.get("backend") != "resdsql" or manifest.get("holdout_domain") != holdout:
        raise ValueError("RESDSQL checkpoint belongs to another backend/holdout fold")
    if stage and manifest.get("stage") != stage:
        raise ValueError(f"Expected RESDSQL stage {stage}, got {manifest.get('stage')}")
    if manifest.get("stage") == "pretrain" and holdout in manifest.get("train_domains", []):
        raise ValueError("Source RESDSQL checkpoint was trained on the target domain")
    if selection is not None:
        expected = set(selection.get("selected_train_uids", {}).get(holdout, []))
        actual = set(manifest.get("selected_train_uids", {}).get(holdout, []))
        if expected != actual:
            raise ValueError("TSG and RESDSQL used different target few-shot QA selections")
    return manifest
