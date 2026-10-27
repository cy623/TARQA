"""Select an executor without changing the Task Spec Generator."""
from pathlib import Path


def build_reasoning_model(config):
    backend = config.get("backend", "causal_lm")
    if backend == "causal_lm":
        from tarqa.reasoning.local_model import LocalTextToSQLModel
        return LocalTextToSQLModel(config)
    if backend == "resdsql":
        from tarqa.reasoning.resdsql.model import RESDSQLModel
        return RESDSQLModel(config)
    raise ValueError(f"Unknown reasoning backend: {backend}")


def resolve_reasoning_config(config, checkpoint=None):
    """Resolve local model paths relative to the TARQA project, not shell cwd."""
    from copy import deepcopy
    from tarqa.config import resolve_path
    from tarqa.utils.io import read_json
    result = deepcopy(config)
    model = result["model"]
    if checkpoint:
        root = resolve_path(checkpoint, result["_config_path"])
        saved = read_json(root / "executor_config.json")
        # Inference preprocessing must be exactly the one saved during training.
        hardware = {key: model[key] for key in ("device", "torch_dtype") if key in model}
        model = {**model, **saved, **hardware, "backend": "resdsql"}
        for key in ("model_path", "ranker_path"):
            path = Path(model[key])
            model[key] = str(path if path.is_absolute() else (root / path).resolve())
        result["model"] = model
        result["reasoning_checkpoint"] = str(root)
    for key in ("model_path", "ranker_path", "adapter_path", "tokenizer_path"):
        if model.get(key):
            model[key] = str(resolve_path(model[key], result["_config_path"]))
    if not checkpoint and model.get("backend") == "resdsql":
        for key in ("model_path", "ranker_path"):
            if (Path(model[key]).parent / "training_manifest.json").exists():
                raise ValueError("Use --checkpoint/--reasoning-checkpoint for fold-trained RESDSQL weights so provenance is checked")
    return result
