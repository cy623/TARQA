"""Stable request hashes and atomic SQL-stage trace writes."""
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def save_json(path, value):
    """Strict JSON and atomic completion files; NaN is never a metric."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)
