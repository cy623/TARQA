"""Namespaced copies of the audited release visibility policies."""
from importlib import import_module

def open_history(paths, reference_time):
    if paths.domain == "hepatitis":
        raise ValueError("Hepatitis has no prediction branch")
    module = import_module(f"tarqa.prediction.history_views.{paths.domain}")
    return module.open_history(paths.database, reference_time, paths.release_root)
