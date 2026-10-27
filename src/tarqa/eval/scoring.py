"""Task-wise scoring with an explicit, answer-independent failure fallback."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

FAILURE_POLICY = {
    "classification": "p=0.5 and predicted class 1; fixed before any test predictions",
    "regression": "predict 0 in target units; fixed before any test predictions",
    "reporting": "Failed/uncovered predictions retain their IDs, use the fixed fallback in all-sample scores, and count against coverage. Also report successful-only scores and common-coverage scores.",
    "data_errors": "Ineligible labels/leaking supervision are listed separately and excluded from formal scores; never silently removed.",
}


def signed_log(value):
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.log1p(np.abs(value))


def inverse_signed_log(value):
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.expm1(np.minimum(np.abs(value), 50.0))


def score(rows, kind):
    eligible = [r for r in rows if r.get("eligible", True)]
    result = {"n_total": len(rows), "n_scored": len(eligible),
              "n_data_excluded": len(rows) - len(eligible),
              "n_success": sum(r.get("status") == "ok" for r in eligible),
              "n_failed": sum(r.get("status") != "ok" for r in eligible)}
    result["coverage"] = result["n_success"] / len(eligible) if eligible else None
    if not eligible:
        return result
    y = np.asarray([r["label"] for r in eligible], dtype=float)
    p = np.asarray([r["prediction"] for r in eligible], dtype=float)
    if kind == "Classification":
        prob = np.asarray([r["probability"] for r in eligible], dtype=float)
        result.update(accuracy=float(accuracy_score(y, p)), f1=float(f1_score(y, p, zero_division=0)),
                      auroc=float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
                      brier=float(brier_score_loss(y, prob)),
                      auroc_note=None if len(np.unique(y)) == 2 else "single_class_labels")
    else:
        result.update(mae=float(np.mean(np.abs(y - p))), rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                      signed_log_mae=float(np.mean(np.abs(signed_log(y) - signed_log(p)))))
    return result


def summarize(rows):
    result = {"by_task": {}, "by_type": {}}
    for task_id in sorted({r["task_id"] for r in rows}):
        part = [r for r in rows if r["task_id"] == task_id]
        result["by_task"][task_id] = {"task_type": part[0]["task_type"], **score(part, part[0]["task_type"])}
    for kind in ("Classification", "Regression"):
        part = [r for r in rows if r["task_type"] == kind]
        value = score(part, kind)
        if kind == "Regression":
            # Raw-scale units differ across tasks; never present a pooled MAE/RMSE.
            value.pop("mae", None)
            value.pop("rmse", None)
        tasks = [m for m in result["by_task"].values() if m["task_type"] == kind and m["n_scored"]]
        metric = "accuracy" if kind == "Classification" else "signed_log_mae"
        value["task_macro_" + metric] = float(np.mean([t[metric] for t in tasks])) if tasks else None
        result["by_type"][kind] = value
    return result


def prediction_row(row, value=None, threshold=.5, error=None):
    result = {k: row[k] for k in ("sample_id", "parent_question_id", "domain", "role", "task_id", "task_type", "label", "eligible")}
    result.update(status="ok" if error is None else "failed", failure_reason=error)
    if row["task_type"] == "Classification":
        p = .5 if error else float(value)
        result.update(probability=p, prediction=int(p >= (.5 if error else threshold)), threshold=threshold)
    else:
        result.update(probability=None, prediction=0.0 if error else float(value))
    return result


