from pathlib import Path
from tarqa.utils.io import read_json

def collect_prediction_metrics(artifact_root):
    root=Path(artifact_root)
    return {str(p.relative_to(root)):read_json(p) for p in sorted(root.rglob("metrics.json"))}


def domain_macro_metrics(metrics_by_domain):
    """Equal-weight domain means; exclude empty domains and undefined metrics.

    Raw regression errors have different units across domains, so only
    signed-log MAE is aggregated. Per-domain reports retain MAE and RMSE.
    """
    import math
    from statistics import fmean

    result = {}
    for task, names in {
        'Classification': ('accuracy', 'f1', 'auroc', 'brier'),
        'Regression': ('signed_log_mae',),
    }.items():
        result[task] = {}
        for name in names:
            values = {}
            for domain, tasks in sorted(metrics_by_domain.items()):
                metrics = tasks.get(task, {})
                value = metrics.get(name)
                if metrics.get('n', 0) > 0 and value is not None and math.isfinite(float(value)):
                    values[domain] = float(value)
            result[task][name] = dict(
                value=fmean(values.values()) if values else None,
                num_domains=len(values), domains=list(values),
                excluded_domains=sorted(set(metrics_by_domain) - set(values)),
            )
    return result
