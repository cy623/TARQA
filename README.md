# TARQA

TARQA: A Unified Framework for Task-Aware Relational Question Answering. Task Spec generation, SQL-based Reasoning, and question-conditioned relational Classification/Regression. The same framework supports pretrained inference, fresh head training, source
pretraining, and per-task target adaptation.

## Files

| File / directory | Contents |
|---|---|
| `run.py` | Entry point for testing and training |
| `src/tarqa/model.py` | Unified model interface |
| `src/tarqa/task_spec/` | Task Spec Generator (TSG) |
| `src/tarqa/reasoning/` | Schema selection, SQL generation and execution |
| `src/tarqa/prediction/` | Historical graphs, encoders and prediction heads |
| `scripts/`, `configs/` | Training scripts and configuration templates |
| `requirements.txt` | Python dependencies |
| `selected_checkpoint_release.json` | Checkpoint archive names and checksums |

## Download parameters and data

Download these **eight archives** from this repository's [Releases](https://github.com/cy623/TARQA/releases/tag/TARQA):


Also download the original base models separately:
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659)
and [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd).
The parameter archives contain LoRA adapters, not the base models.

## Run tests

Use Linux, Python 3.10+, and CUDA. Run all commands from the code directory.
Install dependencies and prepare local paths once, replacing the base-model paths:

```bash
python -m pip install -r requirements.txt
python run.py setup \
  --data-root ../rdl-bench-data-v2 \
  --resources-root ../tarqa-checkpoints-olist-stats-financial \
  --llama-base /path/to/llama-snapshot --qwen-base /path/to/qwen-snapshot \
  --work-dir work
```

**In-domain:** this example evaluates all three task types on Stats with Llama:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --configuration work/runtime/configurations/indomain/llama8b/stats/seed42/indomain.json \
  --dataset-root work/dataset --rows-root work/rows \
  --runtime-root work/runtime --backbone llama8b \
  --output outputs/stats_llama_indomain
```

**Cross-domain:** use configuration
`work/runtime/configurations/crossdomain/llama8b/stats/seed42/few_3.json`
and add `--task prediction`. Available budgets are `few_1`, `few_3`, `few_5`,
`few_10`. Use a different output directory for each run.


## Run training

### In-domain prediction heads

Generate frozen TSG/graph features, then train fresh Classification/Regression heads:

```bash
CFG=work/runtime/configurations/indomain/llama8b/stats/seed42/indomain.json
CACHE=work/cache/llama8b/indomain_stats
CUDA_VISIBLE_DEVICES=0 python run.py build-features --configuration "$CFG" \
  --dataset-root work/dataset --rows-root work/rows --cache-root "$CACHE" \
  --domains stats --roles support/train support/validation
python run.py train-heads --configuration "$CFG" --cache-root "$CACHE" \
  --rows-root work/rows --mode indomain --seed 42 --device cuda \
  --output outputs/heads/llama8b/stats/indomain
```

Training uses `support/train`; checkpoint selection uses `support/validation`.
Defaults are 200 epochs and patience 20, adjustable with `--epochs` and `--patience`.

### Cross-domain adaptation

Reuse the downloaded source heads and adapt on target training supports:

```bash
CFG=work/runtime/configurations/crossdomain/llama8b/stats/seed42/few_3.json
CACHE=work/cache/llama8b/holdout_stats
CUDA_VISIBLE_DEVICES=0 python run.py build-features --configuration "$CFG" \
  --dataset-root work/dataset --rows-root work/rows --cache-root "$CACHE" \
  --domains stats --roles support/train support/validation
python run.py train-heads --configuration "$CFG" --cache-root "$CACHE" \
  --rows-root work/rows --mode few_3 --seed 42 --device cuda \
  --source-heads work/runtime/weights/heads/llama8b/source/stats/seed42/source \
  --output outputs/heads/llama8b/stats/few_3
```

