"""Score newly computed predictions; no bundled historical results are required."""
from copy import deepcopy
import gc
from pathlib import Path
import numpy as np

from tarqa.eval.scoring import prediction_row, summarize
from tarqa.utils.io import read_json, write_json


def _raise_runtime_failure(error):
    text = (type(error).__name__ + ': ' + str(error)).lower()
    if isinstance(error, MemoryError) or any(s in text for s in
            ('out of memory', 'outofmemory', 'cuda error', 'device-side assert')):
        raise RuntimeError('Operational memory/CUDA failure; evaluation is incomplete: ' + str(error)) from error


def _test_rows(rows_root, domain, limit=None):
    rows = read_json(Path(rows_root) / domain / 'scorer_only/qa_test.json')
    if limit:
        rows = [r for kind in ('Classification', 'Regression')
                for r in [x for x in rows if x['task_type'] == kind][:limit]]
    return rows


def evaluate_prediction(configuration, rows_root, output, dataset=None, cache_root=None,
                        head_root=None, limit=None):
    cfg = read_json(configuration); domain = cfg['domain']; output = Path(output)
    (output / 'prediction_metrics.json').unlink(missing_ok=True)
    if dataset:
        cfg['dataset_root'] = str(Path(dataset).resolve())
    if head_root:
        cfg['prediction_heads'] = str(Path(head_root).resolve())
    rows = _test_rows(rows_root, domain, limit); predictions = []
    if cache_root:
        from tarqa.prediction.dataset import load_cached_role
        from tarqa.prediction.trainer import HeadArtifact
        values = {}
        for kind in ('Classification', 'Regression'):
            artifact = HeadArtifact.load(Path(cfg['prediction_heads']) / kind)
            batch = load_cached_role(cache_root, domain, 'qa/test', kind, allow_all_failed=True)
            raw = artifact.predict({k: batch[k] for k in ('X_rel', 'X_query', 'manifest')})
            if kind == 'Classification':
                raw = 1 / (1 + np.exp(-np.clip(raw, -80, 80)))
            values.update({r['sample_id']: float(v) for r, v in zip(batch['rows'], raw) if np.isfinite(v)})
        for row in rows:
            value = values.get(row['sample_id'])
            predictions.append(prediction_row(row, value, .5,
                None if value is not None else 'no_usable_generated_features'))
    else:
        from tarqa.model import TARQAModel
        model = TARQAModel.from_pretrained(cfg)
        try:
            for i, row in enumerate(rows, 1):
                extra = {}
                try:
                    result = model.answer(row['question'], anchor=row['anchor'],
                        observation_cutoff=row['reference_time'], uid=row['sample_id'])
                    spec = result.task_spec
                    extra = dict(task_spec=spec.model_dump(mode='json'),
                                 natural_language_answer=result.natural_language_answer)
                    if spec.track.value != 'Prediction' or spec.task_type.value != row['task_type']:
                        raise ValueError('Generated execution track or output type differs from the scored question')
                    value = result.canonical_answer.probability if row['task_type'] == 'Classification' else result.canonical_answer.value
                    if value is None or not np.isfinite(float(value)):
                        raise ValueError('No finite predicted value')
                    prediction = prediction_row(row, float(value), .5)
                except Exception as exc:
                    _raise_runtime_failure(exc)
                    last_spec = getattr(model.pipeline, 'last_task_spec', None)
                    if last_spec is not None:
                        extra['task_spec'] = last_spec.model_dump(mode='json')
                    prediction = prediction_row(row, error=f'{type(exc).__name__}: {exc}')
                prediction.update(extra); predictions.append(prediction)
                write_json(output / 'prediction_answers.json', predictions)
                print(domain, 'Prediction', i, '/', len(rows), prediction['status'], flush=True)
        finally:
            model.close()
    scope = cfg.get('experiment', {}).get('scope', 'indomain')
    metrics = {'full_population': summarize(predictions)}
    if scope == 'crossdomain':
        train = read_json(Path(rows_root) / domain / 'scorer_only/support_train.json')
        counts = {}
        for r in train:
            if r.get('eligible', True):
                counts[r['task_id']] = counts.get(r['task_id'], 0) + 1
        metrics.update(primary=summarize([r for r in predictions if counts.get(r['task_id'], 0) >= 10]),
                       unseen_tasks=summarize([r for r in predictions if r['task_id'] not in counts]),
                       insufficient_support=summarize([r for r in predictions if 0 < counts.get(r['task_id'], 0) < 10]))
    write_json(output / 'prediction_answers.json', predictions)
    write_json(output / 'prediction_metrics.json', dict(partial_test=bool(limit), metrics=metrics,
        classification_threshold=.5, fresh_tsg_generation=not bool(cache_root),
        fresh_graph_encoding=not bool(cache_root), seed=cfg.get('experiment', {}).get('seed', 42)))
    return metrics


def evaluate_reasoning(configuration, runtime_root, backbone, output, dataset=None, limit=None):
    """Historical paper pairing: fixed Qwen TSG for both frozen SQL backbones."""
    import torch
    from tarqa.data.discovery import discover_domain
    from tarqa.data.metadata import SchemaCatalog
    from tarqa.data.releases import ReleaseDataset
    from tarqa.task_spec.inference import LocalTaskSpecGenerator
    from tarqa.task_spec.prompts import SYSTEM_PROMPT, render_user_prompt
    from tarqa.task_spec.serialization import serialize_schema
    from tarqa.reasoning.pipeline import ReasoningPipeline
    from tarqa.reasoning.executor import DuckDBReadOnlyExecutor
    from tarqa.reasoning.resdsql.evaluation import same_rows
    from tarqa_baselines.export import schema_ddl
    from tarqa_baselines.reasoning_llm import LocalChatModel
    from tarqa.model import _SharedBaseChat, _SQLModel
    cfg = read_json(configuration); output = Path(output); domain = cfg['domain']
    (output / 'reasoning_metrics.json').unlink(missing_ok=True)
    paths = discover_domain(dataset or cfg['dataset_root'], None, domain)
    release = ReleaseDataset(paths, validate_rows=False)
    questions = [q for q in release.qa('test') if q.query_type != 'Predictive']
    if limit:
        questions = questions[:limit]
    template = Path(runtime_root) / 'weights/heads/qwen32b/source/financial/seed42/source/Classification/model_config.json'
    provenance = read_json(template)['task_spec_provenance']
    generator = LocalTaskSpecGenerator(provenance['model_config'], Path(provenance['adapter']['path']).parent)
    catalog = SchemaCatalog.from_files(domain, paths.metadata, database_path=paths.database)
    schema = serialize_schema(catalog); generated = {}
    for q in questions:
        try:
            spec = generator.generate(SYSTEM_PROMPT, render_user_prompt(schema, q.primary_question, generation_strategy='track_last')).spec
            if spec.track.value != 'Reasoning':
                raise ValueError('TSG did not route to Reasoning')
            generated[q.uid] = (spec, None)
        except Exception as exc:
            _raise_runtime_failure(exc)
            generated[q.uid] = (None, f'{type(exc).__name__}: {exc}')
    sql_cfg = dict(cfg['sql_executor']); backend = sql_cfg.pop('backend'); model_path = sql_cfg.pop('model_path')
    sql_cfg.pop('tokenizer_path', None)
    if backbone == 'qwen32b':
        client = _SharedBaseChat(generator, sql_cfg)
    else:
        del generator; gc.collect(); torch.cuda.empty_cache()
        client = LocalChatModel(model_path, sql_cfg, seed=42)
    sql = _SQLModel(client, backend, output / 'sql_stages', cfg['demonstration_bank'], cfg['num_examples'])
    pipeline = ReasoningPipeline(catalog, paths.database, sql, add_bridge_tables=True,
        schema_policy='tsg_selected', max_repair_attempts=0, timeout_seconds=120,
        schema_renderer=lambda c: '-- SQL dialect: DuckDB\n' + schema_ddl(c))
    gold_executor = DuckDBReadOnlyExecutor(paths.database, timeout_seconds=120)
    answers = []
    try:
        for i, q in enumerate(questions, 1):
            spec, error = generated[q.uid]
            row = dict(uid=q.uid, question=q.primary_question, correct=False, executable=False,
                       task_spec=spec.model_dump(mode='json') if spec else None)
            try:
                if error:
                    raise ValueError(error)
                sql.spec = spec; sql.uid = q.uid
                result = pipeline.answer(q.primary_question, spec)
                # Gold SQL enters this scorer only, after model execution.
                expected = gold_executor.execute(q.golden_sql)[1]
                row.update(sql=result.sql, rows=result.rows, executable=True,
                    canonical_answer=result.answer.model_dump(mode='json'),
                    correct=bool(same_rows(result.rows, expected, q.answer_nature)))
            except Exception as exc:
                _raise_runtime_failure(exc)
                row['error'] = f'{type(exc).__name__}: {exc}'
            answers.append(row); write_json(output / 'reasoning_answers.json', answers)
            print(domain, 'Reasoning', i, '/', len(questions), row['correct'], flush=True)
    finally:
        pipeline.close()
    result = dict(n=len(answers), correct=sum(r['correct'] for r in answers),
        execution_accuracy=sum(r['correct'] for r in answers) / len(answers) if answers else None,
        partial_test=bool(limit), tsg='qwen32b/holdout_financial', sql_backend=backend,
        fresh_tsg_generation=True, fresh_sql_generation=True)
    write_json(output / 'reasoning_metrics.json', result)
    return result
