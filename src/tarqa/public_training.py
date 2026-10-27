"""Portable frozen-feature building and genuine downstream optimization."""
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import torch

from tarqa.data.discovery import discover_domains
from tarqa.data.releases import ReleaseDataset
from tarqa.prediction.dataset import PredictionDatasetBuilder, load_cached_role, subset_batch, concat_batches
from tarqa.prediction.generated_specs import (GeneratedSpecStore, claim_cache, schema_prompt,
    select_build_rows, tsg_provenance)
from tarqa.prediction.normalization import FeaturePreprocessor, RegressionTargetTransformer
from tarqa.prediction.question_encoder import FrozenQuestionEncoder
from tarqa.prediction.task_head import QueryConditionedTaskHead
from tarqa.prediction.trainer import HeadArtifact, PredictionHeadTrainer
from tarqa.task_identity import digest
from tarqa.utils.io import read_json, write_json
from tarqa.utils.seed import set_seed

KINDS = ('Classification', 'Regression')


def build_features(configuration, dataset, rows_root, cache_root, domains=None,
                   roles=('support/train', 'support/validation', 'qa/test'), encoder_device='cpu', generate_only=False,
                   task_spec_config=None, task_spec_adapter=None):
    cfg = read_json(configuration); cache_root = Path(cache_root).resolve()
    meta = read_json(Path(cfg['prediction_heads']) / 'Classification/model_config.json')
    provenance = meta['task_spec_provenance']; adapter = Path(provenance['adapter']['path']).parent
    model_cfg = provenance['model_config']
    if bool(task_spec_config) != bool(task_spec_adapter):
        raise ValueError('Supply both --task-spec-config and --task-spec-adapter for a newly trained TSG')
    holdout = provenance['holdout_domain']
    if task_spec_config:
        from tarqa.config import load_config, resolve_path
        fresh = load_config(task_spec_config); model_cfg = dict(fresh['model'])
        for key in ('model_path', 'tokenizer_path'):
            if model_cfg.get(key):
                model_cfg[key] = str(resolve_path(model_cfg[key], fresh['_config_path']))
        model_cfg['max_new_tokens'] = fresh.get('training', {}).get('max_new_tokens', 256)
        adapter = Path(task_spec_adapter).resolve()
        holdout = read_json(adapter / 'training_manifest.json')['holdout_domain']
    actual = tsg_provenance(model_cfg, adapter, holdout)
    if task_spec_config is None and actual != provenance:
        raise ValueError('Run prepare_runtime.py/setup after extracting checkpoint resources')
    domains = list(domains or [cfg['domain']])
    paths = discover_domains(dataset, None, domains)
    claim_cache(cache_root, actual)
    store = GeneratedSpecStore(cache_root / 'generated_task_specs.jsonl', actual,
        model_cfg=model_cfg, adapter=adapter)
    try:
        for domain, path in paths.items():
            schema, _ = schema_prompt(path); release = ReleaseDataset(path, validate_rows=False)
            for role in roles:
                records = select_build_rows(release, role, rows_root=rows_root)
                for i, row in enumerate(records, 1):
                    store.get(schema, row['question'], generate=True)
                    if i % 20 == 0 or i == len(records):
                        print(domain, role, 'TSG', i, '/', len(records), flush=True)
    finally:
        store.close()
    if generate_only:
        return
    encoder_config = deepcopy(meta['encoder_config']); encoder_config['device'] = encoder_device
    node_config = deepcopy(meta['node_encoder_config']); node_config['device'] = encoder_device
    encoder = FrozenQuestionEncoder(encoder_config)
    build_config = dict(question_encoder=encoder_config, node_encoder=node_config,
        representation=meta['representation_config'], aggregation=meta['aggregation_config'])
    for domain, path in paths.items():
        PredictionDatasetBuilder(path, build_config, encoder).build(cache_root / domain,
            spec_source='generated', generated_store=store, rows_root=rows_root, roles=roles)


def task_selection(rows, domain, seed, budget):
    groups = {}
    for row in rows:
        if row.get('eligible', True):
            groups.setdefault((row['task_type'], row['task_id']), []).append(row)
    result = {kind: [] for kind in KINDS}
    for (kind, task), candidates in sorted(groups.items()):
        if len(candidates) < 10:
            continue
        ordered = sorted(candidates, key=lambda r: digest([seed, domain, kind, task, r['sample_id']]))
        chosen = candidates if budget == 'full' else ordered[:int(budget.split('_')[1])]
        result[kind].extend(r['sample_id'] for r in chosen)
    return result


def batch_for(cache, rows_root, domain, role, kind):
    batch = load_cached_role(cache, domain, role, kind, allow_all_failed=True)
    official = read_json(Path(rows_root) / domain / 'scorer_only' / (role.replace('/', '_') + '.json'))
    valid = {r['sample_id']: r for r in official if r['task_type'] == kind and r.get('eligible', True)}
    batch = subset_batch(batch, [i for i, r in enumerate(batch['rows']) if r['sample_id'] in valid])
    batch['y'] = np.asarray([valid[r['sample_id']]['label'] for r in batch['rows']], np.float32)
    return batch


def fit_new(reference, train, validation, params, head_config, seed, mode, domains, holdout, history):
    set_seed(seed)
    architecture = dict(reference['architecture'])
    for key in ('hidden_dim', 'question_dim', 'dropout'):
        if key in params:
            architecture[key] = params[key]
    architecture.update(relational_input_dim=train['X_rel'].shape[1], question_input_dim=train['X_query'].shape[1])
    preprocessor = FeaturePreprocessor(mode='none').fit(train['X_rel'])
    target = RegressionTargetTransformer(mode='signed_log1p').fit(train['y']) if reference['task_type'] == 'Regression' else None
    meta = deepcopy(reference)
    meta.update(architecture=architecture, params=params, seed=seed,
        stage='pretrain' if mode == 'source' else 'indomain_full',
        training_domains=domains, validation_domains=domains, holdout_domain=holdout,
        train_sample_ids=[r['sample_id'] for r in train['rows']],
        validation_sample_ids=[r['sample_id'] for r in validation['rows']], classification_threshold=.5)
    for key in ('task_spec_provenance', 'representation_fingerprint', 'aggregation_config', 'encoder_config',
                'node_encoder_config', 'node_encoder_fingerprint', 'representation_config'):
        meta[key] = train['manifest'][key]
    artifact = HeadArtifact(QueryConditionedTaskHead(**architecture), preprocessor, target, meta)
    config = {'head': dict(head_config, **params, seed=seed)}
    PredictionHeadTrainer(config)._optimize(artifact, train, validation, history_path=history)
    return artifact


def train_heads(configuration, cache, rows_root, output, mode='indomain', source_heads=None,
                domains=None, seed=42, device='cpu', epochs=200, patience=20):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    cfg = read_json(configuration); domain = cfg['domain']; output = Path(output)
    if (output / 'TRAINING_COMPLETE.json').exists() or any((output / k / 'model.pt').exists() for k in KINDS):
        raise FileExistsError('Use a fresh output directory; this entry point does not implicitly resume or overwrite checkpoints')
    reference_root = Path(cfg['prediction_heads'])
    reference = {k: read_json(reference_root / k / 'model_config.json') for k in KINDS}
    holdout = reference['Classification']['task_spec_provenance']['holdout_domain']
    source_mode = mode == 'source'
    active = [d for d in (domains or []) if d != domain] if source_mode else [domain]
    if source_mode:
        active = [d for d in active if read_json(Path(rows_root) / d / 'manifest.json')['status'] != 'not_applicable']
        if not active:
            raise ValueError('Source pretraining needs the other databases and their generated caches')
    head_config = dict(epochs=epochs, patience=patience, gradient_clip=1., device=device, cpu_threads=1,
                       classification_threshold=.5)
    selection = None
    if mode not in ('source', 'indomain'):
        if source_heads is None:
            raise ValueError('Target adaptation requires --source-heads with the matching held-out TSG')
        official = read_json(Path(rows_root) / domain / 'scorer_only/support_train.json')
        selection = task_selection(official, domain, seed, mode)
        write_json(output / 'selection.json', dict(seed=seed, domain=domain, budget=mode,
            unit='K_per_exact_task_id', validation='full separate target validation, extra labels', selections=selection))
    states = {}
    for kind in KINDS:
        train = concat_batches([batch_for(cache, rows_root, d, 'support/train', kind) for d in active])
        val = concat_batches([batch_for(cache, rows_root, d, 'support/validation', kind) for d in active])
        holdout = train['manifest']['task_spec_provenance']['holdout_domain']
        meta = reference[kind]
        params = meta.get('adaptation_params', meta['params']) if selection is not None else meta['params']
        if selection is None:
            if not len(train['rows']) or not len(val['rows']):
                raise ValueError(f'No usable generated train/validation inputs: {kind}')
            artifact = fit_new(meta, train, val, params, head_config, seed, mode, active, holdout,
                              output / (kind + '_epochs.json'))
            status = 'trained'
        else:
            selected = set(selection[kind])
            train = subset_batch(train, [i for i, r in enumerate(train['rows']) if r['sample_id'] in selected])
            set_seed(seed)
            artifact = HeadArtifact.load(Path(source_heads) / kind)
            if domain in artifact.metadata['training_domains']:
                raise ValueError('The supplied source head includes the target database')
            artifact.check_batch(train); artifact.check_batch(val)
            artifact.metadata.update(stage='full' if mode == 'full' else 'few', mode=mode, seed=seed,
                classification_threshold=.5, adaptation_params=params,
                source_checkpoint=str(Path(source_heads).resolve() / kind),
                requested_training_ids=selection[kind], train_sample_ids=[r['sample_id'] for r in train['rows']],
                validation_sample_ids=[r['sample_id'] for r in val['rows']],
                adaptation_protocol='K_per_exact_task_id_fixed_source_and_hyperparameters')
            for module in artifact.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = params['dropout']
            artifact.metadata['architecture']['dropout'] = params['dropout']
            status = ('source_retained_no_usable_training_inputs' if not train['rows'] else
                      'source_retained_no_usable_validation_inputs' if not val['rows'] else 'trained')
            if status == 'trained':
                artifact.metadata.update(training_domains=sorted(set(artifact.metadata['training_domains']) | {domain}),
                                         current_stage_domains=[domain])
                config = {'head': dict(head_config, **params, seed=seed, few_shot_learning_rate=params['learning_rate'])}
                PredictionHeadTrainer(config)._optimize(artifact, train, val, history_path=output / (kind + '_epochs.json'))
        artifact.metadata['adaptation_status'] = status
        artifact.save(output / kind)
        states[kind] = dict(status=status, usable_training_count=len(train['rows']), usable_validation_count=len(val['rows']),
            requested_training_count=len(selection[kind]) if selection is not None else len(train['rows']),
            best_epoch=artifact.metadata.get('best_epoch'), params=params)
    write_json(output / 'TRAINING_COMPLETE.json', dict(status='complete', seed=seed, mode=mode, domain=domain, heads=states))
    return states
