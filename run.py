#!/usr/bin/env python
"""TARQA public entry point; runs directly without pyproject.toml."""
import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import runpy
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'baselines/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('setup', help='Use the published data and relocate pretrained components')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--resources-root', type=Path, required=True)
    p.add_argument('--llama-base', type=Path, required=True)
    p.add_argument('--qwen-base', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=Path('work'))
    p.add_argument('--domains', nargs='+')
    p = commands.add_parser('build-features', help='Generate Task Specs and encode historical graphs once')
    p.add_argument('--configuration', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--rows-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--domains', nargs='+')
    p.add_argument('--roles', nargs='+', default=['support/train','support/validation','qa/test'],
                   choices=['support/train','support/validation','qa/test'])
    p.add_argument('--encoder-device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--task-spec-config', type=Path, help='Optional configuration for a newly trained TSG')
    p.add_argument('--task-spec-adapter', type=Path, help='Matching newly trained pretrain/final adapter')
    p = commands.add_parser('train-heads', help='Train heads or adapt source heads under per-task K-shot')
    for name in ('configuration','cache-root','rows-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=['indomain','source','few_1','few_3','few_5','few_10','full'], default='indomain')
    p.add_argument('--source-heads', type=Path)
    p.add_argument('--domains', nargs='+', help='All source databases for --mode source')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=20)
    p = commands.add_parser('evaluate', help='Compute current Classification, Regression and Reasoning metrics')
    p.add_argument('--configuration', type=Path, required=True)
    p.add_argument('--rows-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--backbone', choices=['llama8b','qwen32b'], required=True)
    p.add_argument('--task', choices=['all','prediction','reasoning'], default='all')
    p.add_argument('--cache-root', type=Path, help='Optional frozen-feature prediction replay; not fresh TSG inference')
    p.add_argument('--head-root', type=Path, help='Evaluate newly trained heads')
    p.add_argument('--limit', type=int, help='Partial check: at most N per output type and N Reasoning questions')
    for name in ('build-tsg-data','train-tsg','eval-tsg'):
        commands.add_parser(name, add_help=False, help='Forward arguments to the original TSG entry point')
    # Preserve the original training entry points' own argument parsing/help.
    if len(sys.argv) > 1 and sys.argv[1] in ('build-tsg-data','train-tsg','eval-tsg'):
        names = {'build-tsg-data':'01_build_task_spec_data.py','train-tsg':'02_train_task_spec_lora.py',
                 'eval-tsg':'03_eval_task_spec.py'}
        script = ROOT / 'scripts' / names[sys.argv[1]]
        sys.argv = [str(script), *sys.argv[2:]]
        runpy.run_path(str(script), run_name='__main__')
        return
    args = parser.parse_args()
    if args.action == 'setup':
        import yaml
        from tarqa.public_data import prepare_data
        from tarqa.utils.io import write_json
        work = args.work_dir.resolve()
        dataset, rows = prepare_data(args.data_root, work, args.domains)
        spec = importlib.util.spec_from_file_location('tarqa_runtime_prepare', ROOT / 'scripts/prepare_runtime.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        runtime = module.prepare(args.resources_root, work / 'runtime', dataset, args.llama_base, args.qwen_base)
        for backbone, model in [('llama8b',args.llama_base),('qwen32b',args.qwen_base)]:
            cfg = yaml.safe_load((ROOT / f'configs/task_spec_{backbone}.yaml').read_text())
            cfg['data'].update(dataset_root=str(dataset), db_root=str(dataset), processed_root=str(work/'processed'),
                               domains=[d.name for d in dataset.iterdir() if d.is_dir()])
            cfg['model'].update(model_path=str(model.resolve()), tokenizer_path=str(model.resolve()))
            cfg['training']['output_root'] = str(work / 'tsg' / backbone)
            path = work / f'configs/task_spec_{backbone}.yaml'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(work / 'settings.json', dict(dataset_root=str(dataset), rows_root=str(rows),
            runtime_root=str(runtime), resources_root=str(args.resources_root.resolve())))
    elif args.action == 'build-features':
        from tarqa.public_training import build_features
        build_features(args.configuration, args.dataset_root, args.rows_root, args.cache_root,
            args.domains, args.roles, args.encoder_device, args.generate_only,
            args.task_spec_config, args.task_spec_adapter)
    elif args.action == 'train-heads':
        from tarqa.public_training import train_heads
        train_heads(args.configuration,args.cache_root,args.rows_root,args.output,args.mode,
            args.source_heads,args.domains,args.seed,args.device,args.epochs,args.patience)
    elif args.action == 'evaluate':
        from tarqa.public_evaluation import evaluate_prediction, evaluate_reasoning
        from tarqa.utils.io import read_json, write_json
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        cfg = read_json(args.configuration)
        (args.output / 'EVALUATION_COMPLETE.json').unlink(missing_ok=True)
        if args.task in ('all','prediction'):
            evaluate_prediction(args.configuration,args.rows_root,args.output,args.dataset_root,
                                args.cache_root,args.head_root,args.limit)
        if args.task in ('all','reasoning'):
            reason_cfg = args.runtime_root / f"configurations/reasoning_paper/{args.backbone}/{cfg['domain']}.json"
            evaluate_reasoning(reason_cfg,args.runtime_root,args.backbone,args.output,args.dataset_root,args.limit)
        write_json(args.output / 'EVALUATION_COMPLETE.json', dict(status='complete',
            domain=cfg['domain'], backbone=args.backbone, tasks=args.task, partial_test=bool(args.limit),
            cached_prediction_features=bool(args.cache_root),
            runtime_llm_max_memory=os.environ.get('TARQA_LLM_MAX_MEMORY'),
            cuda_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')))


if __name__ == '__main__':
    main()
