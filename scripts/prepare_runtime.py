#!/usr/bin/env python
"""Relocate the selected TARQA parameter release without modifying its weights."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def replace(value, mapping):
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, str(new))
        return value
    if isinstance(value, list):
        return [replace(x, mapping) for x in value]
    if isinstance(value, dict):
        return {k: replace(v, mapping) for k, v in value.items()}
    return value


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f'Existing runtime file differs: {target}')
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare(package, runtime, dataset_root, llama_base, qwen_base):
    package, runtime = Path(package).resolve(), Path(runtime).resolve()
    for path in (dataset_root, llama_base, qwen_base):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    index = read(package / 'checkpoint_index.json')
    mapping = {
        '@RESOURCE_ROOT@': package,
        '@RUNTIME_ROOT@': runtime,
        '@DATASET_ROOT@': Path(dataset_root).resolve(),
        '@LLAMA_BASE@': Path(llama_base).resolve(),
        '@QWEN_BASE@': Path(qwen_base).resolve(),
        '@ENCODER@': package / 'encoders/all-MiniLM-L6-v2',
        '@RUN_ROOT@': runtime / 'outputs',
    }
    provenances = {}
    for entry in index['tsg_adapters']:
        rel = Path(entry['directory'])
        source, target = package / rel, runtime / rel
        for path in source.iterdir():
            destination = target / path.name
            if path.name == 'adapter_config.json':
                config = replace(read(path), mapping)
                config['base_model_name_or_path'] = str(mapping[
                    '@LLAMA_BASE@' if entry['backbone'] == 'llama8b' else '@QWEN_BASE@'])
                save(destination, config)
            elif path.suffix == '.json' and path.name != 'tokenizer.json':
                save(destination, replace(read(path), mapping))
            else:
                link_or_copy(path, destination)
        provenances[entry['backbone'], entry['holdout']] = target
    for entry in index['prediction_heads']:
        rel = Path(entry['directory'])
        source, target = package / rel, runtime / rel
        for path in source.iterdir():
            if path.name == 'model_config.json':
                continue
            link_or_copy(path, target / path.name)
        meta = replace(read(source / 'model_config.json'), mapping)
        provenance = deepcopy(meta['task_spec_provenance'])
        adapter = provenances[entry['backbone'], entry['tsg_holdout']]
        provenance['adapter'] = identity(adapter / 'adapter_model.safetensors')
        provenance['adapter_config'] = identity(adapter / 'adapter_config.json')
        provenance['training_manifest'] = identity(adapter / 'training_manifest.json')
        meta['task_spec_provenance'] = provenance
        save(target / 'model_config.json', meta)
    for path in sorted((package / 'configurations').rglob('*.json')):
        save(runtime / path.relative_to(package), replace(read(path), mapping))
    result = dict(status='complete', datasets=index['datasets'],
        n_tsg_adapters=len(index['tsg_adapters']), n_prediction_heads=len(index['prediction_heads']),
        weight_values_changed=False, dataset_root=str(Path(dataset_root).resolve()),
        configuration_root=str(runtime / 'configurations'))
    save(runtime / 'RUNTIME_READY.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return runtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--llama-base', type=Path, required=True)
    parser.add_argument('--qwen-base', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.package_root, args.runtime_dir, args.dataset_root, args.llama_base, args.qwen_base)
