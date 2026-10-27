"""Read the published dataset directly and prepare local feature-building views."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

from tarqa.data.discovery import discover_domain
from tarqa.data.qa_loader import QARecord
from tarqa.data.releases import ReleaseDataset, scalar_label, stamp
from tarqa.task_identity import task_definition, digest
from tarqa.types import TaskSpec, Track, TaskType
from tarqa.utils.io import read_json, write_json

DOMAINS = ('olist', 'stats', 'ncaa', 'hepatitis', 'financial', 'dallas', 'airline')


def event_key(row):
    anchor = row['anchor']; value = anchor['value']
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return digest([row['task_id'], anchor['table'].lower(), anchor['key'].lower(),
                   str(value), stamp(row['reference_time'])])


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f'Data path changed: {destination}; use a new work directory')
    elif not destination.exists():
        destination.symlink_to(source.resolve())


def prepare_data(source, work, domains=None):
    """No label SQL execution, new random splits or source-file changes."""
    source, work = Path(source).resolve(), Path(work).resolve()
    config_root = Path(__file__).resolve().parents[2] / 'configs/data'
    published = source / 'manifest.json'
    manifest = read_json(published) if published.exists() else {}
    is_public = isinstance(manifest.get('domains'), dict)
    domains = list(domains or (manifest['domains'] if is_public else DOMAINS))
    loader = None
    if is_public:
        spec = importlib.util.spec_from_file_location('rdl_bench_public_reader', source / 'load_data.py')
        loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    summary = []
    for domain in domains:
        destination = work / 'dataset' / domain
        destination.mkdir(parents=True, exist_ok=True)
        if is_public:
            info = manifest['domains'][domain]
            qa = {s: loader.qa(domain, s) for s in ('train', 'validation', 'test')}
            support = {s: loader.support(domain, s) for s in ('train', 'validation', 'test')}
            metadata, database = source / info['metadata'], source / info['database']
            cfg = read_json(config_root / f'{domain}.json')
            times = source / info['reference_times'] if info.get('reference_times') else None
            minimum = info.get('prediction_protocol', {}).get('earliest_permitted_reference_time')
        else:
            paths = discover_domain(source, source.parent / 'DB', domain)
            release = ReleaseDataset(paths, validate_rows=False)
            qa = {s: [deepcopy(q.payload) for q in release.qa(s)] for s in ('train', 'validation', 'test')}
            support = {s: release.support(s) for s in ('train', 'validation', 'test')}
            metadata, database, cfg = paths.metadata, paths.database, release.config
            times = release.root / 'qa_reference_times.json'
            minimum = cfg.get('support_test_reference_time_min')
        if not metadata.is_file() or not database.is_file():
            raise FileNotFoundError(f'Need released metadata and database: {metadata}, {database}')
        write_json(destination / 'config.json', cfg)
        for split in qa:
            write_json(destination / f'qa/{split}.json', qa[split])
            write_json(destination / f'support/{split}.json', support[split])
        _link(metadata, destination / 'source' / metadata.name)
        _link(database, destination / 'source' / database.name)
        if times and times.is_file():
            shutil.copy2(times, destination / 'qa_reference_times.json')
        parents = {}; trained_tasks = set()
        for split, records in qa.items():
            for raw in records:
                q = QARecord(domain, raw)
                if q.query_type != 'Predictive':
                    continue
                task, _ = task_definition(q, domain)
                parents[q.instance_id] = (q, task)
                if split == 'train':
                    trained_tasks.add(task)
        role_rows = {}
        for split in ('train', 'validation', 'test'):
            rr = []
            for source_row in support[split]:
                row = deepcopy(source_row); _, task = parents[row['parent_question_id']]
                row.update(role='support/' + split,
                    uid=f"{domain}::{row['sample_id'] if split == 'test' else row['parent_question_id']}",
                    task_id=task, task_seen_in_training=task in trained_tasks, eligible=True,
                    reference_time=stamp(row['reference_time']))
                rr.append(row)
            role_rows['support/' + split] = rr
        candidates = []
        for raw in qa['test']:
            q = QARecord(domain, raw)
            if q.query_type != 'Predictive':
                continue
            _, task = parents[q.instance_id]
            kind = 'Classification' if q.answer_nature == 'Boolean' else 'Regression'
            candidates.append(dict(sample_id=q.uid, uid=q.uid, domain=domain,
                parent_question_id=q.instance_id, parent_split='test', role='qa/test', source_role='qa/test',
                question=q.primary_question, anchor=q.anchor.model_dump(mode='json'),
                reference_time=stamp(q.reference_time), task_type=kind, relevant_tables=q.relevant_tables,
                label=scalar_label(q.raw_answer, kind), task_id=task,
                task_seen_in_training=task in trained_tasks, eligible=True))
        for raw in role_rows['support/test']:
            row = deepcopy(raw); row.update(role='qa/test', source_role='support/test')
            candidates.append(row)
        blocked = {event_key(row) for role in ('support/train', 'support/validation') for row in role_rows[role]}
        for split in ('train', 'validation'):
            for raw in qa[split]:
                q = QARecord(domain, raw)
                if q.query_type == 'Predictive':
                    blocked.add(event_key(dict(task_id=parents[q.instance_id][1],
                        anchor=q.anchor.model_dump(mode='json'), reference_time=q.reference_time)))
        seen = set(); test = []
        for row in candidates:
            key = event_key(row)
            if key in blocked or key in seen or (minimum and stamp(row['reference_time']) < stamp(minimum)):
                continue
            seen.add(key); test.append(row)
        role_rows['qa/test'] = test
        for role in ('support/train', 'support/validation', 'qa/test'):
            for row in role_rows[role]:
                row['spec'] = TaskSpec(relevant_tables=row['relevant_tables'], reference_time=row['reference_time'],
                    task_type=TaskType(row['task_type']), track=Track.PREDICTION).model_dump(mode='json')
            write_json(work / 'rows' / domain / 'scorer_only' / (role.replace('/', '_') + '.json'), role_rows[role])
        counts = {k: len(role_rows[k]) for k in ('support/train', 'support/validation', 'qa/test')}
        status = 'ready' if parents else 'not_applicable'
        write_json(work / 'rows' / domain / 'manifest.json', dict(domain=domain, status=status, counts=counts,
            protocol='released_support_train_validation_and_deduplicated_merged_test', label_sql_executed=False))
        summary.append(dict(domain=domain, status=status, **counts))
        print(domain, counts, flush=True)
    write_json(work / 'data_summary.json', summary)
    return work / 'dataset', work / 'rows'
