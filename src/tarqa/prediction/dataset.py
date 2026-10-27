"""Build model caches from released supervision; never replay or resplit QA."""
from pathlib import Path
import json
import numpy as np

from tarqa.data.releases import ReleaseDataset
from tarqa.prediction.representation import SchemaSemanticPool
from tarqa.prediction.question_encoder import render_question_representation_input
from tarqa.types import TaskSpec, BindingResult, EntityAnchor
from tarqa.prediction.generated_specs import (cache_spec_provenance, claim_cache,
    select_build_rows, schema_prompt, generated_feature_row)
from tarqa.utils.io import write_json, read_json

ROLES = ('support/train', 'support/validation', 'support/test', 'qa/test')


class PredictionDatasetBuilder:
    def __init__(self, paths, config, question_encoder, *, symbolic_time_adapter_path=None):
        self.paths, self.config, self.encoder = paths, config, question_encoder

    def build(self, output_dir, *, spec_source='gold', generated_spec_path=None, smoke_limit=None,
              reuse_roles=(), graph_resource_limits=None, progress_every=50, generated_store=None,
              roles=ROLES, rows_root=None):
        from tarqa.prediction.factory import build_relational_aggregator
        reuse_roles = set(reuse_roles)
        roles = tuple(roles)
        if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(ROLES):
            raise ValueError('Invalid build roles')
        if not reuse_roles <= set(roles) or progress_every < 1:
            raise ValueError('Invalid reuse roles or progress interval')
        if spec_source not in {'gold','generated'}:raise ValueError('Unknown Task Spec source')
        if generated_spec_path:raise ValueError('Use a provenance-bound generated_store')
        if (spec_source=='generated') != (generated_store is not None):
            raise ValueError('Generated mode requires a generated Task Spec store')
        if generated_store is not None and reuse_roles:
            raise ValueError('Generated caches resume LLM outputs automatically; rebuild features instead of --reuse-roles')
        provenance=generated_store.provenance if generated_store else {'source':'gold'}
        claim_cache(output_dir,provenance)
        schema,known_tables=schema_prompt(self.paths) if generated_store else (None,None)
        if smoke_limit is not None and not self.encoder.is_smoke:
            raise ValueError('Partial caches are restricted to --smoke; production includes every released sample.')
        # Cache building reads the supplied records directly, as requested.
        # Publisher hashes, execution audits and cross-record checks are not run.
        release = ReleaseDataset(self.paths, validate_rows=False)
        contract = dict(domain=self.paths.domain, validation_status='skipped_by_user_request',
                        database_path=str(self.paths.database), metadata_path=str(self.paths.metadata))
        print(f'{self.paths.domain}: release validation skipped', flush=True)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Remove the completion marker before any write, so a failed rebuild is unusable.
        (output_dir / 'manifest.json').unlink(missing_ok=True)
        graph_backend = self.config.get('representation', {}).get('backend') == 'hetero_gnn'
        if graph_backend:
            from tarqa.prediction.gnn_representation import FrozenLMGraphPool
            from tarqa.prediction.node_encoder import build_node_encoder
            pool = FrozenLMGraphPool(self.encoder, self.config.get('representation'),
                                     node_encoder=build_node_encoder(self.config, self.encoder))
        else:
            pool = SchemaSemanticPool(self.encoder, self.config.get('representation'))
        aggregator = build_relational_aggregator(self.paths, self.config)
        graph_resource_limits = dict(graph_resource_limits or {})
        if graph_resource_limits and not graph_backend:
            raise ValueError('Graph resource limits require the hetero_graph backend')
        for name, value in graph_resource_limits.items():
            if name not in {'max_nodes', 'max_edges'} or int(value) < 1:
                raise ValueError('Invalid graph resource limit')
            setattr(aggregator, name, int(value))
        manifest = dict(domain=self.paths.domain, format_version=5 if graph_backend else 4, spec_source=spec_source,
            task_spec_provenance=provenance, role_coverage={},
            representation_fingerprint=pool.fingerprint, encoder_fingerprint=self.encoder.fingerprint,
            encoder_config=self.encoder.config, representation_config=pool.config,
            aggregation_config=self.config.get('aggregation', {}),
            smoke_only=self.encoder.is_smoke, smoke_limit=smoke_limit,
            release_contract=contract, visibility_assumptions=release.config, roles={})
        if graph_backend:
            manifest.update(node_encoder_config=pool.node_encoder.config,
                            node_encoder_fingerprint=pool.node_encoder.fingerprint)
        if graph_resource_limits:
            # These are fail-fast memory guards, not graph sampling parameters.
            # Keep the configured aggregation policy unchanged; explicitly record
            # effective overrides and reused roles separately for provenance.
            manifest['runtime_graph_resource_limits'] = dict(
                max_nodes=aggregator.max_nodes, max_edges=aggregator.max_edges,
                graph_sampling='none', applies_to_roles=[r for r in roles if r not in reuse_roles])
        manifest['reused_roles'] = sorted(reuse_roles)
        try:
            for role in roles:
                rows = select_build_rows(release,role,smoke_limit,rows_root)
                if role in reuse_roles:
                    folder = output_dir / role
                    for name in ('features.npz', 'samples.json', 'feature_audit.json'):
                        if not (folder / name).is_file():
                            raise FileNotFoundError(f'Cannot reuse incomplete role: {folder / name}')
                    cached_rows = read_json(folder / 'samples.json')
                    if cached_rows != rows:
                        raise ValueError(f'Reuse requested but source rows changed: {role}')
                    with np.load(folder / 'features.npz', allow_pickle=False) as saved:
                        if saved['X_rel'].shape != (len(rows), pool.output_dim) or saved['X_query'].shape != (len(rows), self.encoder.dimension):
                            raise ValueError(f'Reuse feature dimensions differ: {role}')
                    manifest['roles'][role] = {task: sum(r['task_type']==task for r in rows)
                                              for task in ('Classification','Regression')}
                    print(self.paths.domain, role, len(rows), 'cached (reused)', flush=True)
                    continue
                original_rows=rows
                rows=[];features=[];audits=[];vectors=[];failures=[];successful_indices=[]
                # Group historical contexts by time, then restore release order for saved arrays.
                order=sorted(range(len(original_rows)),key=lambda i:(original_rows[i]['reference_time'],original_rows[i]['sample_id']))
                for processed,index in enumerate(order,1):
                    original=original_rows[index];row=original
                    phase='task_spec'
                    try:
                        if generated_store:
                            record=generated_store.get(schema,row['question'])
                            row=generated_feature_row(row,record,known_tables)
                        phase='graph'
                        result=aggregator.aggregate(TaskSpec.model_validate(row['spec']),
                            BindingResult(anchors=[EntityAnchor.model_validate(row['anchor'])]),sample_key=row['sample_id'])
                    except ValueError as error:
                        if not generated_store:raise
                        failures.append(dict(sample_id=original['sample_id'],uid=original['uid'],
                            parent_question_id=original['parent_question_id'],task_type=original['task_type'],
                            role=role,phase=phase,error=str(error)))
                        continue
                    rows.append(row);audits.append(result.audit);successful_indices.append(index)
                    if graph_backend:vectors.append(pool.transform(result.features))
                    else:features.append(result.features)
                    if processed % progress_every==0 or processed==len(original_rows):
                        print(self.paths.domain,role,f'{processed}/{len(original_rows)} features; {len(failures)} failed',flush=True)
                restore=sorted(range(len(rows)),key=lambda i:successful_indices[i])
                rows=[rows[i] for i in restore];audits=[audits[i] for i in restore]
                if graph_backend:
                    vectors=[vectors[i] for i in restore]
                    x_rel=np.stack(vectors) if vectors else np.empty((0,pool.output_dim),np.float32)
                else:
                    features=[features[i] for i in restore]
                    pool.prepare(k for feature in features for k in feature)
                    x_rel=np.stack([pool.transform(f) for f in features]) if rows else np.empty((0,pool.output_dim),np.float32)
                texts=[render_question_representation_input(r['question'],TaskSpec.model_validate(r['spec'])) for r in rows]
                x_query=self.encoder.encode(texts)
                folder=output_dir/role;folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'features.npz',X_rel=x_rel,X_query=x_query)
                write_json(folder/'samples.json',rows);write_json(folder/'feature_audit.json',audits)
                if generated_store:write_json(folder/'failed_samples.json',failures)
                manifest['roles'][role]={task:sum(r['task_type']==task for r in rows) for task in ('Classification','Regression')}
                manifest['role_coverage'][role]={task:dict(
                    total=sum(r['task_type']==task for r in original_rows),
                    successful=sum(r['task_type']==task for r in rows),
                    failed=sum(r['task_type']==task for r in failures)) for task in ('Classification','Regression')}
                print(self.paths.domain,role,len(rows),'cached;',len(failures),'failed',flush=True)
        finally:
            aggregator.close()
        write_json(output_dir/'manifest.json', manifest)
        return manifest


def load_cached_role(cache_root, domain, role, task_type, *, allow_all_failed=False):
    if role not in ROLES:
        raise ValueError(role)
    root = Path(cache_root) / domain
    manifest = read_json(root/'manifest.json')
    if manifest['domain'] != domain or manifest['format_version'] not in {4, 5}:
        raise ValueError('Wrong cache domain/version')
    if manifest['format_version'] == 5 and manifest.get('representation_config', {}).get('backend') != 'hetero_gnn':
        raise ValueError('Version 5 requires the typed graph representation')
    rows = read_json(root/role/'samples.json')
    with np.load(root/role/'features.npz', allow_pickle=False) as arrays:
        x_rel, x_query = arrays['X_rel'], arrays['X_query']
    if len(rows) != len(x_rel) or len(rows) != len(x_query):
        raise ValueError('Cache length mismatch')
    if any(r['domain'] != domain or r['role'] != role for r in rows):
        raise ValueError('Cache role/domain mismatch')
    counts=manifest.get('role_coverage',{}).get(role,{}).get(task_type,{})
    if not allow_all_failed and role in {'support/train','support/validation'} and counts.get('total',0)>0 and not counts.get('successful',1):
        raise ValueError(f'All generated {role}/{task_type} inputs failed: {root/role/"failed_samples.json"}')
    selected = np.array([r['task_type']==task_type for r in rows],dtype=bool)
    subset = [r for r, keep in zip(rows,selected) if keep]
    return dict(X_rel=x_rel[selected], X_query=x_query[selected],
                y=np.array([r['label'] for r in subset],np.float32), rows=subset, manifest=manifest)


def concat_batches(batches):
    if not batches:
        raise ValueError('No source domains')
    specs={json.dumps(cache_spec_provenance(b['manifest']),sort_keys=True) for b in batches}
    if len(specs)!=1:raise ValueError('Cannot mix gold/generated or different TSG folds/models in one head dataset')
    fingerprints = {b['manifest']['representation_fingerprint'] for b in batches}
    if len(fingerprints) != 1:
        raise ValueError('Caches use different encoders/representations. Rebuild all domains with the same frozen checkpoint.')
    if len({json.dumps(b['manifest']['aggregation_config'], sort_keys=True) for b in batches}) != 1:
        raise ValueError('Source caches use different aggregation policies')
    return dict(X_rel=np.concatenate([b['X_rel'] for b in batches]),
                X_query=np.concatenate([b['X_query'] for b in batches]),
                y=np.concatenate([b['y'] for b in batches]),
                rows=[r for b in batches for r in b['rows']], manifest=batches[0]['manifest'])


def subset_batch(batch, indices):
    indices = np.asarray(indices,dtype=np.int64)
    return dict(X_rel=batch['X_rel'][indices], X_query=batch['X_query'][indices], y=batch['y'][indices],
                rows=[batch['rows'][i] for i in indices],manifest=batch['manifest'])
