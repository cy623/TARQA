"""Frozen TSG predictions for fold-specific head features; no gold fallback."""
from copy import deepcopy
from pathlib import Path
import gc
import hashlib
import json

from tarqa.config import resolve_path
from tarqa.data.metadata import SchemaCatalog
from tarqa.task_spec.prompts import SYSTEM_PROMPT, render_user_prompt
from tarqa.task_spec.serialization import serialize_schema
from tarqa.types import TaskSpec, Track
from tarqa.utils.io import read_json, write_json


def resolve_cache_root(config, holdout=None, override=None):
    value=str(override or config['data']['prediction_cache_root'])
    if '{holdout}' in value:
        if not holdout:raise ValueError('This cache is fold-specific; provide --holdout')
        value=value.format(holdout=holdout.lower())
    return resolve_path(value,config['_config_path'])


def cache_spec_provenance(manifest):
    return manifest.get('task_spec_provenance', {'source':manifest.get('spec_source','gold')})


def select_build_rows(release, role, smoke_limit=None, rows_root=None):
    rows=(read_json(Path(rows_root)/release.domain/'scorer_only'/(role.replace('/','_')+'.json'))
          if rows_root else release.prediction_rows(role))
    if smoke_limit is None:return rows
    if role=='support/test':
        queries=select_build_rows(release,'qa/test',smoke_limit,rows_root)
        parents={r['uid'] for r in queries}
        return [r for r in rows if r['uid'] in parents]
    return [r for task in ('Classification','Regression')
            for r in [x for x in rows if x['task_type']==task][:smoke_limit]]


def schema_prompt(paths):
    catalog=SchemaCatalog.from_files(paths.domain,paths.metadata,paths.profiling,database_path=paths.database)
    text=serialize_schema(catalog)
    if paths.domain=='ncaa':
        text+="\nTime convention: an observation day denotes the NEXT day's midnight cutoff; game facts become available at next-day midnight."
    return text,set(catalog.table_names)


def file_identity(path):
    path=Path(path).resolve();info=path.stat()
    return dict(path=str(path),size=info.st_size,mtime_ns=info.st_mtime_ns)


def tsg_provenance(model_cfg,adapter,holdout):
    adapter=Path(adapter).resolve()
    training=read_json(adapter/'training_manifest.json')
    if training['holdout_domain']!=holdout or training['stage']!='pretrain':
        raise ValueError('Generated head caches require the source-pretrained TSG for this holdout')
    if training.get('selected_train_uids',{}).get(holdout):
        raise ValueError('The source TSG includes target-domain training')
    acfg=read_json(adapter/'adapter_config.json')
    if Path(acfg['base_model_name_or_path']).resolve()!=Path(model_cfg['model_path']).resolve():
        raise ValueError('TSG adapter and configured base model differ')
    return dict(source='generated',holdout_domain=holdout,stage='pretrain',
        adapter=file_identity(adapter/'adapter_model.safetensors'),
        adapter_config=file_identity(adapter/'adapter_config.json'),
        training_manifest=file_identity(adapter/'training_manifest.json'),model_config=model_cfg,
        prompt_protocol='08 schema + question, track_last, zero demonstrations',
        generation=dict(do_sample=False,num_beams=1),
        invalid_policy='record failure; no gold fallback; no replacement of cutoff or task type')


def claim_cache(directory,provenance):
    """Do not overwrite an existing gold cache or a different fold/model cache."""
    directory=Path(directory);marker=directory/'task_spec_source.json'
    if marker.exists() and read_json(marker)!=provenance:
        raise ValueError(f'Existing cache belongs to a different Task Spec source: {directory}')
    manifest=directory/'manifest.json'
    if manifest.exists() and cache_spec_provenance(read_json(manifest))!=provenance:
        raise ValueError(f'Cannot overwrite a different Task Spec cache: {directory}')
    write_json(marker,provenance)


class GeneratedSpecStore:
    def __init__(self,path,provenance,*,model_cfg=None,adapter=None,generator=None):
        self.path=Path(path);self.provenance=provenance
        self.model_cfg,self.adapter=model_cfg,adapter
        self.generator=generator;self.records={}
        if self.path.exists():
            lines=self.path.read_text().splitlines()
            for i,line in enumerate(lines):
                if not line.strip():continue
                try:record=json.loads(line)
                except json.JSONDecodeError:
                    if i!=len(lines)-1:raise
                    # Interrupted last append: retain completed records and regenerate this one.
                    self.path.write_text('\n'.join(lines[:i])+'\n');break
                self.records[record['input_key']]=record

    def key(self,schema,question):
        return hashlib.sha256(json.dumps(dict(provenance=self.provenance,schema=schema,
            question=question,system_prompt=SYSTEM_PROMPT),sort_keys=True).encode()).hexdigest()

    def get(self,schema,question,*,generate=False):
        key=self.key(schema,question)
        if key in self.records:return self.records[key]
        if not generate:raise ValueError('Missing generated Task Spec; run the generation phase first')
        if self.generator is None:
            from tarqa.task_spec.inference import LocalTaskSpecGenerator
            self.generator=LocalTaskSpecGenerator(self.model_cfg,self.adapter)
        record=dict(input_key=key,question=question)
        try:
            generated=self.generator.generate(SYSTEM_PROMPT,render_user_prompt(schema,question,generation_strategy='track_last'))
            record.update(spec=generated.spec.model_dump(mode='json'),raw_text=generated.raw_text,error=None)
        except (ValueError,TypeError) as error:
            record.update(spec=None,raw_text=getattr(error,'raw_text',None),error=f'{type(error).__name__}: {error}')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as target:
            target.write(json.dumps(record,ensure_ascii=False)+'\n');target.flush()
        self.records[key]=record
        return record

    def close(self):
        self.generator=None;gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def generated_feature_row(row,record,known_tables):
    """Keep supervision separate; model features use only the predicted Task Spec."""
    if record['error']:raise ValueError(record['error'])
    spec=TaskSpec.model_validate(record['spec'])
    if spec.validate_dependencies():raise ValueError('; '.join(spec.validate_dependencies()))
    if spec.track!=Track.PREDICTION:raise ValueError('LLM routed the prediction example to Reasoning')
    if spec.task_type.value!=row['task_type']:raise ValueError('LLM predicted the wrong task type')
    if spec.reference_time!=TaskSpec.canonical_time(row['reference_time']):
        raise ValueError('LLM reference_time differs from the supplied observation cutoff')
    unknown=set(spec.relevant_tables)-set(known_tables)
    if unknown:raise ValueError(f'LLM predicted unknown tables: {sorted(unknown)}')
    result=deepcopy(row)
    result.update(spec=spec.model_dump(mode='json'),relevant_tables=list(spec.relevant_tables),
                  reference_time=spec.reference_time,spec_source='generated',generated_spec_key=record['input_key'])
    return result
