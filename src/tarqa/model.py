"""Unified frozen TARQA inference from existing component checkpoints.

One TSG chooses the track and prediction type. The same loaded base model,
with its LoRA adapter temporarily disabled, generates SQL. No label or expected
task type is an argument to answer().
"""
from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from tarqa.data.discovery import discover_domain
from tarqa.eval.e2e import EndToEndPipeline
from tarqa.task_spec.inference import LocalTaskSpecGenerator
from tarqa.utils.io import read_json


class _SharedBaseChat:
    def __init__(self, generator, config):
        import torch
        from tarqa_baselines.reasoning_llm import LocalChatModel
        self._client=LocalChatModel.__new__(LocalChatModel)
        self._client.torch=torch
        self._client.config=dict(config)
        self._client.model=generator.model
        self._client.tokenizer=generator.tokenizer

    def generate(self, messages, *, stage='direct'):
        with self._client.model.disable_adapter():
            return self._client.generate(messages,stage=stage)


class _SQLModel:
    def __init__(self, client, backend, trace_root, demonstration_bank=None, num_examples=3):
        self.client,self.backend=client,backend
        self.spec=None
        self.uid=None
        self.trace_root=Path(trace_root)
        self.last_details=None
        if backend=='din_sql':
            from tarqa_baselines.tsg_din_sql import ArchivedDemoBank,TSGDINSQLEngine
            rows=read_json(demonstration_bank)
            if isinstance(rows,dict):rows=rows['examples']
            self.engine=TSGDINSQLEngine(client,ArchivedDemoBank(rows,num_examples=num_examples),
                trace_root=self.trace_root,mode='local_schema')
        elif backend!='direct':raise ValueError(f'Unknown SQL backend: {backend}')

    def generate(self, question, schema_text, *, error=None, sql=None):
        if error or sql:raise ValueError('The saved paired executor uses zero execution-feedback repairs')
        if self.backend=='direct':
            from tarqa_baselines.reasoning_llm import direct_generate
            result,self.last_details=direct_generate(self.client,question,schema_text)
            return result
        if self.spec is None:raise ValueError('The generated Task Spec must precede SQL generation')
        self.engine.configure(dict(local_schema=schema_text,task_spec=self.spec.model_dump(mode='json')))
        result,self.last_details=self.engine.generate(self.uid,question)
        return result


class _TaskSpecHook:
    def __init__(self, generator, executor):self.generator,self.executor=generator,executor
    def __getattr__(self,name):return getattr(self.generator,name)
    def generate(self,*args,**kwargs):
        result=self.generator.generate(*args,**kwargs)
        self.executor.spec=result.spec
        return result


class TARQAModel:
    """Load TSG, frozen representations and trained heads once for one database.

    Config paths are resolved relative to the supplied JSON config file. A trusted
    observation cutoff may be supplied by the benchmark; it is used for the
    existing explicit time-consistency gate, not passed into TSG generation.
    """
    @classmethod
    def from_pretrained(cls, configuration: str | Path | dict[str,Any]):
        if isinstance(configuration,dict):
            config=deepcopy(configuration)
            base=Path.cwd()
        else:
            path=Path(configuration).resolve();config=read_json(path);base=path.parent
        return cls(config,base)

    def __init__(self, config, config_directory):
        self.config=deepcopy(config)
        self.config_directory=Path(config_directory)
        self.paths=discover_domain(self.path(config['dataset_root']),None,config['domain'])
        self.head_root=self.path(config['prediction_heads'])
        metadata=read_json(self.head_root/'Classification/model_config.json')
        provenance=metadata['task_spec_provenance']
        self.holdout=provenance['holdout_domain']
        self.generator=LocalTaskSpecGenerator(provenance['model_config'],
            Path(provenance['adapter']['path']).parent)
        self._build_pipeline(metadata)

    def path(self,value):
        p=Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (self.config_directory/p).resolve()

    def _build_pipeline(self,metadata):
        from tarqa_baselines.export import schema_ddl
        sql_config=self.config.get('sql_generation',dict(dtype='bfloat16',load_in_4bit=True,
            max_input_tokens=16384,max_new_tokens=1024,num_beams=1))
        client=_SharedBaseChat(self.generator,sql_config)
        bank=self.config.get('demonstration_bank')
        self.sql=_SQLModel(client,self.config['sql_backend'],self.path(self.config['trace_root']),
            self.path(bank) if bank else None,self.config.get('num_examples',3))
        qconfig=deepcopy(metadata['encoder_config'])
        nconfig=deepcopy(metadata.get('node_encoder_config',qconfig))
        # Runtime device/batching do not alter the MiniLM fingerprint.
        qconfig['device']=self.config.get('encoder_device','cpu')
        nconfig['device']=self.config.get('encoder_device','cpu')
        prediction=dict(question_encoder=qconfig,node_encoder=nconfig,
            representation=metadata['representation_config'],aggregation=metadata['aggregation_config'])
        self.pipeline=EndToEndPipeline(self.paths,_TaskSpecHook(self.generator,self.sql),self.sql,
            dict(schema_policy='tsg_selected',add_bridge_tables=True,max_repair_attempts=0,timeout_seconds=120),
            prediction,self.head_root,None,holdout_domain=self.holdout)
        self.pipeline.reasoning.schema_renderer=lambda catalog:'-- SQL dialect: DuckDB\n'+schema_ddl(catalog)

    def use_prediction_heads(self,head_root):
        """Switch only the trained heads; keep the same matching TSG loaded."""
        head_root=self.path(head_root)
        metadata=read_json(head_root/'Classification/model_config.json')
        old=read_json(self.head_root/'Classification/model_config.json')
        for key in ('task_spec_provenance','representation_fingerprint','aggregation_config'):
            if metadata[key]!=old[key]:raise ValueError(f'Cannot switch incompatible heads: {key}')
        self.pipeline.close()
        self.head_root=head_root
        self._build_pipeline(metadata)

    def answer(self,question,*,anchor=None,observation_cutoff=None,uid=None):
        from tarqa.types import EntityAnchor
        if isinstance(anchor,dict):anchor=EntityAnchor.model_validate(anchor)
        self.sql.uid=uid or hashlib.sha256(question.encode()).hexdigest()
        return self.pipeline.answer(question,anchor=anchor,reference_time=observation_cutoff)

    def close(self):
        self.pipeline.close()
        self.pipeline=None
        self.sql=None
        self.generator=None
        gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
