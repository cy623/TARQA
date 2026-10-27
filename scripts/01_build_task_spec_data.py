#!/usr/bin/env python
import argparse
from tarqa.config import load_config,resolve_path
from tarqa.data.discovery import discover_domains
from tarqa.task_spec.dataset import TaskSpecDatasetBuilder


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/task_spec.yaml');p.add_argument('--holdout',required=True,type=str.lower)
    p.add_argument('--stage',choices=['pretrain','full','few'],default='pretrain');p.add_argument('--k',type=int);p.add_argument('--output')
    a=p.parse_args();c=load_config(a.config);t=c['training']
    paths=discover_domains(resolve_path(c['data']['dataset_root'],c['_config_path']),resolve_path(c['data']['db_root'],c['_config_path']),c['data']['domains'])
    builder=TaskSpecDatasetBuilder(paths,generation_strategy=t.get('generation_strategy','track_last'),anonymize_schema=t.get('anonymize_schema',True),
        train_shot_choices=t.get('train_shot_choices',[0]),validation_shots=t.get('validation_shots',0),test_shots=t.get('test_shots',0),
        validation_ratio=0,seed=t.get('seed',42),include_variants=t.get('include_variants',False))
    output=resolve_path(a.output,c['_config_path']) if a.output else resolve_path(c['data']['processed_root'],c['_config_path'])/'task_spec'/f'holdout_{a.holdout}'/(f'few_{a.k}' if a.stage=='few' else a.stage)
    result=builder.build(a.holdout,output,stage=a.stage,few_shot_k=a.k)
    print({k:v for k,v in result.items() if k!='selected_train_uids'})

if __name__=='__main__':main()
