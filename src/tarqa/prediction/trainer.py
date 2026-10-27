"""Six-domain shared heads and explicit target-domain adaptation protocols."""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from tarqa.prediction.dataset import load_cached_role, concat_batches, subset_batch
from tarqa.prediction.generated_specs import cache_spec_provenance
from tarqa.prediction.normalization import FeaturePreprocessor, RegressionTargetTransformer
from tarqa.prediction.task_head import QueryConditionedTaskHead
from tarqa.eval.prediction_eval import domain_macro_metrics
from tarqa.utils.io import read_json, write_json
from tarqa.utils.seed import set_seed, stable_int


def classification_metrics(y, logits, threshold=0.5):
    if not len(y):
        return {'n': 0, 'status': 'no_examples'}
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    predictions = probabilities >= threshold
    return dict(n=len(y), positives=int(np.sum(y)), negatives=int(len(y)-np.sum(y)),
        accuracy=float(accuracy_score(y,predictions)), f1=float(f1_score(y,predictions,zero_division=0)),
        auroc=float(roc_auc_score(y, probabilities)) if len(np.unique(y))==2 else None,
        auroc_status='defined' if len(np.unique(y))==2 else 'undefined_single_class',
        brier=float(np.mean((probabilities-y)**2)))


def regression_metrics(y, prediction):
    if not len(y):
        return {'n': 0, 'status': 'no_examples'}
    difference = np.asarray(prediction,np.float64)-np.asarray(y,np.float64)
    signed_log = lambda a: np.sign(a)*np.log1p(np.abs(a))
    return dict(n=len(y), mae=float(np.mean(abs(difference))), rmse=float(np.sqrt(np.mean(difference**2))),
        signed_log_mae=float(np.mean(abs(signed_log(prediction)-signed_log(y)))),
        constant_target=bool(len(np.unique(y))==1), all_zero_target=bool(np.all(y==0)))


def balanced_weights(rows):
    """Equal total weight per domain, then per parent, regardless of support count."""
    parent_counts=Counter(r['uid'] for r in rows)
    parent_domains={r['uid']:r['domain'] for r in rows}
    domains=Counter(parent_domains.values())
    return np.asarray([1/(parent_counts[r['uid']]*domains[r['domain']]) for r in rows], np.float64)


def choose_support_indices(rows, parent_uid, k, seed=42):
    if k < 1:
        raise ValueError('few-shot k must be positive')
    candidates = [i for i,r in enumerate(rows) if r['uid']==parent_uid and r['role']=='support/test']
    return sorted(candidates,key=lambda i: stable_int(f"{seed}:{rows[i]['sample_id']}"))[:k]


def read_few_shot_selection(path, holdout, k):
    """Reuse the exact target qa/train selection made by 01 --stage few."""
    if k is None or k < 1 or path is None:
        raise ValueError('Domain few-shot requires --k and the 01 few-shot manifest')
    manifest = read_json(path)
    if (manifest.get('stage') != 'few' or manifest.get('holdout_domain') != holdout
            or manifest.get('requested_k') != k):
        raise ValueError('Few-shot manifest stage/holdout/k mismatch')
    selected = manifest.get('selected_train_uids', {}).get(holdout, [])
    if (not selected or len(selected) > k or len(set(selected)) != len(selected)
            or any(not uid.startswith(f'{holdout}::') for uid in selected)):
        raise ValueError('Invalid target qa/train selection')
    return selected


def check_training_splits(train, validation, domains):
    for batch, role in ((train, 'support/train'), (validation, 'support/validation')):
        if any(r['role'] != role or r['domain'] not in domains for r in batch['rows']):
            raise ValueError(f'Unexpected domain/role in {role}')
    if {r['uid'] for r in train['rows']} & {r['uid'] for r in validation['rows']}:
        raise ValueError('Training and validation contain the same parent QA')


def check_head_provenance(metadata, holdout, domain, mode, few_shot_protocol='per_query'):
    if metadata['holdout_domain'] != holdout:
        raise ValueError('Wrong held-out fold checkpoint')
    expected = 'full' if mode == 'full' else ('few' if mode == 'few' and few_shot_protocol == 'domain' else 'pretrain')
    if metadata['stage'] != expected:
        raise ValueError(f'{mode}/{few_shot_protocol} needs a {expected} checkpoint')
    if expected == 'pretrain' and holdout in metadata['training_domains']:
        raise ValueError('Source checkpoint contains target-domain training')
    if mode == 'few' and few_shot_protocol == 'per_query' and domain != holdout:
        raise ValueError('Per-query few-shot has no single adapted model for source testing; use domain few-shot')


class HeadArtifact:
    def __init__(self, model, preprocessor, target, metadata):
        self.model, self.preprocessor, self.target, self.metadata = model,preprocessor,target,metadata

    @classmethod
    def load(cls, directory):
        directory=Path(directory)
        metadata=read_json(directory/'model_config.json')
        model=QueryConditionedTaskHead(**metadata['architecture'])
        model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=True))
        preprocess=joblib.load(directory/'feature_preprocessor.joblib')
        target=joblib.load(directory/'target_transformer.joblib') if metadata['task_type']=='Regression' else None
        return cls(model,preprocess,target,metadata)

    def save(self, directory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        torch.save({k:v.detach().cpu() for k,v in self.model.state_dict().items()},directory/'model.pt')
        joblib.dump(self.preprocessor,directory/'feature_preprocessor.joblib')
        if self.target is not None:
            joblib.dump(self.target,directory/'target_transformer.joblib')
        write_json(directory/'model_config.json',self.metadata)

    def clone(self):
        return HeadArtifact(copy.deepcopy(self.model),self.preprocessor,self.target,copy.deepcopy(self.metadata))

    def check_batch(self,batch):
        manifest=batch['manifest']
        if cache_spec_provenance(manifest)!=self.metadata.get('task_spec_provenance',{'source':'gold'}):
            raise ValueError('Task Spec source/model/fold differs from the checkpoint')
        if manifest['representation_fingerprint'] != self.metadata['representation_fingerprint']:
            raise ValueError('Encoder/representation differs from the checkpoint')
        if manifest['aggregation_config'] != self.metadata['aggregation_config']:
            raise ValueError('Aggregation policy differs from the checkpoint')
        if bool(manifest['smoke_only']) != bool(self.metadata['smoke_only']):
            raise ValueError('Cannot mix smoke and research artifacts')

    def predict(self,batch, batch_size=256):
        self.check_batch(batch)
        if not len(batch['X_rel']):return np.empty(0,np.float32)
        self.model.eval()
        device=next(self.model.parameters()).device
        values=self.preprocessor.transform(batch['X_rel'])
        output=[]
        with torch.inference_mode():
            for start in range(0,len(values),batch_size):
                rel=torch.from_numpy(values[start:start+batch_size]).to(device)
                query=torch.from_numpy(batch['X_query'][start:start+batch_size].astype(np.float32)).to(device)
                output.append(self.model(rel,query).float().cpu().numpy())
        raw=np.concatenate(output) if output else np.empty(0,np.float32)
        return self.target.inverse_transform(raw) if self.target else raw


def metrics_for(artifact,batch):
    pred=artifact.predict(batch)
    fn=classification_metrics if artifact.metadata['task_type']=='Classification' else regression_metrics
    kwargs={'threshold':artifact.metadata['classification_threshold']} if fn==classification_metrics else {}
    return fn(batch['y'],pred,**kwargs)


class PredictionHeadTrainer:
    def __init__(self,config):
        self.config=config
        self.cfg=config.get('head',{})
        self.seed=int(self.cfg.get('seed',42))
        self.device=torch.device(self.cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
        torch.set_num_threads(int(self.cfg.get('cpu_threads',2)))

    def _check_cache_source(self,batch,holdout):
        provenance=cache_spec_provenance(batch['manifest'])
        expected=self.config.get('task_spec_inputs',{}).get('source')
        if expected and provenance['source']!=expected:
            raise ValueError(f"Configuration expects {expected} Task Specs, cache contains {provenance['source']}")
        if provenance['source']=='generated' and provenance.get('holdout_domain')!=holdout:
            raise ValueError('Generated feature cache belongs to a different TSG holdout')

    def _optimize(self,artifact,train,validation=None,*,steps=None,history_path=None):
        artifact.check_batch(train)
        if not len(train['rows']):
            raise ValueError('Empty training set')
        if steps is not None and steps < 1:
            raise ValueError('Adaptation steps must be positive')
        set_seed(self.seed)
        model=artifact.model.to(self.device)
        rel=artifact.preprocessor.transform(train['X_rel'])
        target=artifact.target.transform(train['y']) if artifact.target else train['y']
        dataset=TensorDataset(torch.from_numpy(rel),torch.from_numpy(train['X_query'].astype(np.float32)),torch.from_numpy(target.astype(np.float32)))
        weights=balanced_weights(train['rows'])
        sampler=WeightedRandomSampler(torch.as_tensor(weights),num_samples=len(weights),replacement=True,
                                      generator=torch.Generator().manual_seed(self.seed))
        loader=DataLoader(dataset,batch_size=min(int(self.cfg.get('batch_size',64)),len(dataset)),sampler=sampler)
        learning_rate=float(self.cfg.get('few_shot_learning_rate',1e-4) if steps is not None or artifact.metadata.get('stage')=='few' else self.cfg.get('learning_rate',1e-3))
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=float(self.cfg.get('weight_decay',1e-4)))
        criterion=nn.BCEWithLogitsLoss() if artifact.target is None else nn.HuberLoss()
        best=copy.deepcopy(model.state_dict()); best_loss=float('inf'); stale=0; done=0; best_epoch=0
        max_epochs=int(self.cfg.get('epochs',100)) if steps is None else int(steps)
        if max_epochs<1:raise ValueError('epochs must be positive')
        history=[]
        for epoch in range(max_epochs):
            model.train()
            loss_sum=0.0; examples=0
            for x,q,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x.to(self.device),q.to(self.device)),y.to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(self.cfg.get('gradient_clip',1)))
                optimizer.step(); done+=1
                loss_sum+=float(loss.detach().cpu())*len(y);examples+=len(y)
                if steps is not None and done>=steps:break
            if steps is not None:
                if done>=steps:break
                continue
            record=dict(epoch=epoch+1, task_type=artifact.metadata['task_type'],
                        train_loss=loss_sum/max(examples,1), gradient_steps=done)
            if validation is not None and len(validation['rows']):
                artifact.check_batch(validation)
                model.eval()
                vr=torch.from_numpy(artifact.preprocessor.transform(validation['X_rel'])).to(self.device)
                vq=torch.from_numpy(validation['X_query'].astype(np.float32)).to(self.device)
                vy=artifact.target.transform(validation['y']) if artifact.target else validation['y']
                with torch.inference_mode():
                    logits=model(vr,vq)
                    if artifact.target is None:
                        loss_values=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(vy).to(self.device),reduction='none')
                    else:
                        loss_values=nn.functional.huber_loss(logits,torch.from_numpy(vy).to(self.device),reduction='none')
                    w=torch.from_numpy(balanced_weights(validation['rows'])).to(self.device)
                    val_loss=float((loss_values*w).sum()/w.sum())
                if not np.isfinite(val_loss):
                    raise ValueError('Nonfinite validation loss')
                raw=logits.detach().float().cpu().numpy()
                prediction=artifact.target.inverse_transform(raw) if artifact.target else raw
                metric_fn=regression_metrics if artifact.target else classification_metrics
                metric_kwargs={} if artifact.target else {'threshold':artifact.metadata['classification_threshold']}
                record.update(validation_loss=val_loss, validation=metric_fn(validation['y'],prediction,**metric_kwargs))
                record['validation_by_domain']={}
                for domain in sorted({r['domain'] for r in validation['rows']}):
                    mask=np.asarray([r['domain']==domain for r in validation['rows']])
                    record['validation_by_domain'][domain]=metric_fn(validation['y'][mask],prediction[mask],**metric_kwargs)
                if val_loss<best_loss-1e-8:
                    best_loss=val_loss; best=copy.deepcopy(model.state_dict()); best_epoch=epoch+1;stale=0
                else:stale+=1
            else:
                best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            record['best_epoch']=best_epoch
            history.append(record)
            if history_path is not None:write_json(history_path,history)
            print(json.dumps({k:v for k,v in record.items() if k!='validation_by_domain'}),flush=True)
            if validation is not None and len(validation['rows']) and stale>=int(self.cfg.get('patience',15)):
                break
        if steps is None:model.load_state_dict(best)
        artifact.metadata.update(best_epoch=best_epoch,gradient_steps=done,
            best_validation_loss=best_loss if np.isfinite(best_loss) else None,
            checkpoint_selection='minimum domain/parent-balanced validation loss' if validation is not None and len(validation['rows']) else 'fixed adaptation steps')
        return artifact

    def train_fold(self,cache_root,domains,holdout,output_root,*,stage='pretrain',source_checkpoint=None,
                   allow_smoke=False,k=None,selection_manifest=None):
        holdout=holdout.lower(); domains=[d.lower() for d in domains]
        if holdout not in domains or len(set(domains))!=len(domains):
            raise ValueError('Invalid holdout/domain list')
        if stage not in {'pretrain','full','few'}:raise ValueError(stage)
        if stage=='pretrain' and source_checkpoint:raise ValueError('Pretraining must start a new shared head')
        if stage!='pretrain' and not source_checkpoint:raise ValueError('Target adaptation requires the source pretrained checkpoint')
        selected=read_few_shot_selection(selection_manifest,holdout,k) if stage=='few' else None
        training_domains=[d for d in domains if d!=holdout] if stage=='pretrain' else [holdout]
        result={}
        for task in ('Classification','Regression'):
            train=concat_batches([load_cached_role(cache_root,d,'support/train',task) for d in training_domains])
            validation=concat_batches([load_cached_role(cache_root,d,'support/validation',task) for d in training_domains])
            if selected is not None:
                train=subset_batch(train,[i for i,r in enumerate(train['rows']) if r['uid'] in set(selected)])
            self._check_cache_source(train,holdout);self._check_cache_source(validation,holdout)
            check_training_splits(train,validation,training_domains)
            if not len(train['rows']) and stage=='pretrain':
                result[task]={'status':'no_prediction_training_examples'};continue
            if (train['manifest']['smoke_only'] or validation['manifest']['smoke_only']) and not allow_smoke:
                raise ValueError('Smoke caches are not valid research runs; explicitly pass --smoke')
            if len(train['rows']) and not len(validation['rows']):
                raise ValueError(f'{task} has training data but no validation data for checkpoint selection')
            set_seed(self.seed)
            if stage=='pretrain':
                arch=dict(relational_input_dim=train['X_rel'].shape[1],question_input_dim=train['X_query'].shape[1],
                    hidden_dim=int(self.cfg.get('hidden_dim',256)),question_dim=int(self.cfg.get('question_dim',128)),dropout=float(self.cfg.get('dropout',0.2)))
                preprocess=FeaturePreprocessor(mode=self.config.get('normalization',{}).get('relational_features','none')).fit(train['X_rel'])
                target=RegressionTargetTransformer(mode='signed_log1p').fit(train['y']) if task=='Regression' else None
                metadata=dict(architecture=arch,task_type=task,classification_threshold=0.5,
                    representation_fingerprint=train['manifest']['representation_fingerprint'],
                    encoder_config=train['manifest']['encoder_config'],representation_config=train['manifest']['representation_config'],
                    aggregation_config=train['manifest']['aggregation_config'],smoke_only=train['manifest']['smoke_only'],
                    task_spec_provenance=cache_spec_provenance(train['manifest']))
                if 'node_encoder_config' in train['manifest']:
                    metadata.update(node_encoder_config=train['manifest']['node_encoder_config'],
                                    node_encoder_fingerprint=train['manifest']['node_encoder_fingerprint'])
                artifact=HeadArtifact(QueryConditionedTaskHead(**arch),preprocess,target,metadata)
            else:
                artifact=HeadArtifact.load(Path(source_checkpoint)/task)
                if artifact.metadata['holdout_domain']!=holdout or artifact.metadata['stage']!='pretrain':
                    raise ValueError('Need a source pretrained checkpoint for the same held-out domain')
                if holdout in artifact.metadata['training_domains']:
                    raise ValueError('Held-out domain appeared in source training')
                artifact.check_batch(train)
                artifact.check_batch(validation)
            previous_domains=artifact.metadata.get('training_domains',[])
            previous_parents=artifact.metadata.get('seen_training_parent_uids',artifact.metadata.get('train_parent_uids',[]))
            previous_validation=artifact.metadata.get('seen_validation_parent_uids',artifact.metadata.get('validation_parent_uids',[]))
            effective_domains=sorted({r['domain'] for r in train['rows']})
            artifact.metadata.update(stage=stage,holdout_domain=holdout,
                training_domains=sorted(set(previous_domains+effective_domains)),
                current_stage_domains=effective_domains, planned_training_domains=training_domains,
                source_checkpoint=str(source_checkpoint) if source_checkpoint else None,
                train_sample_ids=[r['sample_id'] for r in train['rows']],
                validation_sample_ids=[r['sample_id'] for r in validation['rows']],
                train_parent_uids=sorted({r['uid'] for r in train['rows']}),
                validation_parent_uids=sorted({r['uid'] for r in validation['rows']}),
                seen_training_parent_uids=sorted(set(previous_parents)|{r['uid'] for r in train['rows']}),
                seen_validation_parent_uids=sorted(set(previous_validation)|{r['uid'] for r in validation['rows']}),
                fold_domains=domains,source_domains=[d for d in domains if d!=holdout],
                prediction_test_role='qa/test',
                few_shot_unit='target qa/train questions and their support/train' if stage=='few' else None,
                requested_k=k if stage=='few' else None,selected_train_qa_uids=selected,
                selection_manifest=str(selection_manifest) if selection_manifest else None,
                adaptation_status='trained' if len(train['rows']) else 'unchanged_no_target_training_support')
            history_path=Path(output_root)/task/'training_history.json'
            if len(train['rows']):
                self._optimize(artifact,train,validation,history_path=history_path)
            else:
                # A target may lack one task type (or all predictions, e.g. Hepatitis).
                # Retain its source head so the adapted system can still test source domains.
                artifact.metadata.update(best_epoch=0,gradient_steps=0,best_validation_loss=None,
                                         checkpoint_selection='unchanged source head; no target training support')
                write_json(history_path,[])
            artifact.metadata['validation']=metrics_for(artifact,validation)
            artifact.save(Path(output_root)/task)
            result[task]={'train_n':len(train['rows']),'validation_n':len(validation['rows']),
                          'train_parent_n':len({r['uid'] for r in train['rows']}),
                          'best_epoch':artifact.metadata['best_epoch'],'status':artifact.metadata['adaptation_status'],
                          'validation':artifact.metadata['validation']}
        write_json(Path(output_root)/'training_summary.json',dict(stage=stage,holdout=holdout,
            requested_k=k if stage=='few' else None,selected_train_qa_uids=selected,
            source_checkpoint=str(source_checkpoint) if source_checkpoint else None,heads=result))
        return result

    def adapt_fold_samples(self,cache_root,domains,holdout,output_root,*,source_checkpoint,k,allow_smoke=False):
        """Adapt once per target head on exactly k support/train rows, without validation labels."""
        holdout=holdout.lower();domains=[d.lower() for d in domains]
        if holdout not in domains or len(set(domains))!=len(domains):raise ValueError('Invalid fold domains')
        if not isinstance(k,int) or k<1:raise ValueError('few-shot k must be a positive sample count')
        steps=int(self.cfg.get('few_shot_steps',20))
        if steps<1:raise ValueError('Adaptation steps must be positive')
        if Path(source_checkpoint).resolve()==Path(output_root).resolve():raise ValueError('Adaptation must not overwrite source checkpoint')
        result={}
        for task in ('Classification','Regression'):
            batch=load_cached_role(cache_root,holdout,'support/train',task)
            self._check_cache_source(batch,holdout)
            if any(r['domain']!=holdout or r['role']!='support/train' for r in batch['rows']):
                raise ValueError('Domain few-shot must use target support/train only')
            if batch['manifest']['smoke_only'] and not allow_smoke:raise ValueError('Smoke cache blocked')
            if 0<len(batch['rows'])<k:raise ValueError(f'{task}: fewer than {k} target training samples')
            indices=sorted(range(len(batch['rows'])),key=lambda i:(stable_int(f"{self.seed}:{batch['rows'][i]['sample_id']}"),batch['rows'][i]['sample_id']))[:k]
            train=subset_batch(batch,indices)
            set_seed(self.seed)
            artifact=HeadArtifact.load(Path(source_checkpoint)/task)
            if artifact.metadata['stage']!='pretrain' or artifact.metadata['holdout_domain']!=holdout or holdout in artifact.metadata['training_domains']:
                raise ValueError('Need a source pretrained checkpoint that excludes the target domain')
            artifact.check_batch(train)
            previous=artifact.metadata
            parents=sorted({r['uid'] for r in train['rows']})
            ids=[r['sample_id'] for r in train['rows']]
            current_domains=[holdout] if ids else []
            artifact.metadata.update(stage='few',holdout_domain=holdout,
                training_domains=sorted(set(previous['training_domains']+current_domains)),
                current_stage_domains=current_domains,planned_training_domains=[holdout],
                source_checkpoint=str(source_checkpoint),train_sample_ids=ids,validation_sample_ids=[],
                train_parent_uids=parents,validation_parent_uids=[],
                seen_training_parent_uids=sorted(set(previous.get('seen_training_parent_uids',[]))|set(parents)),
                seen_validation_parent_uids=previous.get('seen_validation_parent_uids',[]),
                fold_domains=domains,source_domains=[d for d in domains if d!=holdout],prediction_test_role='qa/test',
                few_shot_unit='support/train samples per head',requested_k=k,actual_training_samples=len(ids),
                selected_train_qa_uids=parents,selection_manifest=None,selection_seed=self.seed,
                sampling_policy='label-independent seeded sample-ID ranking; nested increasing budgets',
                adaptation_status='trained' if ids else 'unchanged_no_target_training_support')
            if ids:
                self._optimize(artifact,train,steps=steps)
            else:
                artifact.metadata.update(best_epoch=0,gradient_steps=0,best_validation_loss=None,
                    checkpoint_selection='unchanged source head; no target training support')
            artifact.metadata['validation']={'status':'not_used_fixed_adaptation_steps'}
            artifact.save(Path(output_root)/task)
            write_json(Path(output_root)/task/'training_history.json',[])
            result[task]=dict(train_n=len(ids),validation_n=0,train_parent_n=len(parents),best_epoch=0,
                gradient_steps=artifact.metadata['gradient_steps'],status=artifact.metadata['adaptation_status'],
                validation=artifact.metadata['validation'],train_sample_ids=ids,train_parent_uids=parents)
        write_json(Path(output_root)/'training_summary.json',dict(stage='few',holdout=holdout,requested_k=k,
            few_shot_unit='support/train samples per head',training_role='support/train',validation_role=None,
            checkpoint_selection='fixed adaptation steps; no target validation labels used',
            adaptation_steps=steps,seed=self.seed,source_checkpoint=str(source_checkpoint),heads=result))
        return result

    def evaluate_fold(self,cache_root,holdout,checkpoint,output_dir,*,mode='zero',k=5,allow_smoke=False,
                      evaluation_domain=None,few_shot_protocol='per_query'):
        if mode not in {'zero','full','few'}:raise ValueError(mode)
        if few_shot_protocol not in {'per_query','domain'}:raise ValueError(few_shot_protocol)
        holdout=holdout.lower(); output_dir=Path(output_dir)
        domain=(evaluation_domain or holdout).lower()
        episodic=mode=='few' and few_shot_protocol=='per_query'
        if episodic and domain!=holdout:
            raise ValueError('Per-query few-shot cannot produce a single source-domain retention result; use domain few-shot')
        training_summary_path=Path(checkpoint)/'training_summary.json'
        training_summary=read_json(training_summary_path) if training_summary_path.exists() else {}
        if mode=='few' and not episodic and training_summary.get('requested_k')!=k:
            raise ValueError('Domain few-shot checkpoint budget differs from requested k')
        results={}; predictions=[];coverage={};failures=[];evaluation_provenance=None
        for task in ('Classification','Regression'):
            query=load_cached_role(cache_root,domain,'qa/test',task)
            self._check_cache_source(query,holdout)
            evaluation_provenance=cache_spec_provenance(query['manifest'])
            counts=query['manifest'].get('role_coverage',{}).get('qa/test',{}).get(task,
                dict(total=len(query['rows']),successful=len(query['rows']),failed=0))
            coverage[task]={**counts,'fraction':counts['successful']/counts['total'] if counts['total'] else None}
            if evaluation_provenance['source']=='generated':
                failed=read_json(Path(cache_root)/domain/'qa/test/failed_samples.json')
                failures.extend(r for r in failed if r['task_type']==task)
            if not len(query['rows']):
                results[task]={'n':0,'status':'all_generated_inputs_failed' if counts['total'] else 'domain_has_no_prediction_test'};continue
            if query['manifest']['smoke_only'] and not allow_smoke:raise ValueError('Smoke cache blocked')
            artifact=HeadArtifact.load(Path(checkpoint)/task)
            artifact.model.to(self.device)
            check_head_provenance(artifact.metadata,holdout,domain,mode,few_shot_protocol)
            seen=set(artifact.metadata.get('seen_training_parent_uids',[]))|set(artifact.metadata.get('seen_validation_parent_uids',[]))
            if seen & {r['uid'] for r in query['rows']}:
                raise ValueError('Test QA overlaps checkpoint training/validation parents')
            artifact.check_batch(query)
            if episodic:
                # Only this explicit branch reads support/test. It is inaccessible to zero/full evaluation.
                support=load_cached_role(cache_root,domain,'support/test',task)
                if support['manifest'].get('visibility_assumptions',{}).get('support_test_usage') == 'evaluation_only':
                    raise ValueError('This support/test cache is evaluation-only; its labels cannot adapt the model')
                artifact.check_batch(support)
            scores=[]
            for i,row in enumerate(query['rows']):
                one=subset_batch(query,[i])
                used=list(artifact.metadata.get('train_sample_ids',[])) if mode=='few' and not episodic else []
                current=artifact
                if episodic:
                    indices=choose_support_indices(support['rows'],row['uid'],k,self.seed)
                    used=[support['rows'][j]['sample_id'] for j in indices]
                    if not indices and not allow_smoke:
                        raise ValueError(f"No corresponding test support: {row['uid']}")
                    current=artifact.clone()  # parameters and optimizer reset for EVERY query
                    if indices:
                        train=subset_batch(support,indices)
                        if any(r['uid']!=row['uid'] for r in train['rows']):raise ValueError('Wrong parent in episode')
                        self._optimize(current,train,steps=int(self.cfg.get('few_shot_steps',20)))
                value=float(current.predict(one)[0]);scores.append(value)
                predictions.append(dict(domain=domain,parent_question_id=row['parent_question_id'],
                    task_type=task,mode=mode,few_shot_protocol=few_shot_protocol if mode=='few' else None,
                    requested_k=k if mode=='few' else 0,actual_k=len(used),
                    support_role=('support/test' if episodic else 'support/train') if mode=='few' else None,
                    support_sample_ids=used,gold=row['label'],
                    probability=float(1/(1+np.exp(-np.clip(value,-40,40)))) if task=='Classification' else None,
                    prediction=bool(1/(1+np.exp(-np.clip(value,-40,40)))>=artifact.metadata['classification_threshold']) if task=='Classification' else value))
            values=np.asarray(scores)
            results[task]=classification_metrics(query['y'],values,artifact.metadata['classification_threshold']) if task=='Classification' else regression_metrics(query['y'],values)
            # Always expose the trivial zero baseline for comparison.
            results[task]['zero_baseline']=(classification_metrics(query['y'],np.full(len(values),-40.0))
                                           if task=='Classification' else regression_metrics(query['y'],np.zeros(len(values))))
        report=dict(holdout_domain=holdout,domain=domain,mode=mode,
            evaluation_role='target_test' if domain==holdout else 'source_test',
            checkpoint=str(checkpoint),requested_k=k if episodic else training_summary.get('requested_k',0),
            few_shot_protocol=few_shot_protocol if mode=='few' else None,
            protocol='original qa/test; separate reset and adaptation for each target query' if episodic else 'original qa/test; one frozen checkpoint for every evaluated domain; no test-time training',
            smoke_only=allow_smoke,metrics=results,task_spec_provenance=evaluation_provenance,
            input_coverage=coverage,
            metrics_scope='successful feature builds only; failures and total counts reported in input_coverage',
            failure_count=len(failures),end_to_end=False)
        if failures:write_json(output_dir/'failed_samples.json',failures)
        if mode=='few' and not episodic:
            report['domain_adaptation']=dict(unit=training_summary.get('few_shot_unit'),
                training_role='support/train',checkpoint_selection=training_summary.get('checkpoint_selection'),
                training_samples_by_head={t:h['train_n'] for t,h in training_summary['heads'].items()},
                selected_sample_ids_by_head={t:h.get('train_sample_ids',[]) for t,h in training_summary['heads'].items()})
        write_json(output_dir/'metrics.json',report);write_json(output_dir/'predictions.json',predictions)
        return report

    def evaluate_scopes(self,cache_root,domains,holdout,checkpoint,output_dir,*,scope='target',
                        mode='zero',k=5,allow_smoke=False,few_shot_protocol='per_query'):
        if scope not in {'source','target','both'}:raise ValueError(scope)
        holdout=holdout.lower();domains=[d.lower() for d in domains];output_dir=Path(output_dir)
        if holdout not in domains or len(set(domains))!=len(domains):raise ValueError('Invalid fold domains')
        if scope!='target' and len(domains)<2:raise ValueError('No source domains configured')
        if scope!='target' and mode=='few' and few_shot_protocol=='per_query':
            raise ValueError('Source testing after few-shot requires one domain-adapted checkpoint: --few-shot-protocol domain')
        kwargs=dict(mode=mode,k=k,allow_smoke=allow_smoke,few_shot_protocol=few_shot_protocol)
        if scope=='target':
            return self.evaluate_fold(cache_root,holdout,checkpoint,output_dir,**kwargs)
        reports={};predictions=[]
        for domain in domains:
            if domain==holdout:continue
            folder=output_dir/'source_test'/domain
            reports[domain]=self.evaluate_fold(cache_root,holdout,checkpoint,folder,evaluation_domain=domain,**kwargs)
            predictions.extend(read_json(folder/'predictions.json'))
        source_report=dict(holdout_domain=holdout,scope='source',mode=mode,checkpoint=str(checkpoint),
            test_role='qa/test',num_examples=len(predictions),metrics_by_domain={d:r['metrics'] for d,r in reports.items()},
            domain_macro=domain_macro_metrics({d:r['metrics'] for d,r in reports.items()}),
            aggregation_note='Equal domain weights; undefined metrics excluded with denominators. Raw regression MAE/RMSE reported by domain only.',
            input_coverage_by_domain={d:r['input_coverage'] for d,r in reports.items()},
            failure_count=sum(r['failure_count'] for r in reports.values()),
            metrics_scope='successful feature builds only; see per-domain coverage; not full end-to-end evaluation')
        write_json(output_dir/'source_test'/'metrics.json',source_report)
        write_json(output_dir/'source_test'/'predictions.json',predictions)
        summary=dict(holdout_domain=holdout,mode=mode,scope=scope,checkpoint=str(checkpoint),source_test=source_report)
        if scope=='both':
            summary['target_test']=self.evaluate_fold(cache_root,holdout,checkpoint,output_dir/'target_test'/holdout,**kwargs)
        write_json(output_dir/'evaluation_summary.json',summary)
        return summary
