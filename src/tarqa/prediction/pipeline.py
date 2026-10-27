from pathlib import Path
import numpy as np
from tarqa.answer.normalizer import normalize_prediction
from tarqa.prediction.trainer import HeadArtifact, PredictionHeadTrainer, choose_support_indices
from tarqa.prediction.representation import SchemaSemanticPool
from tarqa.prediction.question_encoder import render_question_representation_input
from tarqa.types import BindingResult,Track,TaskType


class PredictionPipeline:
    def __init__(self,catalog,aggregator,question_encoder,artifact_root,*,holdout_domain=None,task_spec_generator=None):
        self.catalog,self.aggregator,self.encoder=catalog,aggregator,question_encoder
        self.holdout_domain=holdout_domain or catalog.domain
        self.root=Path(artifact_root);self._heads={}
        self.task_spec_generator=task_spec_generator

    def answer(self,question,spec,anchor,*,support_rows=None,k=0,parent_uid=None,trainer_config=None):
        if spec.track!=Track.PREDICTION or spec.validate_dependencies():
            raise ValueError('Prediction requires a valid generated Task Spec')
        task=spec.task_type.value
        if task not in self._heads:
            artifact=HeadArtifact.load(self.root/task)
            provenance=artifact.metadata.get('task_spec_provenance',{'source':'gold'})
            if provenance['source']=='generated':
                from tarqa.prediction.generated_specs import tsg_provenance
                generator=self.task_spec_generator
                if generator is None or not hasattr(generator,'model_cfg'):
                    raise ValueError('Generated-feature heads require the actual runtime Task Spec generator')
                actual=tsg_provenance(generator.model_cfg,generator.adapter_path,self.holdout_domain)
                if actual!=provenance:raise ValueError('Runtime TSG differs from the head training feature generator')
            representation_config=artifact.metadata['representation_config']
            if representation_config.get('backend')=='hetero_gnn':
                from tarqa.prediction.gnn_representation import FrozenLMGraphPool
                if self.aggregator.config.get('backend')!='hetero_graph':
                    raise ValueError('Graph checkpoint requires the historical row graph backend')
                pool=FrozenLMGraphPool(self.encoder,representation_config,
                    node_encoder=getattr(self.aggregator,'node_encoder',self.encoder))
            else:
                pool=SchemaSemanticPool(self.encoder,representation_config)
            if pool.fingerprint!=artifact.metadata['representation_fingerprint']:
                raise ValueError('Inference encoder differs from the frozen training encoder')
            if self.aggregator.config!=artifact.metadata['aggregation_config']:
                raise ValueError('Inference aggregation differs from training')
            # In-domain heads store the frozen TSG fold in its provenance,
            # rather than the source-training-only holdout_domain field.
            checkpoint_holdout = artifact.metadata.get('holdout_domain') or provenance.get('holdout_domain')
            if checkpoint_holdout != self.holdout_domain:
                raise ValueError("Checkpoint belongs to a different held-out domain")
            self._heads[task]=(artifact,pool)
        artifact,pool=self._heads[task]
        features=self.aggregator.aggregate(spec,BindingResult(anchors=[anchor])).features
        batch={'X_rel':pool.transform(features)[None,:],
               'X_query':self.encoder.encode([render_question_representation_input(question,spec)]),
               'manifest':dict(representation_fingerprint=pool.fingerprint,
                   task_spec_provenance=artifact.metadata.get('task_spec_provenance',{'source':'gold'}),
                   aggregation_config=self.aggregator.config,smoke_only=self.encoder.is_smoke)}
        actual_k = 0
        if k:
            if self.catalog.domain != self.holdout_domain:
                raise ValueError('Source retention evaluation must not adapt on source test support')
            if not support_rows or not parent_uid:
                raise ValueError("Per-query adaptation requires this parent's audited support")
            if artifact.metadata["stage"] != "pretrain" or self.catalog.domain in artifact.metadata["training_domains"]:
                raise ValueError("Few-shot must start from the source pretrained checkpoint")
            config = trainer_config or {"head": {}}
            indices = choose_support_indices(support_rows,parent_uid,k,int(config.get("head",{}).get("seed",42)))
            chosen = [support_rows[i] for i in indices]
            if not chosen:
                raise ValueError("No corresponding support for this query")
            rels,queries = [],[]
            for row in chosen:
                if row["domain"] != self.catalog.domain or row["task_type"] != task:
                    raise ValueError("Generated task type or support domain does not match the episode")
                from tarqa.types import TaskSpec,EntityAnchor
                from tarqa.data.releases import stamp
                if stamp(row["reference_time"]) >= stamp(spec.reference_time):
                    raise ValueError("Support does not precede the generated query cutoff")
                support_spec=TaskSpec.model_validate(row["spec"])
                rels.append(pool.transform(self.aggregator.aggregate(support_spec,
                    BindingResult(anchors=[EntityAnchor.model_validate(row["anchor"])])).features))
                queries.append(render_question_representation_input(row["question"],support_spec))
            training=dict(X_rel=np.stack(rels),X_query=self.encoder.encode(queries),
                y=np.asarray([r["label"] for r in chosen],np.float32),rows=chosen,manifest=batch["manifest"])
            artifact=artifact.clone()
            PredictionHeadTrainer(config)._optimize(artifact,training,
                steps=int(config.get("head",{}).get("few_shot_steps",20)))
            actual_k=len(chosen)
        raw=float(artifact.predict(batch)[0])
        if task==TaskType.CLASSIFICATION.value:
            prob=float(1/(1+np.exp(-np.clip(raw,-40,40))))
            result=normalize_prediction(value=prob>=artifact.metadata['classification_threshold'],task_type=task,probability=prob)
            result.metadata["classification_threshold"] = float(artifact.metadata['classification_threshold'])
            result.metadata["actual_k"] = actual_k
            return result
        result=normalize_prediction(value=raw,task_type=task)
        result.metadata["actual_k"] = actual_k
        return result
