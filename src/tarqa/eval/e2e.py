from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tarqa.answer.verbalizer import TemplateVerbalizer
from tarqa.data.discovery import DomainPaths
from tarqa.data.metadata import SchemaCatalog
from tarqa.prediction.factory import build_prediction_runtime
from tarqa.prediction.pipeline import PredictionPipeline
from tarqa.prediction.question_encoder import FrozenQuestionEncoder
from tarqa.reasoning.pipeline import ReasoningPipeline
from tarqa.task_spec.inference import LocalTaskSpecGenerator
from tarqa.task_spec.prompts import SYSTEM_PROMPT, render_user_prompt
from tarqa.task_spec.serialization import serialize_schema
from tarqa.types import CanonicalAnswer, EntityAnchor, TaskSpec, Track


@dataclass
class EndToEndResult:
    task_spec: TaskSpec
    canonical_answer: CanonicalAnswer
    natural_language_answer: str
    details: dict[str, Any]


class EndToEndPipeline:
    def __init__(
        self,
        paths: DomainPaths,
        task_spec_generator: LocalTaskSpecGenerator,
        reasoning_model: Any,
        reasoning_cfg: dict[str, Any],
        prediction_cfg: dict[str, Any],
        prediction_artifact_root: str | Path,
        symbolic_time_adapter_path: str | Path | None,
        *,
        holdout_domain: str | None = None,
        question_encoder: FrozenQuestionEncoder | None = None,
        verbalizer: Any | None = None,
    ) -> None:
        self.paths = paths
        self.holdout_domain = holdout_domain or paths.domain
        self.question_encoder = question_encoder
        self.catalog = SchemaCatalog.from_files(paths.domain, paths.metadata, paths.profiling, database_path=paths.database)
        self.task_spec_generator = task_spec_generator
        self.schema_text = serialize_schema(self.catalog)
        if paths.domain == "ncaa":
            self.schema_text += "\nTime convention: an observation day denotes the NEXT day's midnight cutoff; game facts become available at next-day midnight."
        self.reasoning = ReasoningPipeline(
            self.catalog,
            paths.database,
            reasoning_model,
            add_bridge_tables=bool(reasoning_cfg.get("add_bridge_tables", True)),
            max_repair_attempts=int(reasoning_cfg.get("max_repair_attempts", 2)),
            timeout_seconds=float(reasoning_cfg.get("timeout_seconds", 120)),
            schema_policy=reasoning_cfg.get("schema_policy", "full"),
        )
        # Lazy prediction runtime: Hepatitis/Reasoning do not load a predictor or encoder.
        self.prediction = None
        self.prediction_cfg = prediction_cfg
        self.prediction_artifact_root = prediction_artifact_root
        self.verbalizer = verbalizer if verbalizer is not None else TemplateVerbalizer()

    def answer(self, question: str, *, anchor: EntityAnchor | None = None, reference_time: str | None = None,
               support_rows=None, few_shot_k=0, parent_uid=None) -> EndToEndResult:
        self.last_task_spec = None
        user_prompt = render_user_prompt(self.schema_text, question, generation_strategy="track_last")
        generated = self.task_spec_generator.generate(SYSTEM_PROMPT, user_prompt)
        if generated.dependency_errors:
            raise ValueError(f"Invalid generated Task Spec: {generated.dependency_errors}")
        spec = generated.spec
        self.last_task_spec = spec
        if spec.track == Track.PREDICTION and reference_time is not None:
            if spec.reference_time != TaskSpec.canonical_time(reference_time):
                raise ValueError('Generated reference_time differs from the audited query cutoff; refusing a different history window')
        if spec.track == Track.REASONING:
            result = self.reasoning.answer(question, spec)
            canonical = result.answer
            details = {"sql": result.sql, "attempts": result.attempts, "execution_rows": result.rows}
        else:
            if anchor is None:
                raise ValueError(
                    "The final Predictive benchmark requires one explicit Anchor Entity. "
                    "Pass the audited target_instances anchor to EndToEndPipeline.answer()."
                )
            if self.prediction is None:
                encoder = self.question_encoder or FrozenQuestionEncoder(self.prediction_cfg["question_encoder"])
                self.question_encoder = encoder
                _, aggregator = build_prediction_runtime(self.paths, self.prediction_cfg, encoder, symbolic_time_adapter_path=None)
                self.prediction = PredictionPipeline(self.catalog, aggregator, encoder, self.prediction_artifact_root,
                                                     holdout_domain=self.holdout_domain,task_spec_generator=self.task_spec_generator)
            canonical = self.prediction.answer(question, spec, anchor, support_rows=support_rows, k=few_shot_k, parent_uid=parent_uid, trainer_config=self.prediction_cfg)
            details = {"anchor": anchor.model_dump(mode="json")}
        if hasattr(self.verbalizer, "verbalize_with_spec"):
            natural = self.verbalizer.verbalize_with_spec(question, canonical, spec)
        else:
            natural = self.verbalizer.verbalize(question, canonical)
        return EndToEndResult(spec, canonical, natural, details)

    def close(self) -> None:
        self.reasoning.close()
        if self.prediction is not None:
            self.prediction.aggregator.close()
