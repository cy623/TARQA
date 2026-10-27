"""Verbalize canonical answers with the already-loaded TSG language model.

This module neither loads another model nor updates/disables the TSG adapter.
It uses free-text decoding, separately from the TSG's four-field JSON parser.
"""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy

import torch

from tarqa.task_spec.sft import render_chat_prompt
from tarqa.types import CanonicalAnswer, TaskSpec


class SharedTSGVerbalizer:
    def __init__(self, task_spec_generator, *, max_new_tokens=160, max_attempts=2, streamer=None):
        self.generator = task_spec_generator
        self.model = task_spec_generator.model
        self.tokenizer = task_spec_generator.tokenizer
        self.max_input_length = task_spec_generator.max_input_length
        self.max_new_tokens = int(max_new_tokens)
        self.max_attempts = int(max_attempts)
        self.streamer = streamer
        self.last_trace = None
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    @staticmethod
    def payload(question: str, answer: CanonicalAnswer, spec: TaskSpec | None = None):
        # Deliberately do not forward arbitrary metadata, labels, or gold answers.
        result = dict(question=question, answer_type=answer.answer_type,
                      predicted_value=answer.value, unit=answer.unit)
        if spec is not None:
            result["task_spec"] = spec.model_dump(mode="json")
        result["is_prediction"] = (spec is not None and spec.track.value == "Prediction") or answer.probability is not None
        if answer.probability is not None:
            probability = float(answer.probability)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid prediction probability")
            threshold = float(answer.metadata.get("classification_threshold", .5))
            if bool(answer.value) != (probability >= threshold):
                raise ValueError("Canonical decision disagrees with its probability/threshold")
            result.update(positive_event_probability=probability,
                          positive_event_probability_display=f"{probability * 100:.2f}%",
                          classification_threshold=threshold,
                          expected_prefix="Yes" if bool(answer.value) else "No")
        elif answer.answer_type == "Scalar":
            value = answer.value
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Cannot verbalize a non-finite scalar")
            result["value_display"] = str(value)
        elif answer.answer_type == "Boolean":
            result["expected_prefix"] = "Yes" if bool(answer.value) else "No"
        return result

    @staticmethod
    def validate_text(text: str, payload: dict):
        errors = []
        if not text or text.lstrip().startswith(("{", "[")):
            errors.append("Return natural-language text, not Task Spec JSON")
        prefix = payload.get("expected_prefix")
        if prefix and not re.match(r"^\s*" + prefix + r"\b", text, flags=re.I):
            errors.append(f"Start with {prefix}, matching the supplied decision")
        probability = payload.get("positive_event_probability_display")
        if probability and probability not in text:
            errors.append(f"Report the positive-event probability exactly as {probability}")
        if payload["is_prediction"] and not re.search(r"predict|estimat|forecast", text, flags=re.I):
            errors.append("Explicitly describe the answer as a model prediction or estimate")
        if "value_display" in payload and payload["value_display"] not in text:
            errors.append(f"Preserve the supplied value exactly: {payload['value_display']}")
        # Percentages may come from the question or from the supplied probability,
        # but not from a newly invented or complemented confidence score.
        allowed = set(re.findall(r"\d+(?:\.\d+)?%", payload["question"]))
        if probability:
            allowed.add(probability)
        unexpected = set(re.findall(r"\d+(?:\.\d+)?%", text)) - allowed
        if unexpected:
            errors.append("Do not invent percentages: " + ", ".join(sorted(unexpected)))
        return errors

    def verbalize(self, question: str, answer: CanonicalAnswer):
        return self._verbalize(question, answer, None)

    def verbalize_with_spec(self, question: str, answer: CanonicalAnswer, spec: TaskSpec):
        return self._verbalize(question, answer, spec)

    @torch.inference_mode()
    def _verbalize(self, question, answer, spec):
        payload = self.payload(question, answer, spec)
        system = (
            "Your task is ANSWER VERBALIZATION, not Task Spec generation. "
            "Write one concise English natural-language answer to the supplied question "
            "using ONLY the supplied canonical model result. Do not output JSON or SQL. "
            "Do not calculate a different answer or invent evidence, historical facts, causes, "
            "values, or percentages. For a prediction, explicitly say 'the model predicts' "
            "or 'the estimated probability'; do not present a forecast as an observed fact. "
            "For a Boolean, begin with the exact expected_prefix. If a probability is supplied, "
            "quote positive_event_probability_display exactly as the probability that the "
            "question's POSITIVE event occurs, even when your answer begins No. "
            "Do not replace it with the complementary probability. "
            "For a scalar, preserve value_display exactly."
        )
        examples = (
            "Answer-writing examples (unrelated fictional examples):\n"
            "Question: Will customer X return next month? Decision: Yes. "
            "Positive-event probability: 72.30%.\n"
            "Answer: Yes, the model predicts that customer X will return next month "
            "(positive-event probability: 72.30%).\n"
            "Question: Will parcel Z arrive tomorrow? Decision: No. "
            "Positive-event probability: 12.30%.\n"
            "Answer: No, the model predicts that parcel Z is unlikely to arrive tomorrow "
            "(positive-event probability: 12.30%).\n\n"
        )
        requirements = ["Write a natural-language answer, not JSON or SQL."]
        if payload.get("expected_prefix"):
            requirements.append("Begin with " + payload["expected_prefix"] + ".")
        if payload["is_prediction"]:
            requirements.append("Explicitly say 'the model predicts' or 'the model estimates'.")
        if payload.get("positive_event_probability_display"):
            requirements.append("Include exactly 'positive-event probability: "
                                + payload["positive_event_probability_display"] + "'.")
        if payload.get("value_display"):
            requirements.append("Preserve the value exactly: " + payload["value_display"] + ".")
        base_user = (examples + "Now answer this actual question using its supplied result:\n"
                     + json.dumps(payload, ensure_ascii=False)
                     + "\nAll required constraints:\n- " + "\n- ".join(requirements))
        user = base_user
        attempts = []
        self.last_trace = dict(payload=payload, system_prompt=system, attempts=attempts,
                               adapter_disabled=False, model_reused=True,
                               verbalization_demonstrations=2,
                               decoding=dict(do_sample=False, num_beams=1,
                                             max_new_tokens=self.max_new_tokens))
        for attempt in range(self.max_attempts):
            prompt = render_chat_prompt(self.tokenizer, system, user)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                    truncation=False)
            if inputs["input_ids"].shape[1] > self.max_input_length:
                raise ValueError("Verbalization prompt exceeds the model context limit")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generation = deepcopy(self.model.generation_config)
            generation.do_sample = False
            generation.temperature = None
            generation.top_p = None
            generation.top_k = None
            generated = self.model.generate(**inputs, generation_config=generation, do_sample=False, num_beams=1,
                max_new_tokens=self.max_new_tokens, use_cache=True,
                streamer=self.streamer,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            tokens = generated[0, inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            errors = self.validate_text(raw, payload)
            attempts.append(dict(user_prompt=user, raw_text=raw, errors=errors,
                                 prompt_tokens=int(inputs["input_ids"].shape[1]),
                                 generated_tokens=int(len(tokens))))
            if not errors:
                self.last_trace["status"] = "passed"
                return raw
            user = base_user + "\nAdditional corrections:\n- " + "\n- ".join(errors)
        self.last_trace["status"] = "failed"
        raise ValueError("TSG-model verbalization failed checks: " + "; ".join(errors))
