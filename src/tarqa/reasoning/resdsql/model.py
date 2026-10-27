"""RESDSQL schema ranker + skeleton-aware T5 generator for TARQA."""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tarqa.reasoning.resdsql.content import DuckDBContents
from tarqa.reasoning.resdsql.formatting import public_payload, ranked_input, extract_generated_sql
from tarqa.reasoning.resdsql.ranker import SchemaRanker, device_from_config


def load_generator(path, device="auto", dtype="float32"):
    path = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    chosen = device_from_config(device)
    precision = getattr(torch, dtype) if dtype != "auto" else "auto"
    if chosen.type == "cpu":
        precision = torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, torch_dtype=precision)
    if not model.config.is_encoder_decoder:
        raise ValueError("RESDSQL requires an encoder-decoder SQL checkpoint")
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        raise ValueError("Generator tokenizer and model vocabulary differ")
    return tokenizer, model.to(chosen).eval()


def encode_texts(tokenizer, texts, max_length, *, kind="input"):
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    if encoded["input_ids"].shape[-1] > max_length:
        raise ValueError(f"RESDSQL {kind} exceeds {max_length} tokens; no silent truncation. "
                         "Increase the configured length or review this record.")
    return {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}


class RESDSQLModel:
    backend = "resdsql"
    supports_error_repair = False

    def __init__(self, config):
        self.config = dict(config)
        self.ranker = SchemaRanker(config["ranker_path"], device=config.get("device", "auto"),
                                   max_length=config.get("ranker_max_length", 512))
        self.tokenizer, self.model = load_generator(config["model_path"],
            config.get("device", "auto"), config.get("torch_dtype", "float32"))
        self.contents = {}
        self.last_details = {}
        self._validate_config()

    def _validate_config(self):
        beams = int(self.config.get("num_beams", 8))
        returns = int(self.config.get("num_return_sequences", beams))
        if not 1 <= returns <= beams:
            raise ValueError("Require 1 <= num_return_sequences <= num_beams")
        if self.config.get("sql_dialect", "sqlite") not in {"sqlite", "duckdb"}:
            raise ValueError("Supported RESDSQL SQL dialects: sqlite and duckdb")

    def public_input(self, question, catalog, database_path):
        contents = None
        if self.config.get("use_contents", True):
            key = (catalog.domain, str(Path(database_path).resolve()))
            if key not in self.contents:
                self.contents[key] = DuckDBContents(database_path, catalog,
                    max_distinct=self.config.get("content_max_distinct", 1000),
                    max_matches=self.config.get("content_max_matches", 2),
                    max_value_chars=self.config.get("content_max_value_chars", 80),
                    timeout_seconds=self.config.get("content_timeout_seconds", 120))
            contents = self.contents[key]
        return public_payload(question, catalog, contents)

    @torch.inference_mode()
    def generate_from_input(self, text):
        self.model.eval()
        device = next(self.model.parameters()).device
        encoded = encode_texts(self.tokenizer, [text], int(self.config.get("max_input_length", 512)))
        encoded = {k: v.to(device) for k, v in encoded.items()}
        beams = int(self.config.get("num_beams", 8))
        outputs = self.model.generate(**encoded,
            max_new_tokens=int(self.config.get("max_new_tokens", 512)),
            do_sample=False, num_beams=beams,
            num_return_sequences=int(self.config.get("num_return_sequences", beams)))
        sqls, errors = [], []
        for ids in outputs:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            try:
                eos = self.model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                if not any(e is not None and e in ids[1:].tolist() for e in eos_ids):
                    raise ValueError("Generated sequence reached token limit without EOS")
                sql = extract_generated_sql(text, self.config.get("sql_dialect", "sqlite"))
                if sql not in sqls:
                    sqls.append(sql)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self.last_details["invalid_candidates"] = errors
        if not sqls:
            raise ValueError(f"No valid SQL candidate from RESDSQL: {errors[:2]}")
        return sqls

    def generate_candidates(self, question, catalog, database_path):
        payload = self.public_input(question, catalog, database_path)
        scores = self.ranker.score(payload)
        text, selection = ranked_input(payload, scores, self.config.get("topk_tables", 4),
                                        self.config.get("topk_columns", 5))
        self.last_details = dict(ranker_chunks=scores["chunks"], selected_schema=selection,
                                  input_schema_policy="full_catalog_then_resdsql_ranker")
        return self.generate_from_input(text)

    def release_database(self, catalog, database_path):
        key = (catalog.domain, str(Path(database_path).resolve()))
        contents = self.contents.pop(key, None)
        if contents:
            contents.close()

    def close(self):
        for contents in self.contents.values():
            contents.close()
        self.contents.clear()
