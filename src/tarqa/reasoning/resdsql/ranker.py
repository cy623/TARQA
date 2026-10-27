"""Checkpoint-compatible RESDSQL ranker, including complete-schema chunking."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from tarqa.reasoning.resdsql.classifier import MyClassifier
from tarqa.reasoning.resdsql.formatting import classifier_words, ranker_chunks


def device_from_config(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_ranker(path, device="auto"):
    path = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                             use_fast=True, add_prefix_space=True)
    if not tokenizer.is_fast:
        raise ValueError("The RESDSQL ranker needs a fast tokenizer for word alignment")
    model = MyClassifier(str(path), len(tokenizer), "eval")
    if (path / "model.safetensors").is_file():
        from safetensors.torch import load_file
        state = load_file(str(path / "model.safetensors"))
    elif (path / "dense_classifier.pt").is_file():
        state = torch.load(path / "dense_classifier.pt", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No RESDSQL classifier weights in {path}; use script 14")
    # Old Transformers saved these deterministic buffers; modern versions do not.
    # All learned parameters still load strictly.
    for key in ("plm_encoder.embeddings.position_ids", "plm_encoder.embeddings.token_type_ids"):
        if key in state and key not in model.state_dict():
            value = state.pop(key)
            expected = torch.arange(value.shape[-1]).expand_as(value) if key.endswith("position_ids") else torch.zeros_like(value)
            if not torch.equal(value, expected):
                raise ValueError(f"Unexpected legacy position buffer in ranker: {key}")
    model.load_state_dict(state, strict=True)
    if len(tokenizer) > model.plm_encoder.get_input_embeddings().num_embeddings:
        raise ValueError("Ranker tokenizer and checkpoint vocabulary differ")
    return tokenizer, model.to(device_from_config(device)).eval()


def save_ranker(model, tokenizer, path):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(path / "model.safetensors"))
    model.plm_encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)


def encode_chunk(payload, selections, tokenizer, device, max_length=512):
    words, tids, cids = classifier_words(payload, selections)
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[-1] > max_length:
        raise ValueError("Internal ranker chunk exceeded its token budget")
    word_ids = encoded.word_ids(batch_index=0)
    def align(ids):
        groups = [[i for i, word in enumerate(word_ids) if word == idx] for idx in ids]
        if any(not group for group in groups):
            raise ValueError("Tokenizer dropped a schema item")
        return groups
    inputs = (encoded["input_ids"].to(device), encoded["attention_mask"].to(device),
              [align([0])[0]], [align(cids)], [align(tids)],
              [[len(cols) for _, cols in selections]])
    return inputs


def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = torch.as_tensor(labels, device=logits.device, dtype=torch.long)
    selected = logp.gather(1, targets[:, None]).squeeze(1)
    weights = torch.where(targets == 1, alpha, 1-alpha)
    return (-weights * (1-selected.exp()).pow(gamma) * selected).sum() / weights.sum().clamp_min(1e-8)


class SchemaRanker:
    def __init__(self, path, *, device="auto", max_length=512):
        self.tokenizer, self.model = load_ranker(path, device)
        self.device = next(self.model.parameters()).device
        self.max_length = int(max_length)
        if self.max_length > 512:
            raise ValueError("RoBERTa ranker supports at most 512 input tokens per chunk")

    def chunks(self, payload):
        return ranker_chunks(payload, self.tokenizer, self.max_length)

    def forward_chunk(self, payload, selections):
        return self.model(*encode_chunk(payload, selections, self.tokenizer,
                                         self.device, self.max_length))

    @torch.inference_mode()
    def score(self, payload):
        self.model.eval()
        table_scores = [[] for _ in payload["db_schema"]]
        columns = [[None] * len(t["column_names"]) for t in payload["db_schema"]]
        chunks = self.chunks(payload)
        for selection in chunks:
            outputs = self.forward_chunk(payload, selection)
            ts = torch.softmax(outputs["batch_table_name_cls_logits"][0].float(), -1)[:, 1].cpu().tolist()
            cs = torch.softmax(outputs["batch_column_info_cls_logits"][0].float(), -1)[:, 1].cpu().tolist()
            offset = 0
            for i, (tid, cids) in enumerate(selection):
                table_scores[tid].append(ts[i])
                for cid in cids:
                    columns[tid][cid] = cs[offset]
                    offset += 1
        if any(not s for s in table_scores) or any(v is None for row in columns for v in row):
            raise ValueError("Incomplete schema scoring")
        return dict(tables=[float(np.mean(s)) for s in table_scores], columns=columns,
                    chunks=len(chunks))
