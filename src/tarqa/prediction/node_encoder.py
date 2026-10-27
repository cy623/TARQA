"""Frozen Sentence-Transformer for questions, node attributes and FK descriptions."""
import hashlib
import json
from pathlib import Path
import numpy as np


class FrozenSentenceEncoder:
    def __init__(self, config):
        import torch
        from sentence_transformers import SentenceTransformer
        from tarqa.data.releases import file_hash
        self.config = dict(config)
        if config.get('backend', 'sentence_transformer') != 'sentence_transformer':
            raise ValueError('Sentence encoder backend must be sentence_transformer')
        if config.get('adapter_path'):
            raise ValueError('Sentence encoder must remain frozen; do not load TSG adapters')
        path = Path(config['model_path']).expanduser().resolve()
        if not (path / 'modules.json').is_file():
            raise FileNotFoundError(f'Need a local Sentence-Transformer snapshot including modules.json: {path}')
        self.config['model_path'] = str(path)
        self.model = SentenceTransformer(str(path), local_files_only=True,
            trust_remote_code=False, device=config.get('device', 'cpu')).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.is_smoke = False
        self.max_length = int(config.get('max_length', self.model.max_seq_length))
        if not 8 <= self.max_length <= self.model.max_seq_length:
            raise ValueError('Sentence encoder max_length must fit the pretrained sentence model')
        self.batch_size = int(config.get('batch_size', 64))
        if self.batch_size < 1:
            raise ValueError('Sentence encoder batch_size must be positive')
        self.device = self.model.device
        suffixes = {'.json', '.safetensors', '.bin', '.model', '.txt', '.vocab'}
        hashes = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*'))
                  if p.is_file() and p.suffix in suffixes and not any(x.startswith('.') for x in p.relative_to(path).parts)}
        # Runtime placement/batching do not define a different learned representation.
        payload = dict(version=1, files=hashes, backend='sentence_transformer',
                       max_length=self.max_length, long_text_strategy='token_chunks_weighted_mean',
                       dtype=str(next(self.model.parameters()).dtype))
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self._torch = torch

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokenizer = self.model.tokenizer
        token_ids = tokenizer(list(texts), add_special_tokens=False, truncation=False,
                              return_attention_mask=False, return_token_type_ids=False)['input_ids']
        capacity = self.max_length - tokenizer.num_special_tokens_to_add(pair=False)
        features, owners, weights = [], [], []
        for owner, ids in enumerate(token_ids):
            # Long rows retain all tokens; do not drop columns beyond MiniLM's short context.
            for start in range(0, max(len(ids), 1), capacity):
                part = ids[start:start+capacity]
                features.append(tokenizer.prepare_for_model(part, add_special_tokens=True,
                    truncation=False, return_attention_mask=True))
                owners.append(owner)
                weights.append(max(len(part), 1))
        total = np.zeros((len(texts), self.dimension), dtype=np.float32)
        with self._torch.inference_mode():
            for start in range(0, len(features), self.batch_size):
                batch = tokenizer.pad(features[start:start+self.batch_size], padding=True, return_tensors='pt')
                batch = {key: value.to(self.device) for key, value in batch.items()}
                embeddings = self.model(batch)['sentence_embedding'].float()
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
                for offset, vector in enumerate(embeddings):
                    position = start + offset
                    total[owners[position]] += vector * weights[position]
        total /= np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(total).all():
            raise ValueError('Nonfinite Sentence-Transformer output')
        return total


def build_node_encoder(config, question_encoder):
    """Smoke uses only the explicitly marked hash encoder; production is local-only."""
    node_config = config.get('node_encoder')
    if not node_config or question_encoder.is_smoke:
        return question_encoder
    node_config = dict(node_config)
    if config.get('_config_path'):
        from tarqa.config import resolve_path
        node_config['model_path'] = str(resolve_path(Path(node_config['model_path']).expanduser(), config['_config_path']))
    key = json.dumps(node_config, sort_keys=True)
    # E2E reuses its question encoder across domains; reuse the small encoder too.
    if not hasattr(question_encoder, '_tarqa_node_encoders'):
        question_encoder._tarqa_node_encoders = {}
    cache = question_encoder._tarqa_node_encoders
    if key not in cache:
        cache[key] = FrozenSentenceEncoder(node_config)
    return cache[key]
