"""Frozen local text encoder; hash backend is explicitly restricted to smoke runs."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np


def render_question_representation_input(question, spec):
    return (f'Question: {question}\nRelevant tables: {", ".join(spec.relevant_tables)}\n'
            f'Reference time: {spec.reference_time}\nTask type: {spec.task_type.value if spec.task_type else None}\n'
            f'Track: {spec.track.value}')


class FrozenQuestionEncoder:
    def __init__(self, config):
        self.config = dict(config)
        self.backend = config.get('backend', 'local_llm')
        self.is_smoke = self.backend == 'hash_smoke'
        if self.backend not in {'local_llm', 'sentence_transformer', 'hash_smoke'}:
            raise ValueError(self.backend)
        if config.get('adapter_path'):
            raise ValueError('Keep question/descriptor encoder on the same frozen BASE checkpoint across folds. TSG adapters are trained separately.')
        if self.backend == 'sentence_transformer':
            from tarqa.prediction.node_encoder import FrozenSentenceEncoder
            self._sentence_encoder = FrozenSentenceEncoder(config)
            self.config = dict(self._sentence_encoder.config)
            self.dimension = self._sentence_encoder.dimension
            self.fingerprint = self._sentence_encoder.fingerprint
            self.max_length = self._sentence_encoder.max_length
            self.batch_size = self._sentence_encoder.batch_size
            self.model = self._sentence_encoder.model
            self.tokenizer = self.model.tokenizer
            return
        fingerprint_data = {'config': {k:v for k,v in self.config.items() if k not in {'model_path','tokenizer_path','adapter_path'}}, 'version': 2}
        if self.is_smoke:
            self.dimension = int(config.get('hash_dim', 64))
        else:
            import torch
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
            from tarqa.task_spec.modeling import torch_dtype_from_name
            model_path = Path(config['model_path']).expanduser().resolve()
            if not model_path.is_dir():
                raise FileNotFoundError(f'Configure a local base LLM checkpoint: {model_path}')
            from tarqa.data.releases import file_hash
            suffixes = {'.json','.safetensors','.bin','.pt','.model','.txt','.py','.tiktoken','.vocab'}
            def component_hashes(folder):
                return {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob('*'))
                        if p.is_file() and p.suffix in suffixes and not any(part.startswith('.') for part in p.relative_to(folder).parts)}
            fingerprint_data['model_files'] = component_hashes(model_path)
            source = config.get('tokenizer_path') or str(model_path)
            fingerprint_data['tokenizer_files'] = component_hashes(Path(source).expanduser().resolve())
            self.tokenizer = AutoTokenizer.from_pretrained(source,
                local_files_only=True, trust_remote_code=bool(config.get('trust_remote_code', False)))
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
            model_class = AutoModel if config.get('base_model_only', False) else AutoModelForCausalLM
            self.model = model_class.from_pretrained(str(model_path), local_files_only=True,
                trust_remote_code=bool(config.get('trust_remote_code', False)),
                torch_dtype=torch_dtype_from_name(config.get('torch_dtype', 'auto')),
                device_map='auto').eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.dimension = self.model.config.hidden_size
        self.fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        self.max_length = int(config.get('max_length', 1024))
        self.pooling = config.get('pooling', 'mean')
        self.batch_size = int(config.get('batch_size', 8))

    def encode(self, texts):
        if self.backend == 'sentence_transformer':
            return self._sentence_encoder.encode(texts)
        if self.is_smoke:
            output = np.zeros((len(texts), self.dimension), np.float32)
            for i, text in enumerate(texts):
                for word in re.findall(r'\w+|[^\w\s]', text.lower()):
                    h = hashlib.sha256(word.encode()).digest()
                    output[i, int.from_bytes(h[:4], 'big') % self.dimension] += 1 if h[4] % 2 else -1
            output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
            return output
        import torch
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[start:start+self.batch_size], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors='pt')
                tokens = {k: v.to(self.model.get_input_embeddings().weight.device) for k, v in tokens.items()}
                if self.config.get('base_model_only', False):
                    # Node encoding needs final hidden states only, not vocabulary logits
                    # or every intermediate layer's activation tensors.
                    hidden = self.model(**tokens, output_hidden_states=False, use_cache=False,
                                        return_dict=True).last_hidden_state
                else:
                    hidden = self.model(**tokens, output_hidden_states=True, use_cache=False).hidden_states[-1]
                mask = tokens['attention_mask'].to(hidden.device)
                if self.pooling == 'mean':
                    pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                elif self.pooling == 'last_token':
                    pooled = hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(1)-1]
                else:
                    raise ValueError(self.pooling)
                chunks.append(pooled.float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, self.dimension), np.float32)
