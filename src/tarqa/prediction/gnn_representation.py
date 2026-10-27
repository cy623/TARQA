"""Frozen-LM node initialization + two synchronous, parameter-free GNN layers.

Relation means are kept separate until a fixed semantic binding operation.
This is a lossy fixed encoder, NOT a trained GNN or an injectivity guarantee.
Only downstream task heads contain learned prediction parameters.
"""
from collections import OrderedDict
import hashlib
import json
import numpy as np

from tarqa.prediction.row_graph import RowGraph


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def propagate(initial, edges, relation_vectors, num_layers=2):
    """Each layer reads only the PREVIOUS layer, never in-place node updates.

For active incoming relation r, let m=mean(h_u), a=unit(LM(r)R), d=degree.
message = m * (1+a) + log(1+d)*a
h' = unit(h + sum_r(message)/sqrt(number_of_active_relations))
Isolated nodes keep their state. Root readout retains layers 0, 1, and 2.
"""
    hidden = np.asarray(initial, dtype=np.float32)
    layers = [hidden.copy()]
    for _ in range(num_layers):
        aggregate = np.zeros_like(hidden)
        active = np.zeros(len(hidden), dtype=np.int32)
        for relation in sorted(edges):
            source, target = edges[relation]
            if not len(source):
                continue
            # Work in destination-sized buffers instead of N x E or full N per relation.
            receivers, inverse = np.unique(target, return_inverse=True)
            counts = np.bincount(inverse).astype(np.float32)
            sums = np.zeros((len(receivers), hidden.shape[1]), dtype=np.float32)
            # Bound transient E x hidden_size memory for high-degree relations.
            for start in range(0, len(source), 4096):
                np.add.at(sums, inverse[start:start+4096], hidden[source[start:start+4096]])
            mean = sums / counts[:, None]
            semantic = relation_vectors[relation]
            aggregate[receivers] += mean * (1.0 + semantic) + np.log1p(counts[:, None]) * semantic
            active[receivers] += 1
        hidden = unit(hidden + aggregate / np.sqrt(np.maximum(active, 1))[:, None]).astype(np.float32)
        layers.append(hidden.copy())
    return layers


class FrozenLMGraphPool:
    def __init__(self, encoder, config=None, *, node_encoder=None):
        supplied = dict(config or {})
        unknown = set(supplied) - {'backend', 'num_layers', 'node_dim', 'seed', 'embedding_cache_size'}
        if unknown:
            raise ValueError(f'Unknown graph representation options: {sorted(unknown)}')
        self.config = {
            **dict(backend='hetero_gnn', num_layers=2, node_dim=256, seed=42, embedding_cache_size=32768), **supplied}
        if self.config['backend'] != 'hetero_gnn' or int(self.config['num_layers']) != 2:
            raise ValueError('Use hetero_gnn with exactly 2 layers')
        self.encoder = encoder
        self.node_encoder = node_encoder or encoder
        self.dim = int(self.config['node_dim'])
        self.cache_size = int(self.config['embedding_cache_size'])
        if self.dim < 1 or self.cache_size < 0:
            raise ValueError('Invalid graph dimension/cache size')
        rng = np.random.default_rng(int(self.config['seed']))
        self.projection = (np.eye(self.dim, dtype=np.float32) if self.node_encoder.dimension == self.dim else
                           rng.normal(size=(self.node_encoder.dimension, self.dim)).astype(np.float32) / np.sqrt(self.dim))
        self.projection = self.projection.astype(np.float32)
        self.output_dim = 3 * self.dim
        self._cache = OrderedDict()

    @property
    def fingerprint(self):
        payload = dict(version=1, backend='frozen_lm_two_layer_typed_graph',
                       graph_protocol_version=1, row_serialization_version=1,
                       config={k: v for k, v in self.config.items() if k != 'embedding_cache_size'},
                       encoder=self.encoder.fingerprint, node_encoder=self.node_encoder.fingerprint)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _encode(self, texts):
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        vectors = {}
        pending = {}
        for key, text in zip(keys, texts):
            if key in self._cache:
                vectors[key] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                pending[key] = text
        # Batch BEFORE calling encode to avoid allocating N x full LM hidden_size on CPU.
        items = list(pending.items())
        for start in range(0, len(items), 128):
            part = items[start:start+128]
            raw = self.node_encoder.encode([text for _, text in part])
            embedded = unit(raw @ self.projection).astype(np.float32)
            if not np.isfinite(embedded).all():
                raise ValueError('Nonfinite frozen language-model node embedding')
            for (key, _), vector in zip(part, embedded):
                vectors[key] = vector
                if self.cache_size:
                    self._cache[key] = vector
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return np.stack([vectors[key] for key in keys]) if texts else np.empty((0, self.dim), np.float32)

    def transform(self, graph):
        if not isinstance(graph, RowGraph):
            raise TypeError('hetero_gnn requires row graphs, not SQL statistic dictionaries')
        if not graph.node_texts or not 0 <= graph.anchor_index < len(graph.node_texts):
            raise ValueError('Invalid anchor graph')
        initial = self._encode(['Historical database row: ' + text for text in graph.node_texts])
        names = sorted(graph.edges)
        relation_vectors = dict(zip(names, self._encode(names)))
        layers = propagate(initial, graph.edges, relation_vectors, num_layers=2)
        result = np.concatenate([layer[graph.anchor_index] for layer in layers]).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite graph readout')
        return result
