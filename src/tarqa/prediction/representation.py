"""Fixed-size schema-semantic pooling; no fitted vocabulary or target-domain scaler.

Each role/path/column descriptor is embedded with the SAME frozen text encoder.
A fixed random projection reduces descriptor size. Per-statistic pooling keeps
the meaning of numeric channels consistent when table names and schemas differ.
This is a lossy adapter, not an injective relational representation.
"""
import hashlib
import json
import numpy as np

STATS = ("row_count", "count", "sum", "mean", "min", "max", "std", "distinct_count",
         "recency_days", "oldest_days", "missing", "active", "category_count")


class SchemaSemanticPool:
    def __init__(self, encoder, config=None):
        self.encoder = encoder
        self.config = {"descriptor_dim": 32, "seed": 42, **(config or {})}
        self.dim = int(self.config["descriptor_dim"])
        self.output_dim = len(STATS) * (self.dim + 2)
        self._cache, self._projection = {}, None

    @property
    def fingerprint(self):
        payload = dict(version=1, stats=STATS, config=self.config, encoder=self.encoder.fingerprint)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def prepare(self, keys):
        keys = sorted(set(keys) - set(self._cache))
        if not keys:
            return
        vectors = self.encoder.encode(["Relational statistic: " + k.replace("::", "; ") for k in keys])
        if self._projection is None:
            rng = np.random.default_rng(int(self.config["seed"]))
            self._projection = rng.normal(size=(vectors.shape[1], self.dim)).astype(np.float32) / np.sqrt(self.dim)
        projected = vectors @ self._projection
        projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        self._cache.update(zip(keys, projected))

    def transform(self, features):
        self.prepare(features.keys())
        output = np.zeros((len(STATS), self.dim + 2), np.float32)
        counts = np.zeros(len(STATS), np.int32)
        for key in sorted(features):
            stat = key.split("::")[-1]
            if stat not in STATS:
                raise ValueError(f"Unknown statistic: {stat}")
            i = STATS.index(stat)
            value = float(features[key])
            value = np.sign(value) * np.log1p(abs(value))
            output[i, :self.dim] += value * self._cache[key]
            output[i, self.dim] += value
            counts[i] += 1
        for i, n in enumerate(counts):
            if n:
                output[i, :self.dim + 1] /= np.sqrt(n)
                output[i, -1] = np.log1p(n)
        return output.ravel()
