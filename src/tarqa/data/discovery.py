"""Locate published QA/support releases without inventing a new split."""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class DomainPaths:
    domain: str
    qa: Path  # compatibility: qa/test.json, never a training source
    metadata: Path
    profiling: Path | None
    database: Path
    release_root: Path | None = None


def discover_domain(dataset_root, db_root, domain: str) -> DomainPaths:
    domain = domain.lower()
    root = Path(dataset_root).resolve()
    candidates = []
    if root.is_dir():
        for folder in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
            cfg = folder / 'config.json'
            if cfg.is_file() and (folder / 'qa/train.json').is_file():
                if str(json.loads(cfg.read_text())['domain']).lower() == domain:
                    candidates.append(folder)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'Expected one QA/support release for {domain} beneath {root}; found {candidates}. '
            'Extract the latest full release. This implementation never resplits raw QA.'
        )
    release = candidates[0]
    schemas = sorted((release / 'source').glob('metadata*.json'))
    databases = sorted((release / 'source').glob('*.duckdb'))
    if not databases and db_root:
        databases = [p for p in Path(db_root).glob('*.duckdb') if p.stem.lower() == domain]
    if len(schemas) != 1 or len(databases) != 1:
        raise FileNotFoundError(f'Need one source metadata JSON and DuckDB for {release}')
    return DomainPaths(domain, release / 'qa/test.json', schemas[0], None,
                       databases[0].resolve(), release)


def discover_domains(dataset_root, db_root, domains):
    result = {}
    for domain in domains:
        paths = discover_domain(dataset_root, db_root, domain)
        if paths.domain in result:
            raise ValueError(f'Duplicate domain: {domain}')
        result[paths.domain] = paths
    return result
