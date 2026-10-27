from __future__ import annotations

from tarqa.data.qa_loader import QARecord
from tarqa.types import BindingResult, EntityAnchor


def binding_from_record(record: QARecord) -> BindingResult:
    """Use audited benchmark `target_instances`; no heuristic entity resolver."""
    return BindingResult(anchors=[record.anchor], source="dataset_target_instances", confidence=1.0)


def binding_from_anchor(anchor: EntityAnchor) -> BindingResult:
    return BindingResult(anchors=[anchor], source="explicit_anchor", confidence=1.0)
