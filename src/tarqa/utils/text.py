from __future__ import annotations

import json
import re
from typing import Any


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : idx + 1])
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object.")
                return value
    raise ValueError("Unterminated JSON object in model output.")
