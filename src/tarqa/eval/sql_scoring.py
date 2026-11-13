from decimal import Decimal
import math

def canonical(value):
    if isinstance(value, (tuple, list)):
        return tuple(canonical(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, canonical(v)) for k, v in value.items()))
    if isinstance(value, (float, Decimal)):
        value = float(value)
        if math.isnan(value):
            return ("nan",)
        return round(value, 8)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def same_rows(actual, expected, answer_nature):
    actual, expected = canonical(actual), canonical(expected)
    if answer_nature.lower() in {"set", "sets"}:
        return set(actual) == set(expected)
    return actual == expected
