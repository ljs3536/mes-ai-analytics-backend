FEATURE_KEYS = ("rms", "peak", "crest", "kurtosis", "band1x", "band2x")


def _value(row: dict, key: str) -> float:
    if key in row and row[key] is not None:
        return float(row[key])
    camel = key[:-1] + key[-1].upper() if key[-1].islower() else key
    if camel in row and row[camel] is not None:
        return float(row[camel])
    raise KeyError(key)


def vector(row: dict) -> list[float] | None:
    try:
        return [_value(row, key) for key in FEATURE_KEYS]
    except (KeyError, TypeError, ValueError):
        return None
