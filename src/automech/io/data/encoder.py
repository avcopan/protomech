from typing import Any


def simple(data: dict[Any, Any]) -> str:
    """Default data encoder.

    :param data: Data
    :return: Encoded string
    """
    return ", ".join(f"{k}: {v}" for k, v in data.items() if v is not None)


def mechanalyzer(data: dict[Any, Any]) -> str:
    """Default data encoder.

    :param data: Data
    :return: Encoded string
    """
    data = {k: v for k, v in data.items() if k is not None}
    keys = ".".join(map(str, data.keys()))
    vals = ".".join(map(str, data.values()))
    return "  ".join([keys, vals])
