"""Sequence utility functions."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def duplicates(seq: Sequence[T]) -> list[T]:
    """Find duplicates in a sequence.

    :param seq: Sequence
    :return: Duplicate items
    """
    seen: set[T] = set()
    dups: set[T] = set()

    for item in seq:
        if item in seen:
            dups.add(item)
        else:
            seen.add(item)

    return list(dups)
