"""Sequence utility functions."""

from collections.abc import Sequence
from typing import List, TypeVar

T = TypeVar("T")


def duplicates(seq: Sequence[T]) -> List[T]:
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
