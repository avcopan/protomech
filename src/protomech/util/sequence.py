"""Sequence utility functions."""

import itertools
from collections.abc import Sequence
from typing import TypeVar

import more_itertools as mit
import networkx as nx

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


def starts_with(seq: Sequence[T], prefix: Sequence[T]) -> bool:
    """Check whether a sequence starts with a prefix.

    :param seq: Sequence
    :param prefix: Prefix
    :return: Boolean
    """
    return list(seq[: len(prefix)]) == list(prefix)


def multi_ordering_digraph(seqs: Sequence[Sequence[T]]) -> nx.DiGraph:
    """Generate a digraph for topological ordering purposes.

    :param seqs: Sequences
    :return: Sequence
    """
    D = nx.DiGraph()
    for seq in seqs:
        D.add_nodes_from(seq)
        D.add_edges_from(zip(seq, seq[1:]))
    return D


def unique_adjacent_pair_sequences(
    seqs: Sequence[Sequence[T]],
) -> list[list[tuple[T, T]]]:
    """Determine unique adjacent pairs for each sequence.

    Each sequence is compared to the previous ones.

    :param seqs: Sequences
    :return: Sequences of pairs
    """
    seen_pairs = []
    pair_seqs = []
    for seq in seqs:
        pair_seq = [p for p in mit.pairwise(seq) if p not in seen_pairs]
        pair_seqs.append(pair_seq)
        seen_pairs.extend(pair_seq)
    return pair_seqs
