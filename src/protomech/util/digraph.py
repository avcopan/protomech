"""Digraph utilities."""

import itertools
from collections.abc import Sequence

import networkx as nx


def root_node_keys(D: nx.DiGraph) -> list:
    """Get source node keys.

    :param D: DiGraph
    :return: Node keys
    """
    return [key for key in D.nodes if D.in_degree(key) == 0]


def leaf_node_keys(D: nx.DiGraph) -> list:
    """Get source node keys.

    :param D: DiGraph
    :return: Node keys
    """
    return [key for key in D.nodes if D.out_degree(key) == 0]


def topologically_sorted_node_keys(D: nx.DiGraph) -> list:
    """Generate list of topologically sorted node keys.

    :param D: DiGraph
    :return: Node keys
    """
    return list(nx.topological_sort(D))


def all_simple_root_leaf_node_paths(
    D: nx.DiGraph,
    *,
    root_keys: Sequence[int] | None = None,
    leaf_keys: Sequence[int] | None = None,
) -> list[list]:
    """Get all simple root-to-leaf node paths.

    Optionally, specify roots and leaves.

    :param D: DiGraph
    :param root_keys: Root keys
    :param leaf_keys: Leaf keys
    :return: Paths
    """
    key1s = root_node_keys(D) if root_keys is None else root_keys
    key2s = leaf_node_keys(D) if leaf_keys is None else leaf_keys
    return list(
        itertools.chain.from_iterable(
            nx.all_simple_paths(D, k1, k2) for k1, k2 in itertools.product(key1s, key2s)
        )
    )
