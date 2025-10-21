import functools
import itertools
import operator
import re
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import autochem as ac
import automol
import more_itertools as mit
import networkx as nx
import numpy as np
import polars as pl
import pydantic
from autochem.rate.data import Rate, RateFit

import automech
from automech import Mechanism
from automech.reaction import Reaction

from ..util import array, sequence
from . import inp, out


class Feature(pydantic.BaseModel, ABC):
    energy: float
    fake: bool = False
    mess_body: str

    @property
    @abstractmethod
    def label(self) -> str:
        """Label."""
        pass

    @abstractmethod
    def mess_block(self, label_dct: Mapping[int, str]) -> str:
        """MESS block."""
        pass

    def remove_mess_body_tunneling(self) -> None:
        """Remove Eckart tunneling from MESS body."""
        regex = re.compile(
            r"^\s*Tunneling.*?^\s*End\s*\n?", flags=re.DOTALL | re.MULTILINE
        )
        self.mess_body = regex.sub("", self.mess_body)

    def update_mess_body_energy(self) -> None:
        """Update MESS block with the current energy value."""
        energy = self.energy
        mess_body = self.mess_body
        ground_energy_regex = re.compile(r"^\s*GroundEnergy.*$", flags=re.MULTILINE)
        zero_energy_regex = re.compile(r"^\s*ZeroEnergy.*$", flags=re.MULTILINE)
        comment_energy_regex = re.compile(r"^\s*!\s*ZeroEnergy.*$", flags=re.MULTILINE)
        if ground_energy_regex.search(mess_body):
            energy_line = f"    GroundEnergy[kcal/mol]      {energy:.2f}"
            self.mess_body = ground_energy_regex.sub(energy_line, mess_body)
        elif zero_energy_regex.search(mess_body):
            energy_line = f"      ZeroEnergy[kcal/mol]      {energy:.2f}"
            self.mess_body = zero_energy_regex.sub(energy_line, mess_body)
        elif comment_energy_regex.search(mess_body):
            energy_line = f"      ZeroEnergy[kcal/mol]      {energy:.2f}"
            self.mess_body = comment_energy_regex.sub(energy_line, mess_body)
        else:
            msg = f"Unable to find GroundEnergy or ZeroEnergy line in MESS body:\n{self.mess_body}"
            raise ValueError(msg)


class Node(Feature):
    key: int

    @property
    @abstractmethod
    def names_list(self) -> list[str]:
        """Names."""
        pass


class UnimolNode(Node):
    type: Literal["unimol"] = "unimol"
    name: str

    @property
    def names_list(self) -> list[str]:
        """Label."""
        return [self.name]

    @property
    def label(self) -> str:
        """Label."""
        return self.name

    def mess_block(self, label_dct: Mapping[int, str]) -> str:
        """MESS block."""
        return f"Well  {self.label}\n{self.mess_body}"


class NmolNode(Node):
    type: Literal["nmol"] = "nmol"
    names: Annotated[list[str], pydantic.AfterValidator(sorted)]
    interacting: bool = False

    @property
    def label(self) -> str:
        """Label."""
        label = "+".join(self.names)
        if self.fake:
            label = f"FakeW-{label}"

        return label

    @property
    def names_list(self) -> list[str]:
        """Label."""
        return self.names

    def mess_block(self, label_dct: Mapping[int, str]) -> str:
        """MESS block."""
        if self.interacting:
            return f"Well  {self.label}\n{self.mess_body}"

        return f"Bimolecular  {self.label}\n{self.mess_body}"


class Edge(Feature):
    key: Annotated[frozenset[int], pydantic.BeforeValidator(frozenset)] = (
        pydantic.Field(min_length=2, max_length=2)
    )
    name: str
    energy: float
    barrierless: bool = False

    @property
    def label(self) -> str:
        """Label."""
        return self.name

    def mess_block(self, label_dct: Mapping[int, str]) -> str:
        """MESS block."""
        key1, key2 = sorted(self.key)
        return f"Barrier  {self.label} {label_dct[key1]} {label_dct[key2]}\n{self.mess_body}"


class Surface(pydantic.BaseModel):
    """Potential energy surface."""

    nodes: list[Node]
    edges: list[Edge]
    rates: dict[tuple[int, int], Rate] = {}
    rate_fits: dict[tuple[int, int], RateFit] = {}

    mess_header: str

    @pydantic.model_validator(mode="after")
    def _validate_keys(self):
        keys = [n.key for n in self.nodes]

        dup_keys = sequence.duplicates(keys)
        if dup_keys:
            raise ValueError(f"Non-unique node keys: {dup_keys}")

        dup_labels = sequence.duplicates([n.label for n in self.nodes])
        if dup_labels:
            raise ValueError(f"Non-unique node labels: {dup_labels}")

        edge_keys = [e.key for e in self.edges]
        miss_edge_keys = [k for k in edge_keys if not k <= set(keys)]
        if miss_edge_keys:
            raise ValueError(
                f"Edge keys {miss_edge_keys} do not match node keys {keys}"
            )

        dup_edge_keys = sequence.duplicates(edge_keys)
        if dup_edge_keys:
            for edge_key in dup_edge_keys:
                print(next(e.label for e in self.edges if e.key == edge_key))
            raise ValueError(f"Non-unique edge keys: {dup_edge_keys}")

        return self


# Properties
def node_keys(surf: Surface, labels: Collection[str] | None = None) -> list[int]:
    """Node keys for the surface."""
    if labels is not None:
        return [n.key for n in surf.nodes if n.label in labels]
    return [n.key for n in surf.nodes]


def edge_keys(surf: Surface) -> list[frozenset[int]]:
    """Node keys for the surface."""
    return [e.key for e in surf.edges]


def enantiomer_name(name: str, suffix0: str = "0", suffix1: str = "1") -> str:
    """Translate name into enantiomer name by suffix.

    :param name: Name
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Name
    """
    if name.endswith(suffix0):
        name = name.removesuffix(suffix0) + suffix1
    elif name.endswith(suffix1):
        name = name.removesuffix(suffix1) + suffix0
    return name


def enantiomer_names_list(
    names: Sequence[str], suffix0: str = "0", suffix1: str = "1"
) -> list[str]:
    """Translate name into enantiomer name by suffix.

    :param names: Names
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Name
    """
    return [enantiomer_name(n, suffix0=suffix0, suffix1=suffix1) for n in names]


def enantiomer_node_mapping(
    surf: Surface, *, extend: bool = False, suffix0: str = "0", suffix1: str = "1"
) -> dict[int, int]:
    """Get mapping of chiral nodes onto their enantiomers.

    :param surf: Surface
    :param extend: Optionally, extend the mapping to map remaining nodes to themselves
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Mapping
    """
    enant_dct = {}
    for node in surf.nodes:
        mirror_names = enantiomer_names_list(
            node.names_list, suffix0=suffix0, suffix1=suffix1
        )
        mirror_key = node_key_from_names(surf, mirror_names, fake=node.fake)
        if mirror_key is None:
            msg = f"No enantiomer found for node {node.key}: {node.label}"
            raise ValueError(msg)
        if node.key != mirror_key or extend:
            enant_dct[node.key] = mirror_key
    return enant_dct


def enantiomer_rate_mapping(
    surf: Surface, *, extend: bool = False, suffix0: str = "0", suffix1: str = "1"
) -> dict[tuple[int, int], tuple[int, int]]:
    """Get mapping of chiral nodes onto their enantiomers.

    :param surf: Surface
    :param extend: Optionally, extend the mapping to map remaining nodes to themselves
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Mapping
    """
    enant_dct = enantiomer_node_mapping(
        surf, extend=True, suffix0=suffix0, suffix1=suffix1
    )
    enant_rate_dct = {}
    for rate_key in rate_keys(surf):
        key1, key2 = rate_key
        mirror_rate_key = (enant_dct[key1], enant_dct[key2])
        if extend or mirror_rate_key != rate_key:
            enant_rate_dct[rate_key] = mirror_rate_key
    return enant_rate_dct


def enantiomer_rate_pairs(
    surf: Surface, suffix0: str = "0", suffix1: str = "1"
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Get pairs of enantiomer rates.

    :param surf: Surface
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Rate key pairs
    """
    enant_rate_dct = enantiomer_rate_mapping(surf, suffix0=suffix0, suffix1=suffix1)
    rate_key_pairs = []
    for rate_key1, rate_key2 in enant_rate_dct.items():
        rate_key1, rate_key2 = sorted([rate_key1, rate_key2])
        if (rate_key1, rate_key2) not in rate_key_pairs:
            rate_key_pairs.append((rate_key1, rate_key2))
    return rate_key_pairs


def enforce_equal_enantiomer_rates(
    surf: Surface, *, tol: float = 0.1, suffix0: str = "0", suffix1: str = "1"
) -> Surface:
    """Enforce equal enantiomer rates for enantiomers.

    Rates that are similar within a user-defined tolerance will be replaced with
    their average across the two enantiomers.

    Rates that differ beyond this threshold will be replaced with NaN.

    :param surf: Surface
    :param tol: Threshold for detecting a mismatch, resulting in NaN
    :param suffix0: First enantiomer suffix, defaults to "0"
    :param suffix1: Second enantiomer suffix, defaults to "1"
    :return: Surface
    """
    rates = {rate_key: rate.model_copy() for rate_key, rate in surf.rates.items()}
    rate_key_pairs = enantiomer_rate_pairs(surf, suffix0=suffix0, suffix1=suffix1)
    for rate_key1, rate_key2 in rate_key_pairs:
        rate1 = rates[rate_key1]
        rate2 = rates[rate_key2]
        rate = rate1.merge_equivalent(rate2, tol=tol)
        rates[rate_key1] = rate
        rates[rate_key2] = rate.model_copy()
    return surf.model_copy(update={"rates": rates})


def rate_keys(
    surf: Surface,
    *,
    P_vals: Sequence[float] = (),
    empty: bool = True,
    direct: bool = True,
    well_skipping: bool = True,
) -> list[tuple[int, int]]:
    """Keys for direct rates."""
    edge_keys_ = edge_keys(surf)
    rate_keys_ = []
    for rate_key, rate in surf.rates.items():
        is_direct = frozenset(rate_key) in edge_keys_
        is_included = direct if is_direct else well_skipping
        ok_P_vals = not P_vals or rate.has_pressure_data(P=P_vals)
        ok_empty = empty or not rate.is_empty()
        if is_included and ok_P_vals and ok_empty:
            rate_keys_.append(rate_key)
    return rate_keys_


def node_label_dict(surf: Surface) -> dict[int, str]:
    """Node labels for the surface."""
    return {n.key: n.label for n in surf.nodes}


def fake_node_keys(surf: Surface) -> list[int]:
    """Keys of fake wells in surface.

    :param surf: Surface
    :return: Keys of fake wells
    """
    return [n.key for n in surf.nodes if n.fake]


def fake_edge_keys(surf: Surface) -> list[frozenset[int]]:
    """Keys of fake wells in surface.

    :param surf: Surface
    :return: Keys of fake wells
    """
    return [e.key for e in surf.edges if e.fake]


def fake_node_bimol_keys(surf: Surface) -> dict[int, int]:
    """Get mapping from fake nodes to bimolecular nodes whose complex they represent.

    :param surf: Surface
    :return: Mapping from fake nodes to bimolecular sources
    """
    bimol_dct = {}
    fake_keys = set(fake_node_keys(surf))
    for fake_edge_key in fake_edge_keys(surf):
        (fake_key,) = fake_edge_key & fake_keys
        (real_key,) = fake_edge_key - fake_keys
        bimol_dct[fake_key] = real_key
    return bimol_dct


def node_object_from_label(surf: Surface, label: str) -> Node:
    """Look up node object by label.

    :param surf: Surface
    :param label: Label
    :return: Node
    """
    return next(n for n in surf.nodes if n.label == label)


def edge_object_from_label(surf: Surface, label: str) -> Edge:
    """Look up edge object by label.

    :param surf: Surface
    :param label: Label
    :return: edge
    """
    return next(e for e in surf.edges if e.label == label)


def node_object(
    surf: Surface, key: int, copy: bool = False, deep: bool = False
) -> Node:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    node = get_node_object(surf, key, copy=copy, deep=deep)
    if node is None:
        msg = f"Key {key} is not associated with a node:\n{surf.model_dump()}"
        raise ValueError(msg)
    return node


def edge_object(
    surf: Surface, key: Collection[int], copy: bool = False, deep: bool = False
) -> Edge:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    edge = get_edge_object(surf, key, copy=copy, deep=deep)
    if edge is None:
        msg = f"Key {key} is not associated with an edge:\n{surf.model_dump()}"
        raise ValueError(msg)
    return edge


def edge_chemkin_equation(surf: Surface, key: Collection[int]) -> str:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    key1, key2 = key if isinstance(key, Sequence) else sorted(key)
    node1 = node_object(surf, key1)
    node2 = node_object(surf, key2)
    reac = " + ".join(node1.names_list)
    prod = " + ".join(node2.names_list)
    return " = ".join([reac, prod])


def get_node_object(
    surf: Surface, key: int, copy: bool = False, deep: bool = False
) -> Node | None:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    node = next((n for n in surf.nodes if n.key == key), None)
    return None if node is None else node.model_copy(deep=deep) if copy else node


def get_edge_object(
    surf: Surface, key: Collection[int], copy: bool = False, deep: bool = False
) -> Edge | None:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    key = key if isinstance(key, frozenset) else frozenset(key)
    edge = next((e for e in surf.edges if e.key == key), None)
    return None if edge is None else edge.model_copy(deep=deep) if copy else edge


def node_neighbors(surf: Surface, key: int, skip_fake: bool = False) -> list[int]:
    """Keys of neighboring nodes.

    :param surf: Surface
    :param key: Key
    :param skip_fake: Whether to skip fake over fake neighbors and include the next neighbors
    :return: Neighbor keys
    """
    gra = graph(surf)
    nkeys = []
    for nkey, ndata in gra[key].items():
        if skip_fake and ndata["fake"]:
            for skip_nkey in gra[nkey]:
                if skip_nkey != key:
                    nkeys.append(skip_nkey)
        else:
            nkeys.append(nkey)

    return nkeys


def node_key_from_label(surf: Surface, label: str) -> int:
    key = next((n.key for n in surf.nodes if n.label == label), None)
    if key is None:
        msg = f"No node found matching label {label}."
        raise ValueError(msg)
    return key


def node_key_from_names(
    surf: Surface, names: list[str], fake: bool = False
) -> int | None:
    """Look up node key given names.

    :param surf: Surface
    :param names: Species names
    :param fake: Whether to look for a fake node
    :return: Node key
    """
    keys = [
        n.key
        for n in surf.nodes
        if sorted(names) == sorted(n.names_list) and not n.fake ^ fake
    ]

    if not keys:
        return None

    if len(keys) > 1:
        msg = f"Multiple nodes found with names {names}: {keys}"
        raise ValueError(msg)

    (key,) = keys
    return key


def node_keys_containing(
    surf: Surface, name: str, nmol: bool = True, fake: bool = False
) -> list[int]:
    """Get list of node keys containing a species name.

    :param surf: Surface
    :param name: Name
    :param nmol: Whether to look for n-molecular nodes with n>1
    :param fake: Whether to look for fake nodes
    :return: Node keys
    """
    return [
        n.key
        for n in surf.nodes
        if name in n.names_list
        and not (nmol and len(n.names_list) == 1)
        and not (n.fake ^ fake)
    ]


def shortest_path(surf: Surface, key1: int, key2: int) -> list[int]:
    """Get the shortest path between two nodes.

    :param surf: Surface
    :param key1: Node 1 key
    :param key2: Node 2 key
    :raises ValueError: If node key is not in the surface
    :return: Node keys for path
    """
    gra = graph(surf)
    return nx.shortest_path(gra, source=key1, target=key2)


# Transformations
def update_keys(surf: Surface, key_dct: Mapping[int, int]) -> Surface:
    """Update keys for surface.

    :param surf: Surface
    :param key_dct: Mapping from old to new keys
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    for node in surf.nodes:
        node.key = key_dct[node.key]
    for edge in surf.edges:
        key1, key2 = edge.key
        edge.key = frozenset({key_dct[key1], key_dct[key2]})
    return surf


def update_keys_by_label(surf: Surface, key_dct: Mapping[str, int]) -> Surface:
    """Update keys for surface.

    :param surf: Surface
    :param key_dct: Mapping from labels to new keys
    :return: Surface
    """
    key_dct_ = {n.key: key_dct[n.label] for n in surf.nodes}
    return update_keys(surf, key_dct_)


def shift_keys(surf: Surface, shift: int) -> Surface:
    """Shift keys for surface.

    :param surf: Surface
    :param shift: Shift
    :return: Surface
    """
    key_dct = {k: k + shift for k in node_keys(surf)}
    return update_keys(surf, key_dct)


def shift_energies(surf: Surface, shift: float) -> Surface:
    """Shift energies for surface.

    :param surf: Surface
    :param shift: Shift
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    for node in surf.nodes:
        node.energy += shift
        node.update_mess_body_energy()
    for edge in surf.edges:
        edge.energy += shift
        edge.update_mess_body_energy()
    return surf


def set_no_fake_well_extension(surf: Surface) -> Surface:
    """Turn off well extension for fake wells.

    :param surf: Surface
    :return: Surface
    """
    return set_no_well_extension(surf, fake_node_keys(surf))


def set_no_well_extension(surf: Surface, keys: Sequence[int]) -> Surface:
    """Turn off well extension for given keys.

    :param surf: Surface
    :param keys: Keys of wells to turn off
    :return: Surface with turned off well extension
    """
    nodes = []
    for node0 in surf.nodes:
        if node0.key in keys:
            node = node0.model_copy()
            node.mess_body = f"  NoWellExtension\n{node.mess_body}"
            nodes.append(node)
        else:
            nodes.append(node0)

    return surf.model_copy(update={"nodes": nodes})


def remove_nodes(surf: Surface, keys: Collection[int]) -> Surface:
    """Remove nodes from surface, along with their associated edges.

    :param surf: Surface
    :param keys: Keys of nodes to remove
    :return: Surface
    """
    keys = set(keys)
    nodes = [n for n in surf.nodes if n.key not in keys]
    edges = [e for e in surf.edges if not e.key & keys]
    rates = {k: v for k, v in surf.rates.items() if not set(k) & keys}
    return Surface(nodes=nodes, edges=edges, mess_header=surf.mess_header, rates=rates)


def remove_edges(surf: Surface, keys: Collection[Collection[int]]) -> Surface:
    """Remove edges from surface.

    :param surf: Surface
    :param keys: Keys of edges to remove
    :return: Surface
    """
    keys = list(map(frozenset, keys))
    edges = [e for e in surf.edges if e.key not in keys]
    rates = {k: v for k, v in surf.rates.items() if frozenset(k) not in keys}
    return Surface(
        nodes=surf.nodes, edges=edges, mess_header=surf.mess_header, rates=rates
    )


def remove_isolates(surf: Surface) -> Surface:
    """Remove isolated nodes from surface.

    :param surf: Surface
    :return: Surface
    """
    return remove_nodes(surf, keys=list(nx.isolates(graph(surf))))


def remove_well_skipping_rates(
    surf: Surface, keys: Collection[Collection[int]]
) -> Surface:
    """Remove well skipping rates.

    :param surf: Surface
    :param keys: Keys of well-skipping rates to remove
    :return: Surface
    """
    edge_keys_ = edge_keys(surf)
    for key in keys:
        if frozenset(key) in edge_keys_:
            msg = f"Rate {key} belongs to an edge and is not well-skipping."
            raise ValueError(msg)

    drop_keys = list(map(tuple, map(sorted, keys)))
    drop_keys = [*drop_keys, *map(tuple, map(reversed, drop_keys))]

    surf = surf.model_copy(deep=True)
    surf.rates = {k: v for k, v in surf.rates.items() if k not in drop_keys}
    return surf


def clear_node_rates(surf: Surface, keys: Collection[int]) -> Surface:
    """Clear rates associated with a node (well-skipping rates are dropped).

    :param surf: Surface
    :param keys: Keys of nodes to remove
    :return: Surface
    """
    edge_keys_ = edge_keys(surf)
    rates = {}
    for rate_key, rate in surf.rates.items():
        key = next((k for k in keys if k in rate_key), None)
        if key is None:
            rates[rate_key] = rate.model_copy()
        elif frozenset(rate_key) in edge_keys_:
            rates[rate_key] = rate.clear()

    return Surface(
        nodes=surf.nodes, edges=surf.edges, mess_header=surf.mess_header, rates=rates
    )


def clear_rates(surf: Surface, keys: Collection[tuple[int, int]]) -> Surface:
    """Clear rates.

    :param surf: Surface
    :param keys: Rate keys
    :return: Surface
    """
    rates = {}
    for rate_key, rate in surf.rates.items():
        if rate_key in keys:
            rates[rate_key] = rate.clear()
        else:
            rates[rate_key] = rate.model_copy()

    return surf.model_copy(update={"rates": rates})


def clear_unfittable_pressures(surf: Surface) -> Surface:
    """Drop unfittable pressures from rates.

    :param surf: Surface
    :return: Surface
    """
    rates = {
        rate_key: rate.clear_pressures(rate.unfittable_pressures())
        for rate_key, rate in surf.rates.items()
    }
    return surf.model_copy(update={"rates": rates})


def extend(
    surf: Surface, nodes: Collection[Node] = (), edges: Collection[Edge] = ()
) -> Surface:
    """Extend a surface by adding nodes and edges

    :param surf: Surface
    :param nodes: Nodes to add
    :param edges: Edges to add
    :return: Surface
    """
    nodes = [*surf.nodes, *nodes]
    edges = [*surf.edges, *edges]
    return Surface(
        nodes=nodes, edges=edges, mess_header=surf.mess_header, rates=surf.rates
    )


def node_induced_subsurface(surf: Surface, keys: Collection[int]) -> Surface:
    """Get node-induced sub-network.

    :param surf: Surface
    :param keys: Node keys
    :return: Surface
    """
    keys = set(keys)
    nodes = [n for n in surf.nodes if n.key in keys]
    edges = [e for e in surf.edges if e.key <= keys]
    return Surface(
        nodes=nodes, edges=edges, mess_header=surf.mess_header, rates=surf.rates
    )


def split_stoichiometries(surf: Surface, mech: Mechanism) -> dict[str, Surface]:
    """Split surface into distinct stoichiometries.

    :param surf: Surface
    :param mech: Mechanism
    :return: Surface
    """
    tmp_col = automech.util.df_.c_.temp()
    rxn_df = automech.util.df_.map_(
        mech.reactions,
        Reaction.formula,  # pyright: ignore[reportArgumentType]
        tmp_col,
        automol.form.string,
        dtype_=pl.String(),
    )

    rcts = rxn_df.get_column(Reaction.reactants).to_list()  # pyright: ignore[reportArgumentType]
    prds = rxn_df.get_column(Reaction.products).to_list()  # pyright: ignore[reportArgumentType]
    stoichs = rxn_df.get_column(tmp_col).to_list()  # pyright: ignore[reportArgumentType]
    node_stoich = dict(zip(map(tuple, rcts + prds), stoichs * 2, strict=True))
    # Group nodes by stoichiometry
    stoich_keys = defaultdict(list)
    for node in surf.nodes:
        stoich = node_stoich[tuple(node.names_list)]
        stoich_keys[stoich].append(node.key)

    return {
        stoich: node_induced_subsurface(surf, keys)
        for stoich, keys in stoich_keys.items()
    }


def merge_resonant_instabilities(surf: Surface, mech: Mechanism) -> Surface:
    """Merge resonant instabilities.

    Two cases:

    1. Unimolecular instabilities (same PES):

        A -> B*
        B* -> Y + Z

    2. N-molecular instabilities (different PESs, prompt-like):

        A -> B* + C
        B* -> Y + Z

    :param surf: Surface
    :param mech: Mechanism
    :return: Surface
    """
    rxn_df = automech.resonant_instability_reactions(mech)
    instab_rxns = rxn_df.reactions.select(
        [Reaction.reactants, Reaction.products]
    ).rows()

    instab_path_dct: dict[str, list[int]] = {}
    for (rct_name,), prd_names in instab_rxns:
        rct_key = node_key_from_names(surf, [rct_name])
        prd_key = node_key_from_names(surf, prd_names)
        if rct_key is not None and prd_key is not None:
            instab_path_dct[rct_name] = shortest_path(surf, rct_key, prd_key)

    new_node_key_dct = {}
    all_instab_path_keys = set(itertools.chain.from_iterable(instab_path_dct.values()))
    drop_keys = all_instab_path_keys.copy()
    keep_keys = set()
    for instab_name, instab_path in instab_path_dct.items():
        rct_key, *_, prd_key = instab_path

        # 1. For unimolecular instability:
        #   - Iterate over node neighbors not along the instability path
        case1_nkeys = set(node_neighbors(surf, rct_key)) - all_instab_path_keys
        for conn_key1 in case1_nkeys:
            conn_key2 = instab_path[1]
            edge_key0 = [conn_key1, rct_key]
            edge_key = [conn_key1, conn_key2]
            edge = edge_object(surf, edge_key0, copy=True)
            edge.key = frozenset(edge_key)
            surf = remove_edges(surf, [edge_key0])
            surf = extend(surf, edges=[edge])
            keep_keys.update(instab_path[1:])

        # 2. For n-molecular instability:
        #   - Iterate over n-molecular nodes containing the unstable spacies
        case2_keys = node_keys_containing(surf, instab_name)
        for instab_key in case2_keys:
            # Create a new node to containing the instability products
            new_node = instability_product_node(surf, instab_key, rct_key, prd_key)

            if new_node.label in new_node_key_dct:
                new_key = new_node_key_dct[new_node.label]
                new_node.key = new_key
                new_nodes = []
            else:
                new_key = max(node_keys(surf)) + 1
                new_node.key = new_key
                new_nodes = [new_node]

            # This itertor gets consumed at its first use, to make sure we don't
            # create duplicates below
            new_nodes_iter = iter(new_nodes)

            # Iterate over neighbors of these nodes, skipping fake wells
            # These are the neighbors we want to connect to
            conn_keys = node_neighbors(surf, instab_key, skip_fake=True)
            for conn_key in conn_keys:
                # Add node, if not already added, along with edge
                surf = extend(surf, nodes=list(new_nodes_iter))
                new_node_key_dct[new_node.label] = new_node.key

                # Get the path from the connection node to the n-molecular node
                conn_path = shortest_path(surf, conn_key, instab_key)
                # Create an edge to the new node
                src_edge_key = conn_path[:2]
                new_edge_key = [conn_key, new_key]
                surf = update_instability_product_edge(surf, src_edge_key, new_edge_key)
                # Remove n-molecular node and fake well
                drop_keys.update(conn_path[1:])

    surf = remove_nodes(surf, drop_keys - keep_keys)
    return surf


def correct_fake_well_energies(surf: Surface, *, in_place: bool = False) -> Surface:
    """Correct fake well energies to fall below their lowest associated barrier.

    :param surf: Surface
    :param in_place: Whether to modify the surface in place
    :return: Surface
    """
    surf = surf if in_place else surf.model_copy(deep=True)
    gra = graph(surf)
    # Loop over fake nodes
    for node in surf.nodes:
        if node.fake:
            key = node.key
            edges = [edge_object(surf, [key, nkey]) for nkey in gra[key]]
            # Determine the energy 3 kcal/mol below the lowest barrier
            energy = min(e.energy for e in edges) - 3
            # If this is less than the current node energy, update it
            if energy < node.energy:
                node.energy = energy
                node.update_mess_body_energy()
    return surf


def set_mess_header_temperature_list(
    surf: Surface, temperatures: Sequence[float], *, in_place: bool = False
) -> Surface:
    """Set the temperature list in the MESS header.

    :param surf: Surface
    :param in_place: Whether to modify the surface in place
    :return: Surface
    """
    surf = surf if in_place else surf.model_copy(deep=True)
    temperature_str = "  ".join(f"{t:.1f}" for t in temperatures)
    line = f"TemperatureList[K]                     {temperature_str}"
    line_regex = re.compile(r"^\s*TemperatureList\[K\].*$", re.MULTILINE)
    mess_header = surf.mess_header
    if not line_regex.search(mess_header):
        msg = f"No TemperatureList[K] line found in MESS header: {mess_header}"
        raise ValueError(msg)
    surf.mess_header = line_regex.sub(line, mess_header)
    return surf


# Create nodes
def update_instability_product_edge(
    surf: Surface, src_edge_key: Collection[int], edge_key: Collection[int]
) -> Surface:
    """Create/update edge connecting to an instability product.

    :param surf: Surface
    :param src_edge_key: Edge key of source describing the barrier
    :param edge_key: Edge key of target, to avoid creating duplicate edges
    :return: Surface
    """
    src_edge = edge_object(surf, src_edge_key)
    edge = get_edge_object(surf, edge_key)
    if edge is None:
        # Copy the source edge and set the key
        edge = src_edge.model_copy()
        edge.key = frozenset(edge_key)
        surf = extend(surf, edges=[edge])
    # Make MESS body a Union
    # (Does *not* copy! Updates the edge in-place!)
    else:
        mess_body0 = edge.mess_body
        if "Union" not in mess_body0:
            mess_body0 = textwrap.indent(mess_body0, "  ")
            mess_body0 = f"  Union\n{mess_body0}"
        # If this is already a Union, drop the trailing End keyword
        else:
            lines = mess_body0.strip().splitlines()
            assert "End" in lines[-1], f"Sanity check: 'End' not in '{lines[-1]}'"
            mess_body0 = "\n".join(lines[:-1])
        new_mess_body = textwrap.indent(src_edge.mess_body, "  ")
        edge.mess_body = f"{mess_body0}\n{new_mess_body}\nEnd  ! Union"

    return surf


def instability_product_node(
    surf: Surface, key: int, rct_key: int, prd_key: int
) -> NmolNode:
    """Generate a new node describing the product of an unstable one.

    :param surf: Surface
    :param key: Node key
    :param rct_key: Instability reactant key
    :param prd_key: Instability product key
    :return: Node
    """
    node = node_object(surf, key, copy=True)
    rct_node = node_object(surf, rct_key)
    prd_node = node_object(surf, prd_key)

    if not isinstance(rct_node, UnimolNode):
        msg = f"Instability reactant must be a unimolecular node: {rct_node}"
        raise ValueError(msg)

    if isinstance(node, NmolNode):
        (rct_name,) = rct_node.names_list
        prd_names = prd_node.names_list
        prd_energy = prd_node.energy - rct_node.energy

        names = node.names.copy()
        names.remove(rct_name)
        names.extend(prd_names)
        node.names = sorted(names)
        node.energy += prd_energy
        node.mess_body = f"  ! ZeroEnergy[kcal/mol]      {node.energy:.2f}\n  Dummy"

    else:
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    return node


# N-ary operations
def align_keys(surfs: Sequence[Surface]) -> list[Surface]:
    """Align keys across multiple surfaces.

    :param surfs: Surfaces
    :return: Surfaces
    """
    if not surfs:
        return []

    # Determine new key for each unique node
    all_nodes = itertools.chain.from_iterable(s.nodes for s in surfs)
    label_key_dct = {}
    nodes = []
    for key, node0 in enumerate(mit.unique_everseen(all_nodes, key=lambda n: n.label)):
        node = node0.model_copy(update={"key": key})
        nodes.append(node)
        label_key_dct[node.label] = node.key

    return [update_keys_by_label(surf, label_key_dct) for surf in surfs]


def align_energies(surfs: Sequence[Surface]) -> list[Surface]:
    """Align keys across multiple surfaces.

    :param surfs: Surfaces
    :return: Surfaces
    """
    nsurf = len(surfs)
    shift_mat = np.full((nsurf,) * 2, np.nan, dtype=float)
    for i1, i2 in itertools.combinations(range(nsurf), r=2):
        surf1 = surfs[i1]
        surf2 = surfs[i2]

        node2_dct = {n.label: n for n in surf2.nodes}
        node_pairs = [
            (n1, node2_dct[n1.label]) for n1 in surf1.nodes if n1.label in node2_dct
        ]
        pair_shifts = [n2.energy - n1.energy for (n1, n2) in node_pairs if not n1.fake]

        if not pair_shifts:
            shift_mat[i1, i2] = shift_mat[i2, i1] = np.nan
        elif np.allclose(pair_shifts[0], pair_shifts):
            shift_mat[i1, i2] = np.max(pair_shifts)
            shift_mat[i2, i1] = -shift_mat[i1, i2]
        else:
            msg = f"Node shifts are not all equal: {node_pairs}"
            raise ValueError(msg)

    relative_shifts = []
    for i1, _ in enumerate(surfs):
        i2 = np.nanargmax(shift_mat[i1])
        shift = shift_mat[i1, i2]
        shift_mat[i1, :] -= shift
        shift_mat[:, i1] += shift
        relative_shifts.append(shift)
    shifts = np.subtract(relative_shifts, np.min(relative_shifts))
    return [
        shift_energies(surf, shift) for surf, shift in zip(surfs, shifts, strict=True)
    ]


def combine(surfs: Sequence[Surface]) -> Surface:
    """Combine multiple surfaces.

    WARNING: This needs more care than I initially thought: Energies are
    relative, so simply merging everything together can create conflicts.

    :param surfs: Surfaces
    :return: Combined surface
    """
    mess_header = surfs[0].mess_header

    if not surfs:
        msg = f"Need at least one surface for combination: {surfs}"
        raise ValueError(msg)

    # Determine new key for each unique node
    all_nodes = itertools.chain.from_iterable(s.nodes for s in surfs)
    label_key_dct = {}
    nodes = []
    for key, node0 in enumerate(mit.unique_everseen(all_nodes, key=lambda n: n.label)):
        node = node0.model_copy(update={"key": key})
        nodes.append(node)
        label_key_dct[node.label] = node.key

    # Update edge keys against node key update
    all_edges = []
    for surf in surfs:
        key_dct = {n.key: label_key_dct[n.label] for n in surf.nodes}
        for edge0 in surf.edges:
            key1, key2 = edge0.key
            edge_key = frozenset({key_dct[key1], key_dct[key2]})
            edge = edge0.model_copy(update={"key": edge_key})
            all_edges.append(edge)

    edges = list(mit.unique_everseen(all_edges, key=lambda e: e.key))

    for surf in surfs:
        if surf.rates:
            msg = f"Cannot combine surface with rates: {surf}"
            raise NotImplementedError(msg)

    return Surface(nodes=nodes, edges=edges, mess_header=mess_header)


# Conversions
def graph(surf: Surface) -> nx.Graph:
    """Convert to a networkx Graph."""
    gra = nx.Graph()
    for node in surf.nodes:
        gra.add_node(node.key, **node.model_dump())
    for edge in surf.edges:
        gra.add_edge(*edge.key, **edge.model_dump())
    return gra


# I/O
def from_mess_input(mess_inp: str | Path) -> Surface:
    """Read surface from MESS input.

    :param mess_inp: MESS input
    :return: Surface
    """
    # Parse header
    header = inp.parse_header(mess_inp)

    # Parse blocks and separate node and edge data
    block_data = inp.parse_blocks(mess_inp)
    node_block_data = [d for d in block_data if d.type in ("Well", "Bimolecular")]
    edge_block_data = [d for d in block_data if d.type == "Barrier"]

    # Instantiate nodes and edges
    key_dct = {d.header: i for i, d in enumerate(node_block_data)}
    nodes = [node_from_mess_block_parse_data(d, key_dct) for d in node_block_data]
    edges = [edge_from_mess_block_parse_data(d, key_dct) for d in edge_block_data]

    return Surface(nodes=nodes, edges=edges, mess_header=header)


def with_mess_output_rates(surf: Surface, mess_out: str | Path) -> Surface:
    """Add MESS output data to a surface.

    :param surf: Surface
    :param mess_out: MESS output
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    trans_dct = out.translation_table(mess_out)
    blocks = out.rate_blocks(mess_out)
    for block in blocks:
        res = ac.util.mess.parse_output_channel(block)
        if res.id1 and res.id2:
            node1 = node_object_from_label(surf, trans_dct[res.id1])
            node2 = node_object_from_label(surf, trans_dct[res.id2])
            rate = ac.rate.data.from_mess_channel_output(
                block, order=len(node1.names_list)
            )
            surf.rates[(node1.key, node2.key)] = rate

    return surf


def absorb_fake_nodes(surf: Surface) -> Surface:
    """Absorb fake wells and integrate their rates.

    Case 1 (2 reactants, 2 products):

        Initial:    other_key (1) -> neib_key* (2) -> fake_key (3) -> bimol_key (4)
        Final:      other_key (1) -> bimol_key (4)
        Rate:       rate(1->4) += rate(1->3)

    Case 2 (1 reactant, 2 products):

        Initial:    neib_key (1) -> fake_key (2) -> bimol_key (3)
        Final:      neib_key (1) -> bimol_key (3)
        Rate:       rate(1->3) += rate(1->2)

    :param surf: Surface
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    gra = graph(surf)

    edge_dct = {}
    bimol_dct = fake_node_bimol_keys(surf)
    for fake_key, bimol_key in bimol_dct.items():
        neib_keys = set(gra[fake_key]) - {bimol_key}
        for neib_key in neib_keys:
            edge = edge_object(surf, {fake_key, neib_key}, copy=True)
            other_key = neib_key
            # Case 1 (2 reactants, 2 products)
            if neib_key in bimol_dct:
                other_key = bimol_dct[neib_key]
                add_rate = surf.rates[(other_key, fake_key)].fill_nan(nan=0.0)
                surf.rates[(other_key, bimol_key)] += add_rate
            # Case 2 (1 reactant, 2 products)
            else:
                add_rate = surf.rates[(neib_key, fake_key)].fill_nan(nan=0.0)
                surf.rates[(neib_key, bimol_key)] += add_rate
            edge.key = frozenset({other_key, bimol_key})
            edge_dct[edge.key] = edge

    surf = extend(surf, edges=edge_dct.values())
    surf = remove_nodes(surf, keys=bimol_dct.keys())
    return surf


def partially_unfittable_pressure_independent_rate_keys(
    surf: Surface,
    T_vals: Sequence[float],
    *,
    P_dep_tol: float = 0.2,
    direct: bool = True,
    well_skipping: bool = True,
    empty: bool = True,
) -> list[tuple[int, int]]:
    """Identify unfittable rates.

    :param surf: Surface
    :param min_P_count: Minimum acceptable fittable pressure count
    :return: Surface
    """
    unfit_keys = []
    rate_keys_ = rate_keys(
        surf, direct=direct, well_skipping=well_skipping, empty=empty
    )
    for rate_key in rate_keys_:
        rate = surf.rates[rate_key]
        is_partially_unfittable = not array.float_equal(
            rate.P, rate.fittable_pressures()
        )
        is_pressure_independent = not rate.is_pressure_dependent(
            T=T_vals, tol=P_dep_tol
        )
        if is_partially_unfittable and is_pressure_independent:
            unfit_keys.append(rate_key)
    return unfit_keys


def unfittable_rate_keys(
    surf: Surface,
    *,
    direct: bool = True,
    well_skipping: bool = True,
) -> list[tuple[int, int]]:
    """Identify unfittable rates.

    :param surf: Surface
    :param min_P_count: Minimum acceptable fittable pressure count
    :return: Surface
    """
    unfit_keys = []
    rate_keys_ = rate_keys(surf, direct=direct, well_skipping=well_skipping, empty=True)
    for rate_key in rate_keys_:
        rate = surf.rates[rate_key]
        if not rate.fittable_pressures():
            unfit_keys.append(rate_key)
    return unfit_keys


def branching_fractions(
    surf: Surface, T: Sequence[float], P: Sequence[float], *, z_drop: float = 3.0
) -> dict[tuple[int, int], np.ndarray]:
    """Determine branching fraction for each rate.

    Since the data tend to be noisy, one can drop outliers by Z-score.

    :param surf: Surface
    :param T: Temperatures for evaluating branching fraction
    :param P: Pressures for evaluating branching fraction
    :param z_drop: Z-score for removing outliers
    :return: Branching fractions
    """
    surf = surf.model_copy(deep=True)
    frac_dct = {}
    for key1 in node_keys(surf):
        rates = {(k1, k2): v for (k1, k2), v in surf.rates.items() if k1 == key1}
        total_rate = functools.reduce(
            operator.add, [r.fill_nan(nan=0.0) for r in rates.values()]
        )
        # If the branching fraction is below the threshold, drop it
        for rate_key, rate in rates.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                branch_frac = rate(T=T, P=P) / total_rate(T=T, P=P)
            z = np.abs((branch_frac - branch_frac.mean()) / branch_frac.std())
            branch_frac[z > z_drop] = np.nan
            frac_dct[rate_key] = branch_frac
    return frac_dct


def irrelevant_rate_keys(
    surf: Surface,
    T: Sequence[float],
    P: Sequence[float],
    *,
    direct: bool = True,
    well_skipping: bool = True,
    min_branch_frac: float = 0.01,
    z_drop: float = 3.0,
) -> list[tuple[int, int]]:
    """Identify irrelevant rates by branching fraction.

    :param surf: Surface
    :param min_branch_frac: Minimum acceptable branching fraction
    :return: Surface
    """
    edge_keys_ = edge_keys(surf)
    branch_fracs_dct = branching_fractions(surf, T=T, P=P, z_drop=z_drop)
    irrel_keys = []
    for rate_key, branch_fracs in branch_fracs_dct.items():
        edge_key = frozenset(rate_key)
        max_branch_frac = np.nanmax(np.nan_to_num(branch_fracs, nan=0.0))
        is_irrelevant = max_branch_frac >= min_branch_frac
        is_direct = edge_key in edge_keys_
        is_included = direct if is_direct else well_skipping
        if not is_irrelevant and is_included:
            irrel_keys.append(rate_key)
    return irrel_keys


def fit_rates(
    surf: Surface,
    T_drop: Sequence[float] = (),
    P_dep_tol: float = 0.2,
    A_fill: float | None = None,
) -> Surface:
    """Fit rates to Arrhenius or Plog.

    :param surf: Surface
    :param tol: Threshold for determining pressure dependence
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    surf.rate_fits = {}
    for rate_key, rate in surf.rates.items():
        if T_drop:
            rate = rate.drop_temperatures(T_drop)

        if rate.is_pressure_dependent(tol=P_dep_tol):
            rate_fit = ac.rate.data.PlogRateFit.fit(
                T=rate.T,
                P=rate.P,
                k_data=rate.k_data,
                k_high=rate.k_high,
                A_fill=A_fill,
                order=rate.order,
            )
        elif rate.is_empty():
            rate_fit = ac.rate.data.ArrheniusRateFit.fit(
                T=rate.T, k=[], A_fill=A_fill, order=rate.order
            )
        else:
            rate_fit = ac.rate.data.ArrheniusRateFit.fit(
                T=rate.T, k=rate.high_pressure_values(), A_fill=A_fill, order=rate.order
            )
        surf.rate_fits[rate_key] = rate_fit
    return surf


def match_rate_directions(surf: Surface, mech: Mechanism) -> Surface:
    """Picks a direction and drops the reverse rate based on an existing mechanism.

    :param surf: Surface
    :param mech: Mechanism
    :return: Surface
    """
    # Generate a dictionary mapping edge keys to rate keys
    rate_key_pair_dct = defaultdict(list)
    for rate_key in surf.rates.keys():
        rate_key_pair_dct[frozenset(rate_key)].append(rate_key)
    rate_key_pair_dct: dict[frozenset[int], list[tuple[int, int]]] = {
        k: sorted(v) for k, v in rate_key_pair_dct.items()
    }

    # Give a warning if there are unpaired rate keys
    unpaired_rate_key = next(
        (ks for ks in rate_key_pair_dct.values() if len(ks) != 2), None
    )
    if unpaired_rate_key is not None:
        msg = f"Cannot drop reverse rates with unpaired rate key: {unpaired_rate_key}"
        raise ValueError(msg)

    rcts = list(map(sorted, mech.reactions.get_column(Reaction.reactants).to_list()))  # pyright: ignore[reportArgumentType]
    prds = list(map(sorted, mech.reactions.get_column(Reaction.products).to_list()))  # pyright: ignore[reportArgumentType]
    rxns = list(zip(rcts, prds, strict=True))

    node_dct = {n.key: n for n in surf.nodes}

    # 1. Loop over edges to identify the directions of direct reactions
    rate_keys_ = []
    direct_keys = edge_keys(surf)
    for direct_key in direct_keys:
        found = False
        for rate_key in rate_key_pair_dct[direct_key]:
            key1, key2 = rate_key
            rct = sorted(node_dct[key1].names_list)
            prd = sorted(node_dct[key2].names_list)
            rxn = (rct, prd)
            rxn = next((r for r in rxns if r == rxn), None)
            if rxn:
                rate_keys_.append(rate_key)
                found = True
                break

        if not found:
            msg = f"No matching reaction found for edge {direct_key}"
            raise ValueError(msg)

    # 2. Choose well skipping directions based to maximize a "score", defined as
    #    (nr != 0, np != 0, nr + np, nr, np), where nr is the number of time the
    #    first node appears as a reactant and np is the number of time the
    #    second node appears as a product in the mechanism
    skip_keys = set(rate_key_pair_dct.keys()) - set(direct_keys)
    rct_keys, prd_keys = zip(*rate_keys_, strict=True)

    def score(rate_key: tuple[int, int]) -> tuple[bool, bool, int, int, int]:
        key1, key2 = rate_key
        nr = rct_keys.count(key1)
        np = prd_keys.count(key2)
        return (nr != 0, np != 0, nr + np, nr, np)

    for skip_key in skip_keys:
        rate_key_pair = rate_key_pair_dct[skip_key]
        rate_key = max(rate_key_pair, key=score)
        rate_keys_.append(rate_key)

    rates = {k: v for k, v in surf.rates.items() if k in rate_keys_}
    rate_fits = {k: v for k, v in surf.rate_fits.items() if k in rate_keys_}
    return surf.model_copy(update={"rates": rates, "rate_fits": rate_fits})


def mess_input(surf: Surface) -> str:
    """MESS input for surface.

    :param surf: Surface
    :return: MESS input
    """
    label_dct = node_label_dict(surf)
    mess_header = surf.mess_header
    node_blocks = [n.mess_block(label_dct) for n in surf.nodes]
    edge_blocks = [e.mess_block(label_dct) for e in surf.edges]
    assert isinstance(mess_header, str)
    assert all(isinstance(b, str) for b in node_blocks)
    assert all(isinstance(b, str) for b in edge_blocks)
    return "\n!\n".join([mess_header, *node_blocks, *edge_blocks, "End", ""])  # type: ignore


# Helpers
def node_from_mess_block_parse_data(
    block_data: inp.MessBlockParseData, key_dct: dict[str, int]
) -> Node:
    """Generate Well object from block.

    :param block_data: Parsed MESS block data
    :param key_dct: Dictionary mapping labels to IDs
    :return: Well
    """
    if block_data.type == "Barrier":
        raise ValueError("Cannot create well object from barrier block.")

    assert block_data.header in key_dct, f"{block_data.header} not in {key_dct}"
    key = key_dct[block_data.header]
    energy = block_data.energy
    block_body = block_data.body

    if block_data.type == "Bimolecular":
        names = block_data.header.split("+")
        return NmolNode(
            key=key,
            energy=energy,
            names=names,
            interacting=False,
            fake=False,
            mess_body=block_body,
        )

    if block_data.header.startswith("FakeW-"):
        names = block_data.header.removeprefix("FakeW-").split("+")
        return NmolNode(
            key=key,
            energy=energy,
            names=names,
            interacting=True,
            fake=True,
            mess_body=block_body,
        )

    assert block_data.type == "Well"
    return UnimolNode(
        key=key, energy=energy, name=block_data.header, mess_body=block_body
    )


def edge_from_mess_block_parse_data(
    block_data: inp.MessBlockParseData, key_dct: dict[str, int]
) -> Edge:
    """Generate Barrier object from block.

    :param block_data: Parsed MESS block data
    :param key_dct: Dictionary mapping labels to IDs
    :return: Barrier
    """
    if not block_data.type == "Barrier":
        raise ValueError("Cannot create barrier object from non-barrier block.")

    name, label1, label2 = block_data.header.split()
    fake = name.startswith("FakeB-")
    for label in [label1, label2]:
        if label not in key_dct:
            msg = (
                f"Cannot create edge for unknown node {label}: {block_data}\n{key_dct}"
            )
            raise ValueError(msg)

    barrierless = "PhaseSpaceTheory" in block_data.body
    return Edge(
        key=frozenset({key_dct[label1], key_dct[label2]}),
        name=name,
        energy=block_data.energy,
        fake=fake,
        barrierless=barrierless,
        mess_body=block_data.body,
    )
