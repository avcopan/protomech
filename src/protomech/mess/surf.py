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

from ..util import sequence
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


def node_label_dict(surf: Surface) -> dict[int, str]:
    """Node labels for the surface."""
    return {n.key: n.label for n in surf.nodes}


def fake_well_keys(surf: Surface) -> list[int]:
    """Keys of fake wells in surface.

    :param surf: Surface
    :return: Keys of fake wells
    """
    return [n.key for n in surf.nodes if n.fake]


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
    :param key_dct: Mapping to update keys
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    for node in surf.nodes:
        node.key = key_dct[node.key]
    for edge in surf.edges:
        key1, key2 = edge.key
        edge.key = frozenset({key_dct[key1], key_dct[key2]})
    return surf


def shift_keys(surf: Surface, shift: int) -> Surface:
    """Shift keys for surface.

    :param surf: Surface
    :param shift: Shift
    :return: Surface
    """
    key_dct = {k: k + shift for k in node_keys(surf)}
    return update_keys(surf, key_dct)


def set_no_fake_well_extension(surf: Surface) -> Surface:
    """Turn off well extension for fake wells.

    :param surf: Surface
    :return: Surface
    """
    return set_no_well_extension(surf, fake_well_keys(surf))


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
def combine(surfs: Sequence[Surface]) -> Surface:
    """Combine multiple surfaces.

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


def absorb_fake_wells_with_rates(surf: Surface) -> Surface:
    """Remove fake wells and integrate their rates.

    :param surf: Surface
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    gra = graph(surf)
    fake_keys = set(fake_well_keys(surf))
    for edge in surf.edges:
        if edge.fake:
            (fake_key,) = edge.key & fake_keys
            (prod_key,) = edge.key - fake_keys
            reac_keys = set(gra[fake_key]) - {prod_key}
            for reac_key in reac_keys:
                # 1. Add r -> f rate to r -> p rate
                surf.rates[(reac_key, prod_key)] += surf.rates[(reac_key, fake_key)]
                surf.rates[(reac_key, fake_key)] *= 0.0
                # 2. Directly connect r -> p
                real_edge = edge_object(surf, [reac_key, fake_key])
                real_edge.key = frozenset([reac_key, prod_key])
            # 3. Drop the fake edge and node
            surf = remove_edges(surf, [(fake_key, prod_key)])
            surf = remove_nodes(surf, [fake_key])

    return surf


def drop_irrelevant_well_skipping_rates(surf: Surface, thresh: float = 0.01) -> Surface:
    """Drop irrelevant well-skipping rates according to a threshold.

    :param surf: Surface
    :param thresh: Branching fraction threshold for including a well-skipping rate
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    edge_keys = [e.key for e in surf.edges]
    for key1 in node_keys(surf):
        exit_rates = {(k1, k2): v for (k1, k2), v in surf.rates.items() if k1 == key1}
        total_rate = functools.reduce(operator.add, exit_rates.values())
        skip_rates = {
            k: v for k, v in exit_rates.items() if frozenset(k) not in edge_keys
        }
        for rate_key, rate in skip_rates.items():
            max_frac = np.nanmax(rate.k_data / total_rate.k_data)
            if max_frac < thresh:
                surf.rates.pop(rate_key)
    return surf


def fit_rates(surf: Surface, tol: float = 0.2) -> Surface:
    """Fit rates to Arrhenius or Plog.

    :param surf: Surface
    :param tol: Threshold for determining pressure dependence
    :return: Surface
    """
    surf = surf.model_copy(deep=True)
    surf.rate_fits = {}
    for rate_key, rate in surf.rates.items():
        if rate.is_pressure_dependent(tol=tol):
            rate_fit = ac.rate.data.PlogRateFit.fit(
                T=rate.T,
                P=rate.P,
                k_data=rate.k_data,
                k_high=rate.k_high,
                order=rate.order,
            )
        else:
            rate_fit = ac.rate.data.ArrheniusRateFit.fit(
                T=rate.T, k=rate.high_pressure_values(), order=rate.order
            )
        surf.rate_fits[rate_key] = rate_fit
    return surf


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
