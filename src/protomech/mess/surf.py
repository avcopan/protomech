import itertools
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Annotated, Literal, Self

import more_itertools as mit
import networkx as nx
import pydantic

import automech
from automech import Mechanism
from automech.reaction import Reaction

from . import io


class Feature(pydantic.BaseModel, ABC):
    energy: float
    fake: bool = False
    mess_body: str | None = None

    @property
    @abstractmethod
    def label(self) -> str:
        """Label."""
        pass

    @property
    @abstractmethod
    def mess_block(self) -> str | None:
        """MESS block."""
        pass


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

    @property
    def mess_block(self) -> str | None:
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
            label = f"Fake({label})"

        return label

    @property
    def names_list(self) -> list[str]:
        """Label."""
        return self.names

    @property
    def mess_block(self) -> str | None:
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
    well_labels: tuple[str, str] | None = None

    @property
    def label(self) -> str:
        """Label."""
        return self.name

    @property
    def mess_block(self) -> str | None:
        """MESS block."""
        assert self.well_labels is not None, self
        well_label1, well_label2 = self.well_labels
        return f"Barrier  {self.label} {well_label1} {well_label2}\n{self.mess_body}"


class Surface(pydantic.BaseModel):
    """Potential energy surface."""

    nodes: list[Node]
    edges: list[Edge]

    mess_header: str | None = None

    @pydantic.model_validator(mode="after")
    def _validate_keys(self):
        # Validate node keys
        nkeys = [n.key for n in self.nodes]
        if not len(nkeys) == len(set(nkeys)):
            raise ValueError(f"Non-unique node keys: {nkeys}")

        # Validate edge keys
        ekeys = [e.key for e in self.edges]
        if not set(itertools.chain.from_iterable(ekeys)) == set(nkeys):
            raise ValueError(f"Edge keys {ekeys} do not match node keys {nkeys}")

        return self

    @pydantic.model_validator(mode="after")
    def _update_edges(self):
        # Update edges with well labels
        label_dct = {n.key: n.label for n in self.nodes}
        edges = []
        for edge0 in self.edges:
            edge = edge0.model_copy()
            key1, key2 = sorted(edge.key)
            edge.well_labels = (label_dct[key1], label_dct[key2])
            edges.append(edge)

        self.edges = edges
        return self


# Properties
def node_keys(surf: Surface) -> list[int]:
    """Node keys for the surface."""
    return [n.key for n in surf.nodes]


def fake_well_keys(surf: Surface) -> list[int]:
    """Keys of fake wells in surface.

    :param surf: Surface
    :return: Keys of fake wells
    """
    return [n.key for n in surf.nodes if n.fake]


def node_object(surf: Surface, key: int) -> Node:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    node = next((n for n in surf.nodes if n.key == key), None)
    if node is None:
        msg = f"Key {key} is not associated with a node:\n{surf.model_dump()}"
        raise ValueError(msg)
    return node


def edge_object(surf: Surface, key: Collection[int]) -> Edge:
    """Look up node object by key

    :param surf: Surface
    :param key: Key
    :return: Node
    """
    key = key if isinstance(key, frozenset) else frozenset(key)
    edge = next((e for e in surf.edges if e.key == key), None)
    if edge is None:
        msg = f"Key {key} is not associated with an edge:\n{surf.model_dump()}"
        raise ValueError(msg)
    return edge


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


def node_key(surf: Surface, names: list[str], fake: bool = False) -> int | None:
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


def remove_nodes(surf: Surface, keys: list[int]) -> Surface:
    """Remove nodes from surface, along with their associated edges.

    :param surf: Surface
    :param keys: Keys of nodes to remove
    :return: Surface
    """
    nodes = [n for n in surf.nodes if n.key not in keys]
    edges = [e for e in surf.edges if not e.key & set(keys)]
    return Surface(nodes=nodes, edges=edges, mess_header=surf.mess_header)


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
    return Surface(nodes=nodes, edges=edges, mess_header=surf.mess_header)


def merge_prompt_resonant_instabilities(surf: Surface, mech: Mechanism) -> Surface:
    """Merge prompt resonant instabilities on a surface.

    That is, resonant instabilities for a lower stoichiometry than the rest of
    the surface. Example:

        A -> B + C*
        C* -(unstable)-> Y + Z

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
        rct_key = node_key(surf, [rct_name])
        prd_key = node_key(surf, prd_names)
        if rct_key is not None and prd_key is not None:
            instab_path_dct[rct_name] = shortest_path(surf, rct_key, prd_key)

    for instab_name, instab_path in instab_path_dct.items():
        rct_key, *_, prd_key = instab_path
        # Iterate over n-molecular nodes containing the unstable spacies
        nmol_keys = node_keys_containing(surf, instab_name)
        for nmol_key in nmol_keys:
            # Iterate over neighbors of these nodes, skipping fake wells
            # These are the neighbors we want to connect to
            conn_keys = node_neighbors(surf, nmol_key, skip_fake=True)
            for conn_key in conn_keys:
                # Get the path from the connection node to the n-molecular node
                conn_node = node_object(surf, conn_key)
                conn_path = shortest_path(surf, conn_key, nmol_key)
                # Update n-molecular node to give unstable products
                new_key = max(node_keys(surf)) + 1
                new_node = instability_product_node(
                    surf, nmol_key, rct_key, prd_key, new_key=new_key
                )
                new_edge_key = [conn_key, new_key]
                new_edge_labels = [conn_node.label, new_node.label]
                new_edge = edge_object(surf, conn_path[:2]).model_copy()
                new_edge.key = frozenset(new_edge_key)
                new_edge = edge_set_labels(new_edge, new_edge_key, new_edge_labels)
                # Remove n-molecular node and fake well
                surf = remove_nodes(surf, conn_path[1:])
                surf = extend(surf, nodes=[new_node], edges=[new_edge])

        surf = remove_nodes(surf, instab_path)

    return surf


# Create nodes
def instability_product_node(
    surf: Surface, key: int, rct_key: int, prd_key: int, new_key: int | None = None
) -> NmolNode:
    """Generate a new node describing the product of an unstable one.

    :param surf: Surface
    :param key: Node key
    :param rct_key: Instability reactant key
    :param prd_key: Instability product key
    :return: Node
    """
    new_key = key if new_key is None else new_key
    node0 = node_object(surf, key)
    rct_node = node_object(surf, rct_key)
    prd_node = node_object(surf, prd_key)

    if not isinstance(rct_node, UnimolNode):
        msg = f"Instability reactant must be a unimolecular node: {rct_node}"
        raise ValueError(msg)

    if isinstance(node0, NmolNode):
        (rct_name,) = rct_node.names_list
        prd_names = prd_node.names_list
        prd_energy = prd_node.energy - rct_node.energy

        node = node0.model_copy(deep=True)
        node.names.remove(rct_name)
        node.names = sorted([*node.names, *prd_names])
        node.energy += prd_energy
        node.mess_body = f"  ! ZeroEnergy[kcal/mol]      {node.energy:.2f}\n  Dummy"
        node.key = new_key

    else:
        msg = "Not yet implemented"
        raise NotImplementedError(msg)

    return node


def edge_set_labels(
    edge: Edge, key: Collection[int], well_labels: Sequence[str]
) -> Edge:
    """Set labels for an edge.

    :param edge: Edge
    :param key: Edge key, as sorted or unsorted pair of node keys
    :param well_labels: Labels in the order of key (if Sequence) or in the
        sorted order of key (if unordered Collection)
    :return: Edge
    """
    key = key if isinstance(int, Sequence) else sorted(key)
    if not len(key) == len(well_labels) == 2:
        msg = f"The key and labels for an edge must have length 2:\nkey={key}\nlabels={well_labels}"
        raise ValueError(msg)

    _, labels = zip(*sorted(zip(key, well_labels, strict=True)))
    edge.well_labels = tuple(labels)
    return edge


# N-ary operations
def combine(surfs: Sequence[Surface]) -> Surface:
    """Combine multiple surfaces.

    :param surfs: Surfaces
    :return: Combined surface
    """
    mess_header = surfs[0].mess_header

    key_dct = {}
    all_nodes = itertools.chain.from_iterable(surf.nodes for surf in surfs)
    nodes = []
    for key, node in enumerate(mit.unique_everseen(all_nodes, key=lambda n: n.label)):
        nodes.append(node.model_copy(update={"key": key}, deep=True))
        key_dct[node.label] = key

    all_edges = itertools.chain.from_iterable(surf.edges for surf in surfs)
    edges = []
    for edge in mit.unique_everseen(all_edges, key=lambda e: e.label):
        edges.append(
            edge.model_copy(
                update={"key": frozenset(map(key_dct.get, edge.well_labels))}
            )
        )

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
    header = io.parse_header(mess_inp)

    # Parse blocks and separate node and edge data
    block_data = io.parse_blocks(mess_inp)
    node_block_data = [d for d in block_data if d.type in ("Well", "Bimolecular")]
    edge_block_data = [d for d in block_data if d.type == "Barrier"]

    # Instantiate nodes and edges
    key_dct = {d.header: i for i, d in enumerate(node_block_data)}
    nodes = [node_from_mess_block_parse_data(d, key_dct) for d in node_block_data]
    edges = [edge_from_mess_block_parse_data(d, key_dct) for d in edge_block_data]

    return Surface(nodes=nodes, edges=edges, mess_header=header)


def mess_input(surf: Surface) -> str:
    """MESS input for surface.

    :param surf: Surface
    :return: MESS input
    """
    mess_header = surf.mess_header
    node_blocks = [n.mess_block for n in surf.nodes]
    edge_blocks = [e.mess_block for e in surf.edges]
    assert isinstance(mess_header, str)
    assert all(isinstance(b, str) for b in node_blocks)
    assert all(isinstance(b, str) for b in edge_blocks)
    return "\n!\n".join([mess_header, *node_blocks, *edge_blocks, "End", ""])  # type: ignore


# Helpers
def node_from_mess_block_parse_data(
    block_data: io.MessBlockParseData, key_dct: dict[str, int]
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
    block_data: io.MessBlockParseData, key_dct: dict[str, int]
) -> Edge:
    """Generate Barrier object from block.

    :param block_data: Parsed MESS block data
    :param key_dct: Dictionary mapping labels to IDs
    :return: Barrier
    """
    if not block_data.type == "Barrier":
        raise ValueError("Cannot create barrier object from non-barrier block.")

    name, *well_labels = block_data.header.split()
    fake = name.startswith("FakeB-")
    assert all(label in key_dct for label in well_labels), (
        f"{well_labels} not in {key_dct}"
    )
    barrierless = "PhaseSpaceTheory" in block_data.body
    key = frozenset(map(key_dct.get, well_labels))
    return Edge(
        key=key,
        name=name,
        energy=block_data.energy,
        fake=fake,
        barrierless=barrierless,
        mess_body=block_data.body,
    )
