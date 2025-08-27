import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import pydantic

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
        return f"Well  {self.label}\n{self.mess_body}\nEnd"


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
            return f"Well  {self.label}\n{self.mess_body}\nEnd"

        return f"Bimol  {self.label}\n{self.mess_body}\nEnd"


class Edge(Feature):
    key: Annotated[frozenset, pydantic.BeforeValidator(frozenset)] = pydantic.Field(
        min_length=2, max_length=2
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
        return (
            f"Barrier  {self.label} {well_label1} {well_label2}\n{self.mess_body}\nEnd"
        )


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
def fake_well_keys(surf: Surface) -> list[int]:
    """Keys of fake wells in surface.

    :param surf: Surface
    :return: Keys of fake wells
    """
    return [n.key for n in surf.nodes if n.fake]


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
    key_dct = {d.label: i for i, d in enumerate(node_block_data)}
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
    return "\n!\n".join([mess_header, *node_blocks, *edge_blocks])  # type: ignore


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

    assert block_data.label in key_dct, f"{block_data.label} not in {key_dct}"
    key = key_dct[block_data.label]
    energy = block_data.energy
    block_body = block_data.body

    if block_data.type == "Bimolecular":
        names = block_data.label.split("+")
        return NmolNode(
            key=key,
            energy=energy,
            names=names,
            interacting=False,
            fake=False,
            mess_body=block_body,
        )

    if block_data.label.startswith("FakeW-"):
        names = block_data.label.removeprefix("FakeW-").split("+")
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
        key=key, energy=energy, name=block_data.label, mess_body=block_body
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

    name, *well_labels = block_data.label.split()
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
