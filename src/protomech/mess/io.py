"""Read MESS file format."""

import itertools
import re
from pathlib import Path
from typing import Literal

import more_itertools as mit
import pyparsing as pp
from pydantic import BaseModel
from pyparsing import pyparsing_common as ppc

COMMENT = pp.Literal("!") + pp.rest_of_line()
END_KEY = pp.Keyword("End") | pp.Keyword("Dummy")
END_FILE = END_KEY + pp.StringEnd()

BLOCK_TYPES = ("Well", "Bimolecular", "Barrier")
BLOCK_START_REGEX = re.compile(r"^(?:Well|Bimolecular|Barrier)\s")
END_LINE_REGEX = re.compile(r"^End")


def parse_header(mess_inp: str | Path) -> str:
    """Read header.

    :param mess_inp: MESS input
    :return: Headers
    """
    header, *_ = split_input(mess_inp)
    return header


def split_input(mess_inp: str | Path) -> tuple[str, str, str]:
    """Split input into header, body, and footer.

    :param mess_inp: MESS input
    :return: Header, body, footer
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    all_lines = mess_inp.splitlines()
    body_start = next(
        (i for i, line in enumerate(all_lines) if BLOCK_START_REGEX.match(line)), -1
    )
    last_end_index = max(
        *(i for i, line in enumerate(all_lines) if END_LINE_REGEX.match(line)), -1
    )
    header = "\n".join(all_lines[:body_start])
    body = "\n".join(all_lines[body_start:last_end_index])
    footer = "\n".join(all_lines[last_end_index:])
    return header, body, footer


class MessBlockParseData(BaseModel):
    """Parsed MESS block data."""

    type: Literal["Well", "Bimolecular", "Barrier"]
    header: str
    energy: float
    energy_unit: str
    body: str


def parse_blocks(mess_inp: str | Path) -> list[MessBlockParseData]:
    """Read well/bimol/barrier blocks.

    :param mess_inp: MESS input
    :return: A list of type, contents tuples for each block
    """
    _, body, *_ = split_input(mess_inp)
    all_lines = body.splitlines()
    blocks = [
        "\n".join(lines)
        for lines in mit.split_before(all_lines, BLOCK_START_REGEX.match)
    ]

    block_data_lst = []
    for block in blocks:
        block_data = _parse_block(block)
        block_data_lst.append(block_data)

    return block_data_lst


# Helpers
class Key:
    """Keys for parsed data."""

    unit = "unit"
    energy = "energy"
    header = "header"


UNIT = pp.Suppress("[") + pp.Word(pp.alphas + "/")(Key.unit) + pp.Suppress("]")
ZERO_ENERGY = pp.Literal("ZeroEnergy") + UNIT + ppc.number(Key.energy)
GROUND_ENERGY = pp.Literal("GroundEnergy") + UNIT + ppc.number(Key.energy)


def _parse_block(block: str) -> MessBlockParseData:
    """Read an individual block.

    :param type: Block type
    :param contents: Block contents
    :return: Block
    """
    header, *body_lines = block.splitlines()
    type, header = header.split(maxsplit=1)
    header, *_ = header.split("!", maxsplit=1)
    if type not in BLOCK_TYPES:
        msg = f"Parsed block type as {type} which is not in {BLOCK_TYPES}."
        raise ValueError(msg)

    body = "\n".join(body_lines)

    # Parse energy
    expr = ... + (
        GROUND_ENERGY
        if (type == "Bimolecular" and "Dummy" not in body)
        else ZERO_ENERGY
    )
    res = expr.parse_string(body)
    energy = res.get(Key.energy)
    unit = res.get(Key.unit)
    if not isinstance(energy, float):
        msg = f"Unable to parse energy:{body}"
        raise ValueError(msg)

    if not isinstance(unit, str):
        msg = f"Unable to parse unit:{body}"
        raise ValueError(msg)

    return MessBlockParseData(
        type=type,
        header=header.strip(),
        energy=energy,
        energy_unit=unit,
        body=body,
    )
