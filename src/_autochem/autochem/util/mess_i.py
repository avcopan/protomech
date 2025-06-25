"""Read MESS file format."""

import re
from pathlib import Path
from typing import Literal

import pyparsing as pp
from pydantic import BaseModel
from pyparsing import pyparsing_common as ppc

COMMENT = pp.Literal("!") + pp.rest_of_line()
END_FILE = pp.Keyword("End") + pp.StringEnd()

BLOCK_START = pp.Keyword("Well") | pp.Keyword("Bimolecular") | pp.Keyword("Barrier")
BLOCK_END = pp.Keyword("End") + pp.FollowedBy(BLOCK_START | END_FILE)
FILE_HEADER = pp.SkipTo(BLOCK_START)


def parse_header(mess_inp: str | Path) -> str | None:
    """Read header.

    :param mess_inp: MESS input
    :return: Headers
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    expr = FILE_HEADER(Key.header)
    return expr.parse_string(mess_inp).get(Key.header)


class MessBlockParseData(BaseModel):
    """Parsed MESS block data."""

    type: Literal["Well", "Bimolecular", "Barrier"]
    label: str
    energy: float
    energy_unit: str
    body: str


def parse_blocks(mess_inp: str | Path) -> list[MessBlockParseData]:
    """Read well/bimol/barrier blocks.

    :param mess_inp: MESS input
    :return: A list of type, contents tuples for each block
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    block_expr = BLOCK_START("type") + pp.SkipTo(BLOCK_END)("contents") + BLOCK_END

    expr = pp.Suppress(FILE_HEADER) + pp.OneOrMore(pp.Group(block_expr))
    expr.ignore(COMMENT)

    block_data_lst = []
    for res in expr.parse_string(mess_inp):
        block_data = _parse_block(res.get("type"), res.get("contents"))
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


def _parse_block(
    type: Literal["Well", "Bimolecular", "Barrier"], contents: str
) -> MessBlockParseData:
    """Read an individual block.

    :param type: Block type
    :param contents: Block contents
    :return: Block
    """
    label_line, *body_lines = contents.splitlines()
    label, *_ = re.split("!|\n", label_line, maxsplit=1)
    body = "\n".join(body_lines)
    # Parse energy
    expr = ... + (GROUND_ENERGY if type == "Bimolecular" else ZERO_ENERGY)
    res = expr.parse_string(contents)

    return MessBlockParseData(
        type=type,
        label=label.strip(),
        energy=res.get(Key.energy),
        energy_unit=res.get(Key.unit),
        body=body,
    )
