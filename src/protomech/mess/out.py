import re
from pathlib import Path

import autochem as ac
import more_itertools as mit
import pyparsing as pp
from pydantic import BaseModel


class Key:
    wells = "wells"
    bimols = "bimols"
    eq = "eq"


class MessRateParseData(BaseModel):
    """Parsed MESS block data."""

    reactant: str
    product: str
    temperatures: list[float]
    pressures: list[float]
    k_data: list[list[float]]
    k_high: list[float]


def rate_data_parse_results(
    mess_out: str | Path,
) -> list[ac.util.mess.MessOutputChannelParseResults]:
    """Parse rate data.

    :param mess_out: MESS output
    :return: Rate data
    """
    trans_dct = translation_table(mess_out)
    blocks = rate_blocks(mess_out)
    rates = []
    for block in blocks:
        res = ac.util.mess.parse_output_channel(block)
        if res.id1 and res.id2:
            res.id1 = trans_dct[res.id1]
            res.id2 = trans_dct[res.id2]
            rates.append(res)
    return rates


# Helpers
def rate_blocks(mess_out: str | Path) -> list[str]:
    """Parse rate table blocks from MESS output.

    :param mess_out: MESS output
    :return: Rate table blocks
    """
    mess_str = mess_out.read_text() if isinstance(mess_out, Path) else mess_out
    section_regex = re.compile(
        "(?<=Temperature-Pressure Rate Tables:).*", flags=re.DOTALL
    )
    match = section_regex.search(mess_str)
    if not match:
        msg = "No Temperature-Pressure Rate Tables section found."
        raise ValueError(msg)

    section = match.group(0).strip()
    lines = section.splitlines()
    return list(map("\n".join, mit.split_before(lines, lambda s: "->" in s)))


def translation_table(mess_out: str | Path) -> dict[str, str]:
    """Parse translation table from MESS output.

    :param mess_out: MESS output
    :return: Translation table
    """
    mess_str = mess_out.read_text() if isinstance(mess_out, Path) else mess_out
    header = header_expression("Names Translation Tables")
    wells = translation_table_section_expression("Wells:")
    bimols = translation_table_section_expression("Bimolecular products:")
    result = (
        (
            pp.SkipTo(header, include=True)
            + pp.Opt(wells)(Key.wells)
            + pp.Opt(bimols)(Key.bimols)
        )
        .parse_string(mess_str)
        .as_dict()
    )
    translations = result.get(Key.wells, []) + result.get(Key.bimols, [])
    return dict(translations)


def translation_table_section_expression(header: str) -> pp.ParserElement:
    """Expression for translation table secion.

    :param header: Section header
    :return: Parser element
    """
    translation = pp.Word(pp.alphanums) + pp.Suppress("-") + pp.Word(pp.printables)
    return header_expression(header) + pp.OneOrMore(pp.Group(translation))


def header_expression(header: str, suppress: bool = True) -> pp.ParserElement:
    """Expression for header

    :param header: Section header
    :param suppress: Whether to suppress this parser element
    :return: Parser element
    """
    expr = pp.Literal(header) + pp.rest_of_line()
    return pp.Suppress(expr) if suppress else expr
