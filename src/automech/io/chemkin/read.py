"""Functions for reading CHEMKIN-formatted files."""

import itertools
import re

import autochem as ac
import more_itertools as mit
import polars
import pyparsing as pp
from autochem import unit_
from autochem.unit_ import Units
from pyparsing import common as ppc

from ... import reaction
from ... import species as m_species
from ..._mech import Mechanism
from ...reaction import Reaction, ReactionRate
from ...species import Species, SpeciesTherm
from ...util import df_, io_
from ...util.io_ import TextInput, TextOutput


class KeyWord:
    # Blocks
    ELEMENTS = "ELEMENTS"
    THERM = "THERM"
    SPECIES = "SPECIES"
    REACTIONS = "REACTIONS"
    END = "END"
    # Units
    # # Energy (E) units
    CAL_MOLE = "CAL/MOLE"
    KCAL_MOLE = "KCAL/MOLE"
    JOULES_MOLE = "JOULES/MOLE"
    KJOULES_MOLE = "KJOULES/MOLE"
    KELVINS = "KELVINS"
    # # Prefactor (A) units
    MOLES = "MOLES"
    MOLECULES = "MOLECULES"


# generic
COMMENT_REGEX = re.compile(r"!.*$", flags=re.M)
HASH_COMMENT_REGEX = re.compile(r"# .*$", flags=re.M)
COMMENT_START = pp.Suppress(pp.Literal("!"))
COMMENT_END = pp.Suppress(pp.LineEnd())
COMMENT = COMMENT_START + ... + COMMENT_END
COMMENTS = pp.ZeroOrMore(COMMENT)

# units
E_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.CAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KCAL_MOLE)
    ^ pp.CaselessKeyword(KeyWord.JOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KJOULES_MOLE)
    ^ pp.CaselessKeyword(KeyWord.KELVINS)
)
A_UNIT = pp.Opt(
    pp.CaselessKeyword(KeyWord.MOLES) ^ pp.CaselessKeyword(KeyWord.MOLECULES)
)


# mechanism
def mechanism(
    inp: TextInput, out: TextOutput = None, spc_out: TextOutput = None
) -> Mechanism:
    """Extract the mechanism from a CHEMKIN file.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(inp, out=spc_out)
    rxn_df = reactions(inp, out=out, spc_df=spc_df)
    return Mechanism(reactions=rxn_df, species=spc_df)


# reactions
def reactions(
    inp: TextInput, spc_df: polars.DataFrame | None = None, out: TextOutput = None
) -> polars.DataFrame:
    """Extract reaction information as a dataframe from a CHEMKIN file.

    Automatically converts to internal units.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param units: Convert the rates to these units, if needed
    :param spc_df: A species dataframe to be used for validation
    :param out: Optionally, write the output to this file path
    :return: The reactions dataframe, along with any errors that were encountered
    """
    units0 = reactions_units(inp)

    def _is_reaction_line(string: str) -> bool:
        return re.search(r"\d\s*$", string)

    # Do the parsing
    rxn_block_str = reactions_block(inp, comments=False)
    line_iter = itertools.dropwhile(
        lambda s: not _is_reaction_line(s), rxn_block_str.splitlines()
    )
    rxn_strs = list(map("\n".join, mit.split_before(line_iter, _is_reaction_line)))
    rxns = [ac.rate.from_chemkin_string(r, units=units0) for r in rxn_strs]

    data = {
        Reaction.reactants: [r.reactants for r in rxns],
        Reaction.products: [r.products for r in rxns],
        ReactionRate.reversible: [r.reversible for r in rxns],
        ReactionRate.rate: [r.rate.model_dump() for r in rxns],
    }
    rxn_df = reaction.bootstrap(data, spc_df=spc_df)

    df_.to_csv(rxn_df, out)

    return rxn_df


def reactions_block(
    inp: TextInput, comments: bool = True, strip: bool = True
) -> str | None:
    """Get the reactions block, starting with 'REACTIONS' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param comments: Include comments?
    :param strip: Strip spaces from the ends?
    :return: The block
    """
    return block(inp, KeyWord.REACTIONS, comments=comments, strip=strip)


def reactions_units(inp: TextInput) -> Units:
    """Get the units for reaction rate constants.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param default: Return default values, if missing?
    :return: The units for E and A, respectively
    """
    rxn_block_str = reactions_block(inp, comments=False, strip=False)
    assert isinstance(rxn_block_str, str), f"inp = {inp}"

    line1, *_ = rxn_block_str.split("\n", maxsplit=1)
    line1_units = list(map(str.lower, re.split(r"(?<!/)\s+(?!/)", line1)))
    return unit_.system.from_unit_sequence(line1_units)


# species
def species(inp: TextInput, out: TextOutput = None) -> polars.DataFrame:
    """Get the list of species, along with their comments.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: A species dataframe
    """
    species_name = pp.Word(pp.printables)
    word = pp.Word(pp.printables, exclude_chars=":")
    value = pp.Group(word + pp.Suppress(":") + word)
    values = pp.ZeroOrMore(value)
    comment_values = COMMENT_START + values + COMMENT_END
    entry = species_name("name") + comment_values("values")
    parser = pp.Suppress(...) + pp.OneOrMore(pp.Group(entry))

    spc_block_str = species_block(inp, comments=True)

    data = [
        {Species.name: r.get("name"), **dict(r.get("values").as_list())}
        for r in parser.parse_string(spc_block_str)
    ]
    spc_df = m_species.bootstrap(data)
    spc_df = thermo(inp, spc_df=spc_df)

    df_.to_csv(spc_df, out)

    return spc_df


def species_block(inp: TextInput, comments: bool = True) -> str:
    """Get the species block, starting with 'SPECIES' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.SPECIES, comments=comments)


def species_names(inp: TextInput) -> list[str]:
    """Get the list of species.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The species
    """
    parser = pp.OneOrMore(pp.Word(pp.printables))
    spc_block_str = species_block(inp, comments=False)
    return parser.parse_string(spc_block_str).as_list()


# therm
def thermo(
    inp: TextInput, spc_df: polars.DataFrame | None = None, out: TextOutput = None
) -> polars.DataFrame | None:
    """Get thermodynamic data as a dataframe.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param spc_df: Optionally, join this to a species dataframe
    :return: A thermo dataframe
    """
    _, T_mid, _ = mit.padded(thermo_temperatures(inp) or [], fillvalue=None, n=3)
    spc_strs = thermo_entries(inp)
    if spc_strs is None:
        return spc_df

    spcs = [ac.therm.from_chemkin_string(s, T_mid=T_mid) for s in spc_strs]
    data = {
        Species.name: [s.name for s in spcs],
        SpeciesTherm.therm: [s.therm.model_dump() for s in spcs],
    }
    therm_df = polars.DataFrame(data, strict=False)
    therm_df = therm_df.unique(subset=Species.name)
    if spc_df is not None:
        therm_df = m_species.left_update(spc_df, therm_df, key_col_=Species.name)

    df_.to_csv(therm_df, out)

    return therm_df


def thermo_block(inp: TextInput, comments: bool = True) -> str:
    """Get the therm block, starting with 'THERM' and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The block
    """
    return block(inp, KeyWord.THERM, comments=comments)


def thermo_temperatures(inp: TextInput) -> list[float] | None:
    """Get the therm block temperatures.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The temperatures
    """
    therm_block_str = thermo_block(inp, comments=False)
    if therm_block_str is None:
        return None

    parser = therm_temperature_expression()
    temps = parser.parse_string(therm_block_str).as_list()
    return list(map(float, temps))


def thermo_entries(inp: TextInput) -> list[str] | None:
    """Get the therm block entries.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The entries
    """
    therm_block_str = thermo_block(inp, comments=False)
    if therm_block_str is None:
        return None

    parser = pp.Suppress(therm_temperature_expression()) + pp.OneOrMore(
        therm_entry_expression()
    )
    entries = parser.parse_string(therm_block_str).as_list()
    return entries


# generic
def block(
    inp: TextInput, key: str, comments: bool = False, strip: bool = True
) -> str | None:
    """Get a keyword block, starting with a key and ending in 'END'.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :param key: The key that the block starts with
    :param comments: Include comments?
    :param strip: Strip spaces from the ends?
    :return: The block
    """
    inp = io_.read_text(inp)

    pattern = rf"{key[:4]}\S*(.*?){KeyWord.END}"
    match = re.search(pattern, inp, re.M | re.I | re.DOTALL)
    if not match:
        return None

    block_str = match.group(1)
    # Remove comments, if requested
    if not comments:
        block_str = without_comments(block_str)

    return block_str.strip() if strip else block_str


def without_comments(inp: TextInput) -> str:
    """Get a CHEMKIN string or substring with comments removed.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The string, without comments
    """
    inp = io_.read_text(inp)

    inp = re.sub(COMMENT_REGEX, "", inp)
    return re.sub(HASH_COMMENT_REGEX, "", inp)


def all_comments(inp: TextInput) -> list[str]:
    """Get all comments from a CHEMKIN string or substring.

    :param inp: A CHEMKIN mechanism, as a file path or string
    :return: The comments
    """
    inp = io_.read_text(inp)

    return re.findall(COMMENT_REGEX, inp)


def therm_temperature_expression() -> pp.ParseExpression:
    """Generate a pyparsing expression for the therm block temperatures."""
    return pp.Suppress(... + pp.Opt(pp.CaselessKeyword("ALL"))) + pp.Opt(ppc.number * 3)


def therm_entry_expression() -> pp.ParseExpression:
    """Generate a pyparsing expression for a therm entry."""
    return pp.Combine(
        therm_line_expression(1)
        + therm_line_expression(2)
        + therm_line_expression(3)
        + therm_line_expression(4)
    )


def therm_line_expression(num: int) -> pp.ParseExpression:
    """Generate a pyparsing expression for a therm line."""
    num = pp.Literal(f"{num}")
    end = pp.LineEnd()
    return pp.AtLineStart(pp.Combine(pp.SkipTo(num + end, include=True)))
