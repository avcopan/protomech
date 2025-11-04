"""Functions for writing CHEMKIN-formatted files."""

import functools
import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import automol
import polars
from autochem import rate, therm, unit_
from autochem.unit_ import UNITS
from autochem.util import chemkin

from ... import reaction, species
from ..._mech import Mechanism
from ...reaction import ReactionSorted
from ...species import Species, SpeciesTherm
from ...util import c_, pandera_
from ..data import encoder
from .read import KeyWord

ENERGY_PER_SUBSTANCE = unit_.string(UNITS.energy_per_substance).upper().replace(" ", "")
SUBSTANCE = unit_.string(UNITS.substance).upper().replace(" ", "")
REACTION_BLOCK_HEADER = f"   {ENERGY_PER_SUBSTANCE}   {SUBSTANCE}"


def mechanism(
    mech: Mechanism,
    out: str | Path | None = None,
    fill_rates: bool = False,
    elem: bool = True,
    therm: bool = True,
    rxn_data_col_groups: Sequence[Sequence[str]] = (),
    sort_data: bool = False,
    encoder: Callable[[Any], str] = encoder.simple,
) -> str:
    """Write a mechanism to CHEMKIN format.

    :param mech: A mechanism
    :param out: Optionally, write the output to this file path
    :param fill_rates: Whether to fill missing rates with dummy values
    :param elem: Whether to include elements block
    :param therm: Whether to include thermo block
    :return: The CHEMKIN mechanism as a string
    """
    spc_block = species_block(mech)
    rxn_block = reactions_block(
        mech,
        fill_rates=fill_rates,
        data_col_groups=rxn_data_col_groups,
        sort_data=sort_data,
        encoder=encoder,
    )
    blocks = [spc_block, rxn_block]
    if elem:
        blocks.insert(0, elements_block(mech))

    if therm:
        blocks.insert(-2, thermo_block(mech))

    mech_str = "\n\n\n".join(b for b in blocks if b is not None)
    if out is not None:
        out: Path = Path(out)
        out.write_text(mech_str)

    return mech_str


def elements_block(mech: Mechanism) -> str:
    """Write the elements block to a string.

    :param mech: A mechanism
    :return: The elements block string
    """
    fmls = list(map(automol.amchi.formula, mech.species[Species.amchi].to_list()))
    elem_strs = set(itertools.chain(*(f.keys() for f in fmls)))
    elem_strs = automol.form.sorted_symbols(elem_strs)
    return block(KeyWord.ELEMENTS, elem_strs)


def species_block(mech: Mechanism) -> str:
    """Write the species block to a string.

    :param mech: A mechanism
    :return: The species block string
    """
    if mech.species.is_empty():
        return block(KeyWord.SPECIES, [])

    name_width = 1 + mech.species[Species.name].str.len_chars().max()
    smi_width = 1 + mech.species[Species.smiles].str.len_chars().max()
    spc_strs = [
        f"{n:<{name_width}} ! SMILES: {s:<{smi_width}} AMChI: {c}"
        for n, s, c in mech.species.select(
            Species.name, Species.smiles, Species.amchi
        ).rows()
    ]
    return block(KeyWord.SPECIES, spc_strs)


def thermo_block(mech: Mechanism) -> str:
    """Write the thermo block to a string.

    :param mech: A mechanism
    :return: The thermo block string
    """
    if (SpeciesTherm.therm not in mech.species) or mech.species.is_empty():
        return None

    spc_df = mech.species

    # Add species therm objects
    obj_col = c_.temp()
    spc_df = species.with_therm_objects(spc_df, obj_col)

    # Add Chemkin thermo strings
    ck_col = c_.temp()
    tmin_col, tmid_col, tmax_col = c_.temp(), c_.temp(), c_.temp()
    spc_df = spc_df.with_columns(
        polars.col(obj_col)
        .map_elements(therm.chemkin_string, return_dtype=polars.String)
        .alias(ck_col),
        polars.col(obj_col)
        .map_elements(therm.temperature_minimum, return_dtype=polars.Float64)
        .alias(tmin_col),
        polars.col(obj_col)
        .map_elements(therm.temperature_middle, return_dtype=polars.Float64)
        .alias(tmid_col),
        polars.col(obj_col)
        .map_elements(therm.temperature_maximum, return_dtype=polars.Float64)
        .alias(tmax_col),
    )

    # Generate the header
    T_min = spc_df.get_column(tmin_col).max()
    T_mid = spc_df.get_column(tmid_col).mode().item(0)
    T_max = spc_df.get_column(tmax_col).min()
    header = f"ALL\n    {T_min:.3f}  {T_mid:.3f}  {T_max:.3f}"

    # Generate the thermo strings
    therm_strs = spc_df.select(ck_col).to_series()

    # Write the block
    return block(KeyWord.THERM, therm_strs, header=header)


def reactions_block(
    mech: Mechanism,
    frame: bool = True,
    fill_rates: bool = False,
    comment_sep: str = "!",
    data_col_groups: Sequence[Sequence[str]] = (),
    sort_data: bool = False,
    encoder: Callable[[Any], str] = encoder.simple,
) -> str:
    """Write the reactions block to a string.

    :param mech: A mechanism
    :param frame: Whether to frame the block with its header and footer
    :param fill_rates: Whether to fill missing rates with dummy values
    :param col_groups: Column groups to sort by
    :param comment_sep: Comment separator
    :return: The reactions block string
    """
    rxn_df = mech.reactions

    # Quit if no reactions
    if rxn_df.is_empty():
        return block(KeyWord.REACTIONS, "", header=REACTION_BLOCK_HEADER, frame=frame)

    # Identify duplicates
    dup_col = c_.temp()
    rxn_df = reaction.with_duplicate_column(rxn_df, dup_col)

    # Add reaction rate objects
    obj_col = c_.temp()
    rxn_df = reaction.with_rate_object_column(rxn_df, obj_col, fill=fill_rates)

    # Add reaction equations to determine apppropriate width
    eq_col = c_.temp()
    rxn_df = rxn_df.with_columns(
        polars.col(obj_col)
        .map_elements(rate.chemkin_equation, return_dtype=polars.String)
        .alias(eq_col)
    )
    eq_width = rxn_df.get_column(eq_col).str.len_chars().max()
    assert isinstance(eq_width, int), f"eq_width = {eq_width}"
    eq_width += 8

    # Add Chemkin rate strings
    ck_col = c_.temp()
    chemkin_string_ = functools.partial(rate.chemkin_string, eq_width=eq_width)
    rxn_df = rxn_df.with_columns(
        polars.col(obj_col)
        .map_elements(chemkin_string_, return_dtype=polars.String)
        .alias(ck_col)
    )

    # Add duplicate keywords
    rxn_df = rxn_df.with_columns(
        polars.when(dup_col)
        .then(
            polars.col(ck_col).map_elements(
                chemkin.write_with_dup, return_dtype=polars.String
            )
        )
        .otherwise(polars.col(ck_col))
    )

    if data_col_groups:
        # Add comments column
        comment_col = c_.temp()
        rxn_df = rxn_df.with_columns(
            polars.concat_str(
                [
                    polars.struct(cols).map_elements(
                        encoder, return_dtype=polars.String
                    )
                    for cols in data_col_groups
                ],
                separator="\n",
            ).alias(comment_col)
        )

        # Determine Chemkin entry column width
        # (More complex expression required to handle multilines strings)
        ck_width = (
            rxn_df.get_column(ck_col).str.split("\n").list.get(0).str.len_chars().max()
        )
        assert isinstance(ck_width, int), f"ck_width = {ck_width}"
        ck_width += 8

        # Join comment column with
        rxn_df = rxn_df.with_columns(
            polars.concat_arr([ck_col, comment_col])
            .map_elements(
                lambda x: text_with_comments(
                    x[0], x[1], sep=comment_sep, text_width=ck_width
                ),
                return_dtype=polars.String,
            )
            .alias(ck_col)
        )

        # Sort by column values
        if sort_data:
            data_col_groups = list(itertools.chain.from_iterable(data_col_groups))
            rxn_df = rxn_df.sort(data_col_groups, nulls_last=True)

    rxn_strs = rxn_df.get_column(ck_col).to_list()
    return block(KeyWord.REACTIONS, rxn_strs, header=REACTION_BLOCK_HEADER, frame=frame)


def block(key, val, header: str | None = None, frame: bool = True) -> str:
    """Write a block to a string.

    :param key: The starting key for the block
    :param val: The block value(s)
    :param header: A header for the block
    :param frame: Whether to frame the block with its header and footer
    :return: The block
    """
    start = key if header is None else f"{key} {header}"
    val = val if isinstance(val, str) else "\n".join(val)
    if not frame:
        return val

    if not val:
        return "\n\n".join([start, KeyWord.END])

    components = [start, val, KeyWord.END] if val else [start, KeyWord.END]
    return "\n\n".join(components)


def text_with_comments(
    text: str, comments: str, sep: str = "!", text_width: int | None = None
) -> str:
    """Write text with comments to a combined string.

    :param text: Text
    :param comments: Comments
    :return: Combined text and comments
    """
    text_lines = text.splitlines()
    comm_lines = comments.strip().splitlines()
    text_width = max(map(len, text_lines)) + 2 if text_width is None else text_width

    lines = [
        f"{t:<{text_width}} {sep} {c}" if c else t
        for t, c in itertools.zip_longest(text_lines, comm_lines, fillvalue="")
    ]
    return "\n".join(lines)
