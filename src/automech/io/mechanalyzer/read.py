"""Functions for reading Mechanalyzer-formatted files."""

import io
import os
from pathlib import Path

import pandas
import polars

from ... import species as m_species
from ..._mech import Mechanism
from ...util import df_
from ..chemkin import read as chemkin_read


def mechanism(
    rxn_inp: str,
    spc_inp: pandas.DataFrame | str | Path,
    rxn_out: str | None = None,
    spc_out: str | None = None,
) -> Mechanism:
    """Extract the mechanism from MechAnalyzer files.

    :param rxn_inp: A mechanism (CHEMKIN format), as a file path or string
    :param spc_inp: A Mechanalyzer species table, as a file path or string or dataframe
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(spc_inp, out=spc_out)
    spc_df = chemkin_read.thermo(rxn_inp, spc_df=spc_df)
    rxn_df = chemkin_read.reactions(rxn_inp, out=rxn_out, spc_df=spc_df)
    return Mechanism(reactions=rxn_df, species=spc_df)


def species(
    inp: pandas.DataFrame | str | Path, out: str | None = None
) -> polars.DataFrame:
    """Extract species information as a dataframe from a Mechanalyzer species CSV.

    :param inp: A Mechanalyzer species CSV, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    if isinstance(inp, str | Path):
        inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)
        spc_df = polars.read_csv(io.StringIO(inp), quote_char="'")
    else:
        assert isinstance(inp, pandas.DataFrame), f"Invalid species input: {inp}"
        spc_df = polars.from_pandas(inp)

    spc_df = m_species.bootstrap(spc_df)
    df_.to_csv(spc_df, out)

    return spc_df
