"""Functions for reading RMG-formatted files."""

import os
from pathlib import Path

import automol
import polars
import pyparsing as pp
from automol.graph import RMG_ADJACENCY_LIST
from pyparsing import pyparsing_common as ppc
from tqdm.auto import tqdm

from ... import species as m_species
from ..._mech import Mechanism
from ...species import Species
from ...util import df_, io_
from ..chemkin import read as chemkin_read

MULTIPLICITY = pp.CaselessLiteral("multiplicity") + ppc.integer("mult")
SPECIES_NAME = pp.Word(pp.printables)
SPECIES_ENTRY = (
    SPECIES_NAME("species") + pp.Opt(MULTIPLICITY) + RMG_ADJACENCY_LIST("adj_list")
)
SPECIES_DICT = pp.OneOrMore(pp.Group(SPECIES_ENTRY))("dict")


def mechanism(
    rxn_inp: io_.TextInput,
    spc_inp: io_.TextInput,
    out: io_.TextOutput = None,
    spc_out: io_.TextOutput = None,
) -> Mechanism:
    """Extract the mechanism from RMG files.

    :param inp: An RMG mechanism (CHEMKIN format), as a file path or string
    :param spc_inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path (reactions)
    :param spc_out: Optionally, write the output to this file path (species)
    :return: The mechanism dataclass
    """
    spc_df = species(spc_inp)
    spc_df = chemkin_read.thermo(rxn_inp, spc_df=spc_df, out=spc_out)
    rxn_df = chemkin_read.reactions(rxn_inp, out=out, spc_df=spc_df)
    return Mechanism(reactions=rxn_df, species=spc_df)


def species(inp: io_.TextInput, out: str | None = None) -> polars.DataFrame:
    """Extract species information as a dataframe from an RMG species dictionary.

    :param inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    inp = io_.read_text(inp)

    spc_par_rets = SPECIES_DICT.parse_string(inp, parse_all=True).asDict()["dict"]

    names = []
    mults = []
    smis = []
    chis = []
    for spc_par_ret in tqdm(spc_par_rets):
        adj_par_ret = spc_par_ret["adj_list"]
        gra = automol.graph.from_parsed_rmg_adjacency_list(adj_par_ret)

        names.append(spc_par_ret["species"])
        mults.append(spc_par_ret.get("mult", 1) - 1)
        chis.append(automol.graph.amchi(gra))
        smis.append(automol.graph.smiles(gra))

    data = {
        Species.name: names,
        Species.spin: mults,
        Species.amchi: chis,
        Species.smiles: smis,
    }
    spc_df = m_species.bootstrap(data)  # pyright: ignore[reportArgumentType]
    df_.to_csv(spc_df, out)

    return spc_df
