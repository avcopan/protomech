"""Functions for reading CHEMKIN-formatted files."""

import os

import cantera
import polars


def expand_stereo(inp: str, out: str | None = None) -> polars.DataFrame:
    """Extract reaction information as a dataframe from a Cantera file.

    :param inp: A Cantera mechanism, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The reactions dataframe
    """
    if not os.path.exists(inp):
        raise NotImplementedError("Reading from text not yet implemented!")

    inp = str(inp)

    all_species = cantera.Species.list_from_file(inp)
    ref_phase = cantera.Solution(
        thermo="ideal-gas", kinetics="gas", species=all_species
    )
    all_reactions = cantera.Reaction.list_from_file(inp, ref_phase)
    print(all_species)
    print(all_reactions)

    # gas = cantera.Solution(inp)
    # gas.set_multiplier(2.0, 0)
    # gas.modify_reaction()
    # for rxn in gas.reactions():
    #     print(
    #         rxn.reactants,
    #         rxn.products,
    #         rxn.rate.input_data,
    #         rxn.equation,
    #         rxn.third_body,
    #     )
