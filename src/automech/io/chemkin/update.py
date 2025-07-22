"""Functions for updating mechanisms from CHEMKIN-formatted files."""

from collections.abc import Sequence

import polars

from ... import reaction
from ..._mech import Mechanism
from ...util.io_ import TextInput
from . import read


def thermo(mech: Mechanism, inp_: TextInput | Sequence[TextInput]) -> Mechanism:
    """Update thermochemical data in mechanism.

    :param mech: Mechanism
    :param inp_: ChemKin file(s) or string(s)
    :return: Mechanism
    """
    inp_ = [inp_] if isinstance(inp_, TextInput) else inp_

    mech = mech.model_copy()
    for inp in inp_:
        mech.species = read.thermo(inp, spc_df=mech.species)
    return mech


def rates(mech: Mechanism, inp_: TextInput | Sequence[TextInput]) -> Mechanism:
    """Update rate data in mechanism.

    :param mech: Mechanism
    :param inp_: ChemKin file(s) or string(s)
    :return: Mechanism
    """
    inp_ = [inp_] if isinstance(inp_, TextInput) else inp_

    mech = mech.model_copy()
    rxn_dfs = [read.reactions(inp, spc_df=mech.species) for inp in inp_]
    mech.reactions = reaction.update(
        mech.reactions, polars.concat(rxn_dfs, how="vertical_relaxed")
    )
    return mech
