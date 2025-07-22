"""Test automech functions."""

from pathlib import Path

import pytest
from automol.graph import enum

import automech
from automech.reaction import ReactionSorted
from automech.species import Species
from automech.util import df_

DATA_PATH = Path(__file__).parent / "data"

MECH_EMPTY = automech.from_smiles(spc_smis=[], rxn_smis=[])
MECH_NO_REACIONS = automech.from_smiles(
    spc_smis=["CC=CC"], rxn_smis=[], name_dct={"CC=CC": "C4e2"}
)
MECH_PROPANE = automech.from_smiles(
    rxn_smis=["CCC.[OH]>>CC[CH2].O"],
    name_dct={"CCC": "C3", "[OH]": "OH", "CC[CH2]": "C3y1"},
)
MECH_BUTENE = automech.from_smiles(
    rxn_smis=[
        "CC=CC.[OH]>>CC=C[CH2].O",
        "CC=CC.[OH]>>C[CH]C(O)C",
        "CC=CC.[O]>>CC1C(O1)C",
    ],
    name_dct={"CC=CC": "C4e2", "CC=C[CH2]": "C4e2y1", "CC1C(O1)C": "C4x23"},
)
MECH_BUTENE_NO_REACTIONS = automech.without_reactions(MECH_BUTENE)
MECH_BUTENE_SUBSET = automech.from_smiles(
    rxn_smis=[
        "O.CC=C[CH2]>>[OH].CC=CC",
    ],
    name_dct={
        "CC=CC": "C4e2",
        "CC=C[CH2]": "C4e2y1",
    },
)
MECH_BUTENE_ALTERNATIVE_NAMES = automech.from_smiles(
    spc_smis=["CC=CC", "CC=C[CH2]", "O", "[OH]"],
    name_dct={
        "CC=CC": "2-butene",
        "CC=C[CH2]": "1-methylallyl",
        "O": "water",
        "[OH]": "hydroxyl",
    },
)
MECH_BUTENE_WITH_EXCLUDED_REACTIONS = automech.from_smiles(
    rxn_smis=[
        "CC=CC.[OH]>>CC=C[CH2].O",
        "CC=CC.[OH]>>C[CH]C(O)C",
        "CC=CC.[O]>>CC1C(O1)C",
        "[OH].[OH]>>OO",
    ],
    name_dct={"CC=CC": "C4e2", "CC=C[CH2]": "C4e2y1", "CC1C(O1)C": "C4x23"},
)
MECH_ETHANE = automech.from_smiles(
    rxn_smis=[
        "CC.[OH]>>C[CH2].O",
        "C[CH2].[O][O]>>CCO[O]",
        "C[CH2]>>C=C.[H]",
        "CCO[O]>>C=C.O[O]",
        "CCO[O]>>[CH2]COO",
        "CCO[O]>>C[CH]OO",
    ]
)


@pytest.mark.parametrize(
    "mech0",
    [
        MECH_EMPTY,
        MECH_NO_REACIONS,
        MECH_PROPANE,
        MECH_BUTENE,
        MECH_BUTENE_WITH_EXCLUDED_REACTIONS,
    ],
)
def test__network(mech0):
    """Test automech.network."""
    print(mech0)
    print(automech.species_count(mech0))
    print(automech.reaction_count(mech0))
    mech = automech.from_network(automech.network(mech0))
    print(mech)
    assert automech.species_count(mech0) == automech.species_count(mech)
    assert automech.reaction_count(mech0) == automech.reaction_count(mech)


@pytest.mark.parametrize(
    "mech, smis, eqs",
    [
        (MECH_EMPTY, None, None),
        (MECH_NO_REACIONS, None, None),
        (MECH_PROPANE, ("CCC", "[OH]"), ("C3+OH=C3y1+H2O",)),
        (MECH_BUTENE, ("CC=CC", "CC=C[CH2]"), ("C4e2+OH=C4e2y1+H2O",)),
    ],
)
def test__display(mech, smis, eqs):
    """Test automech.display."""
    automech.display(mech, open_browser=False)
    automech.display_species(mech, spc_vals_=smis, spc_key_=Species.smiles)
    automech.display_reactions(mech, eqs=eqs)


@pytest.mark.parametrize(
    "mech, enant, ref_rcount, ref_scount, ref_err_rcount, ref_err_scount, drop_unused",
    [
        (MECH_NO_REACIONS, True, 0, 2, 0, 2, False),
        (MECH_BUTENE, True, 6, 12, 1, 7, False),
        (MECH_BUTENE, False, 4, 10, 1, 7, False),
        (MECH_BUTENE, True, 6, 8, 1, 3, True),
        (MECH_BUTENE_NO_REACTIONS, True, 0, 12, 0, 12, False),
        (MECH_BUTENE_NO_REACTIONS, False, 0, 10, 0, 10, False),
    ],
)
def test__expand_stereo(
    mech, enant, ref_rcount, ref_scount, ref_err_rcount, ref_err_scount, drop_unused
):
    """Test automech.expand_stereo."""
    exp_mech, err_mech = automech.expand_stereo(mech, enant=enant)
    if drop_unused:
        exp_mech = automech.without_unused_species(exp_mech)
        err_mech = automech.without_unused_species(err_mech)
    print(exp_mech)
    print(err_mech)
    rcount = automech.reaction_count(exp_mech)
    scount = automech.species_count(exp_mech)
    err_rcount = automech.reaction_count(err_mech)
    err_scount = automech.species_count(err_mech)
    assert rcount == ref_rcount, f"{rcount} != {ref_rcount}"
    assert scount == ref_scount, f"{scount} != {ref_scount}"
    assert err_rcount == ref_err_rcount, f"{err_rcount} != {ref_err_rcount}"
    assert err_scount == ref_err_scount, f"{err_scount} != {ref_err_scount}"


@pytest.mark.parametrize(
    "mech0, name_mech, nspcs",
    [
        (MECH_BUTENE, MECH_BUTENE_ALTERNATIVE_NAMES, 4),
    ],
)
def test__rename(mech0, name_mech, nspcs):
    """Test automech.rename."""
    name_dct, missing_names = automech.rename_dict(mech0, name_mech)
    print(name_dct)
    print(missing_names)
    mech = automech.rename(mech0, name_dct)
    mech_drop = automech.rename(mech0, name_dct, drop_missing=True)

    print(mech)
    print(mech_drop)
    assert len(name_dct) + len(missing_names) == automech.species_count(mech0)
    assert automech.species_count(mech) == automech.species_count(mech0)
    assert automech.species_count(mech_drop) == nspcs


@pytest.mark.parametrize(
    "par_mech, mech, rcount, scount",
    [(MECH_BUTENE, MECH_NO_REACIONS, 6, 8)],
)
def test__expand_parent_stereo(par_mech, mech, rcount, scount):
    """Test automech.expand_parent_stereo."""
    exp_mech, _ = automech.expand_stereo(mech)
    exp_par_mech = automech.expand_parent_stereo(mech=par_mech, sub_mech=exp_mech)
    print(exp_par_mech)
    assert automech.reaction_count(exp_par_mech) == rcount
    assert automech.species_count(exp_par_mech) == scount


@pytest.mark.parametrize(
    "mech0, smarts, smis_, rcount, scount, src_mech",
    [
        (
            MECH_BUTENE_SUBSET,
            enum.ReactionSmarts.abstraction,
            [["C1=CCCC1", "CC=CC"], "[OH]"],
            5,
            9,
            MECH_BUTENE,
        )
    ],
)
def test__enumerate_reactions(mech0, smarts, smis_, rcount, scount, src_mech):
    """Test automech.enumerate_reactions_from_smarts."""
    mech = automech.enumerate_reactions(
        mech0, smarts, rcts_=smis_, spc_col_=Species.smiles, src_mech=src_mech
    )
    print(mech)
    assert automech.reaction_count(mech) == rcount
    assert automech.species_count(mech) == scount


@pytest.mark.parametrize(
    "mech0, rxn_smi, rcount, scount",
    [(MECH_BUTENE, "CC=CC.[OH]>>C[CH]C(O)C", 2, 7)],
)
def test__drop_reactions_by_smiles(mech0, rxn_smi, rcount, scount):
    """Test automech.enumerate_reactions_from_smarts."""
    mech = automech.drop_reactions_by_smiles(mech0, rxn_smis=[rxn_smi])
    print(mech)
    assert automech.reaction_count(mech) == rcount
    assert automech.species_count(mech) == scount


@pytest.mark.parametrize(
    "mech0, srt_dct0",
    [
        (
            MECH_ETHANE,
            {
                0: (1, 1, 1),
                1: (2, 1, 1),
                2: (2, 1, 2),
                3: (2, 1, 3),
                4: (2, 1, 4),
                5: (3, 1, 1),
            },
        )
    ],
)
def test__with_sort_data(mech0, srt_dct0):
    """Test automech.with_sort_data."""
    mech = automech.with_sort_data(mech0)
    rxn_df = df_.with_index(mech.reactions, "id")
    srt_dct = df_.lookup_dict(
        rxn_df,
        "id",
        [ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel],
    )
    print(srt_dct)
    assert srt_dct == srt_dct0, f"\n{srt_dct} !=\n{srt_dct0}"


@pytest.mark.parametrize(
    "rxn_file_name, spc_file_name, rxn_count, err_count",
    [
        ("syngas.dat", "syngas_species.csv", 74, 4),
    ],
)
def test__sanitize(rxn_file_name, spc_file_name, rxn_count, err_count):
    """Test sanitize function."""
    rxn_path = DATA_PATH / rxn_file_name
    spc_path = DATA_PATH / spc_file_name
    mech = automech.io.mechanalyzer.read.mechanism(rxn_path, spc_path)
    rxn_df, err_df = automech.reaction.sanitize(mech.reactions, spc_df=mech.species)
    assert df_.count(rxn_df) == rxn_count, f"{df_.count(rxn_df)} != {rxn_count}"
    assert df_.count(err_df) == err_count, f"{df_.count(err_df)} != {err_count}"


if __name__ == "__main__":
    # test__from_smiles()
    # test__expand_stereo(MECH_BUTENE, False, 4, 10, 1, 7, False)
    # test__expand_parent_stereo(MECH_BUTENE, MECH_NO_REACIONS, 6, 8)
    test__rename(MECH_BUTENE, MECH_BUTENE_ALTERNATIVE_NAMES, 4)
    # test__update_parent_reaction_data(MECH_BUTENE, MECH_BUTENE_SUBSET, 6, 9)
    # test__display(MECH_EMPTY, None, None)
    # test__network(MECH_EMPTY)
    # test__network(MECH_NO_REACIONS)
    # test__display(MECH_NO_REACIONS, None, None)
    # test__display(MECH_PROPANE, ("CCC", "[OH]"), ("C3+OH=C3y1+H2O",))
    # test__network(MECH_NO_REACIONS)
    # test__enumerate_reactions(
    #     MECH_BUTENE_SUBSET,
    #     enum.ReactionSmarts.abstraction,
    #     [["C1=CCCC1", "CC=CC"], "[OH]"],
    #     5,
    #     9,
    #     MECH_BUTENE,
    # )
    # test__with_sort_data(
    #     MECH_ETHANE,
    #     {
    #         0: (1, 1, 1),
    #         1: (2, 1, 1),
    #         2: (2, 1, 2),
    #         3: (2, 1, 3),
    #         4: (2, 1, 4),
    #         5: (3, 1, 1),
    #     },
    # )
    # test__drop_reactions_by_smiles(MECH_BUTENE, "CC=CC.[OH]>>C[CH]C(O)C", 2, 7)
    # test__expand_stereo(MECH_BUTENE_NO_REACTIONS, True, 0, 12, 0, 12, False)
    # test__expand_stereo(MECH_BUTENE_NO_REACTIONS, False, 0, 10, 0, 10, False)
