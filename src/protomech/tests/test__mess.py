"""Test autochem.pes functions."""

import subprocess
from pathlib import Path

import pytest

import automech
from protomech import mess

DATA_PATH = Path(__file__).parent / "data"
RUN_PATH = Path(__file__).parent / "run"
RUN_PATH.mkdir(exist_ok=True)


@pytest.mark.parametrize("mess_inp_name", ["cyclopentene-ho2-addition.inp"])
def test__from_mess_input(mess_inp_name):
    """Test mess.surf.from_mess_input."""
    mess_inp = DATA_PATH / mess_inp_name
    surf = mess.surf.from_mess_input(mess_inp)
    surf = mess.surf.set_no_fake_well_extension(surf)
    print(mess.surf.mess_input(surf))


def test__plot_paths():
    """Test mess.surf.plot_paths"""
    mess_inp = DATA_PATH / "cyclopentenyl-o2-mess.inp"
    mech_file = DATA_PATH / "cyclopentenyl-o2.json"
    surf = mess.surf.from_mess_input(mess_inp)
    mech = automech.io.read(mech_file)
    mess.net.display(surf, height="1000px", mech=mech, open_browser=False)
    node_paths = mess.surf.node_paths_from_source(surf, 25, leaf_keys=[21, 22])
    print(node_paths)
    mess.surf.plot_paths(surf, node_paths)


def test__prompt_instability():
    """Test prompt (bimolecular) instability merge."""
    mech_file = DATA_PATH / "cyclopentene_mech.json"
    mech = automech.io.read(mech_file)

    mess_inp1 = DATA_PATH / "acrolein-formation.inp"
    mess_inp2 = DATA_PATH / "cyclopentene-prompt-instability.inp"
    surf1 = mess.surf.from_mess_input(mess_inp1)
    surf2 = mess.surf.from_mess_input(mess_inp2)
    surf = mess.surf.combine([surf1, surf2])
    surf = mess.surf.merge_resonant_instabilities(surf, mech)

    # Run MESS on the input to make sure it works
    run_dir = RUN_PATH / "prompt_instability"
    run_dir.mkdir(exist_ok=True)
    run_mess_inp = run_dir / "mess.inp"
    run_mess_inp.write_text(mess.surf.mess_input(surf))
    subprocess.run(["mess", "mess.inp"], cwd=run_dir, check=True)


def test__unimol_instability():
    """Test unimolecular instability merge."""
    mech_file = DATA_PATH / "cyclopentene_mech.json"
    mech = automech.io.read(mech_file)

    mess_inp = DATA_PATH / "cyclopentene-unimol-instability.inp"
    surf = mess.surf.from_mess_input(mess_inp)
    surf = mess.surf.merge_resonant_instabilities(surf, mech)

    # Run MESS on the input to make sure it works
    run_dir = RUN_PATH / "unimol_instability"
    run_dir.mkdir(exist_ok=True)
    run_mess_inp = run_dir / "mess.inp"
    run_mess_inp.write_text(mess.surf.mess_input(surf))
    subprocess.run(["mess", "mess.inp"], cwd=run_dir, check=True)


if __name__ == "__main__":
    # test__from_mess_input("cyclopentene-ho2-addition.inp")
    # test__prompt_instability()
    test__unimol_instability()
