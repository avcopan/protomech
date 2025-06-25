"""Test autochem.pes functions."""

from pathlib import Path

import pytest
from autochem import pes

DATA_PATH = Path(__file__).parent / "data"


@pytest.mark.parametrize("mess_inp_name", ["cyclopentene-ho2-addition.inp"])
def test__from_mess_input(mess_inp_name):
    """Test autochem.pes.from_mess_input."""
    mess_inp = DATA_PATH / mess_inp_name
    surf = pes.from_mess_input(mess_inp)
    fake_keys = pes.fake_well_keys(surf)
    surf = pes.set_no_well_extension(surf, fake_keys)
    print(pes.mess_input(surf))


if __name__ == "__main__":
    test__from_mess_input("cyclopentene-ho2-addition.inp")
