"""Test autochem.pes functions."""

from pathlib import Path

import pytest

from protomech import mess

DATA_PATH = Path(__file__).parent / "data"


@pytest.mark.parametrize("mess_inp_name", ["cyclopentene-ho2-addition.inp"])
def test__from_mess_input(mess_inp_name):
    """Test autochem.pes.from_mess_input."""
    mess_inp = DATA_PATH / mess_inp_name
    surf = mess.surf.from_mess_input(mess_inp)
    surf = mess.surf.set_no_fake_well_extension(surf)
    print(mess.surf.mess_input(surf))


if __name__ == "__main__":
    test__from_mess_input("cyclopentene-ho2-addition.inp")
