"""Functions for reading and writing AutoMech-formatted files."""

import os
from pathlib import Path

from .._mech import Mechanism


def write(mech: Mechanism, out: str | Path | None = None) -> str | None:
    """Write a mechanism to JSON format.

    :param mech: A mechanism
    :param path: The path to write to (either directory or reactions file)
    :param prefix: File name prefix, used if path is a directory
    """
    mech_str = mech.model_dump_json()
    if out is None:
        return mech_str

    out = Path(out)
    out = out if out.suffix == ".json" else out.with_suffix(".json")
    out.write_text(mech_str)


def read(inp: str | Path | None = None) -> Mechanism:
    """Read a mechanism from JSON format.

    :param path: The path to write to (either directory or reactions file)
    :param prefix: File name prefix, used if path is a directory
    :return: The mechanism
    """
    inp = Path(inp).read_text() if os.path.exists(inp) else str(inp)
    return Mechanism.model_validate_json(inp)
