"""Path functions."""

from pathlib import Path


# Root path functions
def data(root_path: str | Path) -> Path:
    """Determine data path from root."""
    return Path(root_path) / "data"


def rmg(root_path: str | Path) -> Path:
    """Determine rmg path from root."""
    return data(root_path) / "rmg"


def mechanalyzer(root_path: str | Path) -> Path:
    """Determine mechanalyzer path from root."""
    return data(root_path) / "mechanalyzer"


def chemkin(root_path: str | Path) -> Path:
    """Determine chemkin path from root."""
    return data(root_path) / "chemkin"


def cantera(root_path: str | Path) -> Path:
    """Determine cantera path from root."""
    return data(root_path) / "cantera"


def cantera_o2(root_path: str | Path) -> Path:
    """Determine cantera path from root."""
    return data(root_path) / "cantera" / "O2"


def cantera_t(root_path: str | Path) -> Path:
    """Determine cantera path from root."""
    return data(root_path) / "cantera" / "T"


def calc(root_path: str | Path, tag: str) -> Path:
    """Determine data path from root."""
    return Path(root_path) / "calc" / tag


def ckin(root_path: str | Path, tag: str) -> Path:
    """Determine data path from root."""
    return Path(root_path) / "calc" / tag / "CKIN"


# Mechanism names and paths
def parent_mechanism(
    ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the parent mechanism."""
    return handle_extension_and_path("full_parent", ext=ext, path=path)


def generated_mechanism(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the original mechanism."""
    return handle_extension_and_path(f"{tag}_gen", ext=ext, path=path)


def stereo_mechanism(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the calculated mechanism."""
    return handle_extension_and_path(f"{tag}_ste", ext=ext, path=path)


def calculated_mechanism(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the calculated mechanism."""
    return handle_extension_and_path(f"{tag}_calc", ext=ext, path=path)


def full_calculated_mechanism(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the full (merged) calculated mechanism."""
    return handle_extension_and_path(f"full_{tag}_calc", ext=ext, path=path)


def full_control_mechanism(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Determine the name of the full (merged) calculated mechanism."""
    return handle_extension_and_path(f"full_{tag}_control", ext=ext, path=path)


# Simulation
def simulation_species(tag: str, path: str | Path | None = None) -> str | Path:
    """Species table for a simulation."""
    return handle_extension_and_path(f"{tag}_species", ext="csv", path=path)


# Helpers
def handle_extension_and_path(
    tag: str, ext: str | None = None, path: str | Path | None = None
) -> str | Path:
    """Add extension to tag.

    :param tag: Mechanism tag
    :param ext: Mechanism file extension
    :param path: Mechanism path
    :return: Name or path of mechanism
    """
    name = tag if ext is None else f"{tag}.{ext}"
    return name if path is None else Path(path) / name
