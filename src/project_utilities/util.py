"""General utility functions."""

from pathlib import Path

import IPython


def is_notebook() -> bool:
    """Check if this is a notebook."""
    return IPython.get_ipython() is not None  # pyright: ignore[reportPrivateImportUsage]


def notebook_file() -> str:
    """Get the notebook file path, if this is a notebook."""
    ipy = IPython.get_ipython()  # pyright: ignore[reportPrivateImportUsage]
    assert ipy is not None
    return ipy.user_ns["__vsc_ipynb_file__"]


# Tag functions
def file_tag(file_path: str) -> str:
    """Determine the prefix of the current IPython notebook."""
    return Path(file_path).stem.split(".")[0]


def previous_tag(tag: str) -> str | None:
    """Determine the previous tags."""
    tags = previous_tags(tag)
    return tags[-1] if tags else None


def previous_tags(tag: str) -> list[str]:
    """Determine the previous tags."""
    pre = tag[:-1]
    ver = int(tag[-1])
    return [f"{pre}{i}" for i in range(ver)]
