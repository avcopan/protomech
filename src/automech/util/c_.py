"""Column schema."""

import random
import string
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")
ORIG = "orig"
DEFAULT_SEP = "_"


def temp(length: int = 9) -> str:
    """Generate a unique temporary column name for a DataFrame.

    :param length: The length of the temporary column name, defaults to 24
    :return: The column name
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def prefix(
    col_: str | Sequence[str], pre: str, sep: str = DEFAULT_SEP
) -> str | list[str]:
    """Add prefix to column name.

    :param col_: Column(s)
    :param pre: Prefix
    :param sep: Separator
    :return: Column(s)
    """
    f_ = (f"{pre}{sep}" + "{}").format
    return f_(col_) if is_bare_column_argument(col_) else list(map(f_, col_))


def to_(col_: str | Sequence[str], pre: str, sep: str = DEFAULT_SEP) -> dict[str, str]:
    """Create mapping to prefixed columns.

    :param col_: Column(s)
    :param pre: Prefix
    :param sep: Separator
    :return: Mapping
    """
    col_ = normalize_column_argument(col_)
    return {c: prefix(c, pre=pre, sep=sep) for c in col_}


def from_(
    col_: str | Sequence[str], pre: str, sep: str = DEFAULT_SEP
) -> dict[str, str]:
    """Create mapping from prefixed-columns.

    :param col_: Column(s)s
    :param pre: Prefix
    :param sep: Separator
    :return: Mapping
    """
    col_ = normalize_column_argument(col_)
    return {prefix(c, pre=pre, sep=sep): c for c in col_}


def orig(col_: str | Sequence[str]) -> str | list[str]:
    """Add `orig_` prefix to column name.

    :param col_: Column(s)
    :return: Column(s)
    """
    return prefix(col_, pre=ORIG)


def to_orig(col_: str | Sequence[str]) -> dict[str, str]:
    """Create mapping to `orig_` columns.

    :param col_: Column(s)s
    :return: Mapping
    """
    return to_(col_, pre=ORIG, sep="_")


def from_orig(col_: str | Sequence[str]) -> dict[str, str]:
    """Create mapping from `orig_` columns.

    :param col_: Column(s)s
    :return: Mapping
    """
    col_ = normalize_column_argument(col_)
    return {orig(c): c for c in col_}


# helpers
def is_bare_column_argument(col_: str | Sequence[str]) -> bool:
    """Determine whether a column(s) argument is bare.

    :param col_: Column(s)
    :return: `True` if it is, `False` if it isn't
    """
    return isinstance(col_, str)


def normalize_column_argument(col_: str | Sequence[str]) -> list[str]:
    """Normalize column(s) argument.

    :param col_: Column(s)
    :return: Column(s)
    """
    return [col_] if is_bare_column_argument(col_) else col_
