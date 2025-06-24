"""Functions acting on pandera DataFrame models."""

import itertools
from collections.abc import Sequence
from typing import TypeAlias

import more_itertools as mit
import pandera.polars as pa
import polars

Model: TypeAlias = pa.DataFrameModel


# Model-only functions
def columns(
    model_: Model | Sequence[Model], df: polars.DataFrame | None = None
) -> list[str]:
    """Get model column names.

    :param model_: Model(s)
    :return: Schema, as a mapping of column names to types
    """
    models = normalize_model_input(model_)
    cols = list(
        mit.unique_everseen(
            itertools.chain.from_iterable(m.to_schema().columns for m in models)
        )
    )
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return cols


def schema(
    model_: Model | Sequence[Model],
    col_: str | Sequence[str] | None = None,
    py: bool = False,
) -> dict[str, type]:
    """Get model schema, mapping column names to types.

    :param model_: Model(s)
    :param col_: Column(s)
    :param py: Whether to return Python types instead of Polars types.
    :return: Schema, as a mapping of column names to types
    """
    models = normalize_model_input(model_)

    schema_dct = {
        c: v.dtype.type for m in models for c, v in m.to_schema().columns.items()
    }

    if col_ is not None:
        cols = normalize_column_input(col_)
        schema_dct = {k: v for k, v in schema_dct.items() if k in cols}

    if py:
        schema_dct = {k: v.to_python() for k, v in schema_dct.items()}

    return schema_dct


def dtype(
    model_: Model | Sequence[Model], col: str, py: bool = False
) -> type | list[type]:
    """Get model column data types.

    :param model_: Model(s)
    :param col: Column
    :param py: Whether to return Python types instead of Polars types.
    :return: Data types
    """
    schema_dct = schema(model_, col, py=py)
    return schema_dct.get(col)


def dtypes(
    model_: Model | Sequence[Model], cols: Sequence[str], py: bool = False
) -> list[type]:
    """Get model column data types.

    :param model_: Model(s)
    :param cols: Columns
    :param py: Whether to return Python types instead of Polars types.
    :return: Data types
    """
    schema_dct = schema(model_, cols, py=py)
    return list(map(schema_dct.get, cols))


def empty(model_: Model | Sequence[Model]) -> polars.DataFrame:
    """Create empty DataFrame matching schema.

    :param model_: Model(s)
    :return: Data types
    """
    return polars.DataFrame([], schema=schema(model_))


# Data frame functions
def validate(
    model_: Model | Sequence[Model], df: polars.DataFrame, sort: bool = True
) -> polars.DataFrame:
    """Validate DataFrame against model(s).

    :param model_: Model(s)
    :param df: DataFrame
    :param sort: Whether to sort the columns after validation
    :return: DataFrame
    """
    for model in normalize_model_input(model_):
        df = model.validate(df)
    if sort:
        df = sort_columns(model_, df)
    return df


def impose_schema(
    model_: Model | Sequence[Model], df: polars.DataFrame
) -> polars.DataFrame:
    """Cast DataFrame to model schema(s).

    :param model_: Model(s)
    :param df: DataFrame
    :return: DataFrame
    """
    cols = columns(model_, df)
    return df.cast(schema(model_, cols))


def has_columns(
    model_: Model | Sequence[Model], df: polars.DataFrame
) -> polars.DataFrame:
    """Determine if DataFrame has columns from model(s).

    :param model_: Model(s)
    :param df: DataFrame
    :return: DataFrame
    """
    return all(c in df for c in columns(model_))


def sort_columns(
    model_: Model | Sequence[Model], df: polars.DataFrame
) -> polars.DataFrame:
    """Sort DataFrame columns against model(s).

    :param model_: Model(s)
    :param df: DataFrame
    :return: DataFrame
    """
    cols = columns(model_)
    cols.extend(c for c in df.columns if c not in cols)
    return df.select(cols)


def add_missing_columns(
    model_: Model | Sequence[Model],
    df: polars.DataFrame,
    col_: str | Sequence[str] | None = None,
) -> polars.DataFrame:
    """Add missing model column(s) to DataFrame.

    :param model_: Model(s)
    :param df: DataFrame
    :param col_: Column(s)
    :return: DataFrame
    """
    for col, dtyp in schema(model_, col_).items():
        null_val = polars.lit(None, dtype=dtyp)
        df = df if col in df else df.with_columns(null_val.alias(col))
    return df


def drop_extra_columns(
    model_: Model | Sequence[Model], df: polars.DataFrame
) -> polars.DataFrame:
    """Drop extra column(s) from DataFrame.

    Extra columns are those not in the model(s).

    :param model_: Model(s)
    :param df: DataFrame
    :return: DataFrame
    """
    return df.drop([c for c in df.columns if c not in columns(model_)])


# Helpers
def normalize_model_input(model_: Model | Sequence[Model]) -> list[Model]:
    """Normalize model(s) input.

    :param model_: Model(s)
    :return: List of models
    """
    return list(model_) if isinstance(model_, Sequence) else [model_]


def normalize_column_input(col_: str | Sequence[str]) -> list[str]:
    """Normalize column(s) input.

    :param col_: Column(s)
    :return: List of columns
    """
    return [col_] if isinstance(col_, str) else list(col_)
