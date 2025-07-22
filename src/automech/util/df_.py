"""DataFrame utilities."""

from collections.abc import Callable, Sequence
from pathlib import Path

import polars
import polars.dataframe
from polars import selectors as s_
from tqdm.auto import tqdm

from . import c_

Key = str
Keys = Sequence[str]
Key_ = Key | Keys
Value = object
Values = Sequence[object]
Value_ = Value | Values

DEFAULT_COL_SEP = "_"
DEFAULT_LIST_SEP = ","


def count(df: polars.DataFrame) -> int:
    """Count the number of rows in a DataFrame.

    :param df: The DataFrame
    :return: The number of rows
    """
    return df.select(polars.len()).item()


def dtype(df: polars.DataFrame, col_: str | Sequence[str]) -> str | list[str]:
    """Get column(s) data type(s).

    :param df: DataFrame
    :param col_: Column(s)
    :return: Data type(s)
    """
    is_bare = c_.is_bare_column_argument(col_)
    col_ = c_.normalize_column_argument(col_)
    dtypes = [df.schema[col] for col in col_]
    return dtypes[0] if is_bare else dtypes


def fields(df: polars.DataFrame, col: str) -> list[str]:
    """Get fields of a Struct column.

    :param df: DataFrame
    :param col: Column
    :return: Fields
    """
    return [f.name for f in df.schema[col].fields]


def values(
    df: polars.DataFrame,
    col_: str | Sequence[str],
    vals_in_: Sequence[object | Sequence[object]] | None = None,
    col_in_: str | Sequence[str] | None = None,
) -> list[object | tuple[object, ...]]:
    """Get values from a DataFrame.

    :param df: DataFrame
    :param col_: Column(s) to get value(s) for
    :param vals_in_: Optionally, get value(s) for rows matching these input value(s)
    :param col_in_: Column(s) corresponding to `vals_in_`
    :return: _description_
    """
    is_bare = c_.is_bare_column_argument(col_)
    col_ = c_.normalize_column_argument(col_)

    vals_ = df.select(*col_).rows()

    # If requested, narrow the values to rows matching the input values
    if vals_in_ is not None or col_in_ is not None:
        assert vals_in_ is not None and col_in_ is not None, f"{vals_in_} {col_in_}"
        vals_in_, col_in_ = normalize_values_arguments(vals_in_, col_in_)

        idx_col = c_.temp()
        df = with_match_index_column(df, idx_col, vals_=vals_in_, col_=col_in_)
        df = df.filter(polars.col(idx_col).is_not_null()).unique(idx_col)
        miss_idxs = [
            i for i, _ in enumerate(vals_in_) if i not in df.get_column(idx_col)
        ]
        miss_df = polars.DataFrame({idx_col: miss_idxs})
        df = polars.concat([df, miss_df], how="diagonal_relaxed")
        df = df.sort(idx_col)
        vals_ = df.select(col_).rows()

    return [v[0] for v in vals_] if is_bare else vals_


def with_index(
    df: polars.DataFrame, col: str = "index", offset: int = 0
) -> polars.DataFrame:
    """Add index column to DataFrame.

    :param df: DataFrame
    :param col: _description_, defaults to "index"
    :return: _description_
    """
    return df.with_row_index(name=col, offset=offset)


def update(
    df1: polars.DataFrame,
    df2: polars.DataFrame,
    col_: str | Sequence[str],
    drop_orig: bool = True,
    how: str = "full",
) -> polars.DataFrame:
    """Update one DataFrame by another.

    By default, this is a full update merging all data from both mechanisms with
    priority given to the second one. This behavior can be modified by selecting
    different polars join strategy via the `how` keyword.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param col_: Column(s) to join on
    :param drop_orig: Whether to drop the original column values
    :param how: Polars join strategy
    :return: DataFrame
    """
    col_ = c_.normalize_column_argument(col_)

    # Form join columns (needed if multiple are used)
    join_col = c_.temp()
    df1 = with_concat_string_column(df1, join_col, col_)
    df2 = with_concat_string_column(df2, join_col, col_)

    # Cast to a common schema
    schema = polars.concat([df1, df2], how="diagonal_relaxed").schema
    schema1 = {c: t for c, t in schema.items() if c in df1.columns}
    schema2 = {c: t for c, t in schema.items() if c in df2.columns}
    df1 = df1.cast(schema1, strict=False)
    df2 = df2.cast(schema2, strict=False)

    # Join, adding suffix to identify overlapping columns
    suff = f"_{c_.temp()}"
    df1 = df1.join(df2, on=join_col, how=how, suffix=suff, maintain_order="left_right")
    df1 = df1.drop(join_col, f"{join_col}{suff}", strict=False)

    # Handle overlapping column values
    for col in s_.expand_selector(df1, s_.ends_with(suff)):
        col0 = col.removesuffix(suff)

        # If requested to keep original values, back them up
        if not drop_orig:
            df1 = df1.with_columns(polars.col(col0).alias(c_.orig(col0)))

        # Overwrite original values where update values are not null
        df1 = df1.with_columns(
            polars.when(polars.col(col).is_not_null())
            .then(polars.col(col))
            .otherwise(polars.col(col0))
            .alias(col0)
        ).drop(col)

    return df1


def left_update(
    df1: polars.DataFrame,
    df2: polars.DataFrame,
    col_: str | Sequence[str],
    drop_orig: bool = True,
) -> polars.DataFrame:
    """Left-update one DataFrame by another.

    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param col_: Column(s) to join on
    :param drop_orig: Whether to drop the original column values
    :return: DataFrame
    """
    return update(df1, df2, col_=col_, drop_orig=drop_orig, how="left")


def with_match_index_column(
    df: polars.DataFrame,
    col: str,
    vals_: Sequence[object | Sequence[object]],
    col_: str | Sequence[str],
) -> polars.DataFrame:
    """Add match index column for values in a list.

    Requires that all column values be convertible to strings.

    :param df: DataFrame
    :param col: Column name
    :param vals_: Column value(s) to match
    :param col_: Column name(s) corresponding to `vals_`
    :return: DataFrame
    """
    vals_, col_ = normalize_values_arguments(vals_, col_)

    tmp_col = c_.temp()

    # Form DataFrame with concatenated values and indices
    val_data = dict(zip(col_, zip(*vals_, strict=True), strict=True))
    val_df = polars.DataFrame(val_data)
    val_df = with_concat_string_column(val_df, tmp_col, col_=col_)
    val_df = with_index(val_df, col)
    val_df = val_df.drop(*col_)

    # Add concatenated values and join to get match indices
    df = with_concat_string_column(df, tmp_col, col_=col_)
    df = df.join(val_df, on=tmp_col, how="left")
    return df.drop(tmp_col)


def with_concat_string_column(
    df: polars.DataFrame,
    col_out: str,
    col_: str | Sequence[str],
    col_sep: str = DEFAULT_COL_SEP,
    list_sep: str = DEFAULT_LIST_SEP,
) -> polars.DataFrame:
    """Add a column concatenating other columns as strings.

    (AVC note: Could be extended by adding a distinct separator for lists, along with
    sorting options.)

    :param df: DataFrame
    :param col: Column name
    :param src_col_: Source column name(s)
    :param col_sep: Column separator
    :return: DataFrame
    """
    exprs = [concat_string_column_expression(df, c, list_sep=list_sep) for c in col_]
    return df.with_columns(polars.concat_str(*exprs, separator=col_sep).alias(col_out))


def with_sorted_columns(
    df: polars.DataFrame,
    col_: Sequence[str],
    col_out_: Sequence[str] | None = None,
    cross_sort: bool | str = False,
) -> polars.DataFrame:
    """Sort within and, optionally, across list-valued column(s).

    :param df: DataFrame
    :param col_: Column(s) to sort
    :param col_out_: Output column(s), if different from input
    :param cross_sort: Whether to sort across columns, as well as within them.
        Can be Boolean column indicating where to cross-sort.
    :return: DataFrame
    """
    # Process arguments
    col_out_ = col_ if col_out_ is None else col_out_
    col_ = c_.normalize_column_argument(col_)
    col_out_ = c_.normalize_column_argument(col_out_)
    assert len(col_) == len(col_out_), f"{col_} !~ {col_out_}"
    assert all(df.schema[c].base_type() is polars.List for c in col_)

    df = df.with_columns(
        polars.col(c).list.sort().alias(o) for c, o in zip(col_, col_out_, strict=True)
    )

    if cross_sort and not df.is_empty():
        # Convert lists to structs to make them sortable
        df = list_to_struct(df, col_out_)
        # Combine columns into a temp list column for sorting
        srt_col = c_.temp()
        df = df.with_columns(polars.concat_list(col_out_).alias(srt_col))
        # Sort
        df = df.with_columns(
            polars.when(cross_sort)
            .then(polars.col(srt_col).list.sort())
            .otherwise(polars.col(srt_col))
        )
        # Unnest to recover original columns and drop temp column
        df = df.with_columns(
            polars.col(srt_col).list.to_struct(fields=col_out_).struct.unnest()
        )
        df = df.drop(srt_col)
        # Convert structs back to lists
        df = struct_to_list(df, col_out_)

    return df


def has_values(df: polars.DataFrame) -> bool:
    """Determine if DataFrame has non-null values.

    :param df: DataFrame
    :return: `True` if it does, `False` if it doesn't
    """
    return df.select(polars.any_horizontal(polars.col("*").is_not_null().any())).item()


def from_csv(path: str) -> polars.DataFrame:
    """Read a DataFrame from a CSV file.

    :param path: The path to the CSV file
    :return: The DataFrame
    """
    try:
        df = polars.read_csv(path)
    except polars.exceptions.ComputeError:
        df = polars.read_csv(path, quote_char="'")
    return df


def to_csv(
    df: polars.DataFrame, path: str | None, quote_char: str | None = None
) -> None:
    """Write a DataFrame to a CSV file.

    If `path` is `None`, this function does nothing.

    :param df: The DataFrame
    :param path: The path to the CSV file
    :param quote_char: Optionally, override the default quote character
    """
    kwargs = (
        {}
        if quote_char is None
        else {"quote_char": quote_char, "quote_style": "non_numeric"}
    )
    if path is not None:
        path: Path = Path(path)
        df.write_csv(path, **kwargs)


def map_(
    df: polars.DataFrame,
    in_: Key_,
    out_: Key_ | None,
    func_: Callable,
    dct: dict[object, object] | None = None,
    dtype_: polars.DataType | Sequence[polars.DataType] | None = None,
    bar: bool = False,
) -> polars.DataFrame:
    """Map columns from a DataFrame onto a new column.

    :param df: The DataFrame
    :param in_: The input key or keys
    :param out_: The output key or keys; if `None`, the function output will be ignored
    :param func_: The mapping function
    :param dct: A lookup dictionary; if the arguments are a key in this dictionary, its
        value will be returned in place of the function value
    :param dtype_: The data type of the output
    :param bar: Include a progress bar?
    :return: The resulting DataFrame
    """
    dct = {} if dct is None else dct
    in_ = (in_,) if isinstance(in_, str) else tuple(in_)
    dct = {((k,) if isinstance(k, str) else k): v for k, v in dct.items()}

    def row_func_(row: dict[str, object]):
        args = tuple(map(row.get, in_))
        return dct[args] if dct and args in dct else func_(*args)

    row_iter = df.iter_rows(named=True)
    if bar:
        row_iter = tqdm(row_iter, total=df.shape[0])

    vals = list(map(row_func_, row_iter))
    if out_ is None:
        return

    # Post-process the output
    if isinstance(out_, str):
        out_ = (out_,)
        dtype_ = (dtype_,)
        vals_lst = (vals,)
    else:
        vals_lst = list(zip(*vals, strict=True))
        nvals = len(vals_lst)
        assert len(out_) == nvals, f"Cannot match {nvals} output values to keys {out_}"
        dtype_ = (None,) * nvals if dtype_ is None else dtype_

    assert len(out_) == len(dtype_), f"Cannot match {dtype_} dtypes to keys {out_}"

    for out, dtype, vals in zip(out_, dtype_, vals_lst, strict=True):
        df = df.with_columns(
            polars.Series(name=out, values=vals, dtype=dtype, strict=False)
        )

    return df


def lookup_dict(
    df: polars.DataFrame, in_: Key_ | None = None, out_: Key_ | None = None
) -> dict[Value_, Value_]:
    """Form a lookup dictionary mapping one column onto another in a DataFrame.

    Allows mappings between sets of columns, using a tuple of column keys.

    :param df: The DataFrame
    :param in_: The input key or keys; if `None`, the index is used
    :param out_: The output key or keys; if `None`, the index is used
    :return: The dictionary mapping input values to output values
    """
    cols = df.columns

    def check_(key_):
        return (
            True
            if key_ is None
            else key_ in cols if isinstance(key_, str) else all(k in cols for k in key_)
        )

    def values_(key_):
        return (
            range(df.shape[0])
            if key_ is None
            else (
                df[key_].to_list()
                if isinstance(key_, str)
                else df[list(key_)].iter_rows()
            )
        )

    assert check_(in_), f"{in_} not in {df}"
    assert check_(out_), f"{out_} not in {df}"

    return dict(zip(values_(in_), values_(out_), strict=True))


# helpers
def normalize_values_arguments(
    vals_: Sequence[object | Sequence[object]], col_: str | Sequence[str]
) -> tuple[list[object], list[str]]:
    """Normalize value(s) arguments.

    :param vals_: Value(s) list
    :param col_: Column(s)
    :return: Normalized value(s) list and column(s)
    """
    is_bare = c_.is_bare_column_argument(col_)
    col_ = [col_] if is_bare else list(col_)
    vals_ = [[v] for v in vals_] if is_bare else list(vals_)
    return vals_, col_


def concat_string_column_expression(
    df: polars.DataFrame, col: str, list_sep: str = DEFAULT_LIST_SEP
) -> polars.Expr:
    """Form an expression for a concat string column.

    :param df: DataFrame
    :param col: Column
    :return: Expression
    """
    expr = polars.col(col)

    type_ = df.schema[col].base_type()

    if type_ == polars.Struct:
        expr = polars.concat_list(expr.struct.unnest())
        type_ = polars.List

    if type_ == polars.List:
        expr = (
            expr.list.drop_nulls()
            .list.eval(polars.element().cast(polars.String))
            .list.join(list_sep)
        )

    return expr.cast(polars.String)


def list_to_struct(
    df: polars.DataFrame,
    col_: str | Sequence[str],
    col_out_: str | Sequence[str] | None = None,
) -> polars.DataFrame:
    """Convert List column(s) to Struct column(s).

    :param df: DataFrame
    :param col_: Column(s)
    :param col_out_: Output column(s), if different from input
    :return: DataFrame
    """
    # Process arguments
    col_out_ = col_ if col_out_ is None else col_out_
    col_ = c_.normalize_column_argument(col_)
    col_out_ = c_.normalize_column_argument(col_out_)
    assert len(col_) == len(col_out_), f"{col_} !~ {col_out_}"
    assert all(df.schema[c].base_type() is polars.List for c in col_)
    # Convert list to struct
    return df.with_columns(
        list_to_struct_expression(df, c).alias(o)
        for c, o in zip(col_, col_out_, strict=True)
    )


def list_to_struct_expression(df: polars.DataFrame, col: str) -> polars.Expr:
    """Form an expression for converting list to struct column.

    :param df: DataFrame
    :param col: Column
    :return: Expression
    """
    nmax = df.get_column(col).list.len().max() or 0
    return polars.col(col).list.to_struct(
        # Note: The field names *must* be consistent for cross-sorting!
        "max_width",
        fields=[f"field_{i}" for i in range(nmax)],
    )


def struct_to_list(
    df: polars.DataFrame,
    col_: str | Sequence[str],
    col_out_: str | Sequence[str] | None = None,
    drop_nulls: bool = True,
) -> polars.DataFrame:
    """Convert Struct column(s) to List column(s).

    :param df: DataFrame
    :param col_: Column(s)
    :param col_out_: Output column(s), if different from input
    :param drop_nulls: Whether to drop nulls from the list values
    :return: DataFrame
    """
    # Process arguments
    col_out_ = col_ if col_out_ is None else col_out_
    col_ = c_.normalize_column_argument(col_)
    col_out_ = c_.normalize_column_argument(col_out_)
    assert len(col_) == len(col_out_), f"{col_} !~ {col_out_}"
    assert all(df.schema[c].base_type() is polars.Struct for c in col_)
    # Convert struct to list
    # (sort the field names to be sure they come out in the expected order)
    fields_ = {c: sorted(f.name for f in df.schema[c].fields) for c in col_}.get
    df = df.with_columns(
        polars.concat_list(polars.col(c).struct.field(f) for f in fields_(c)).alias(o)
        for c, o in zip(col_, col_out_, strict=True)
    )
    # Drop nulls, if requested
    if drop_nulls:
        df = df.with_columns(polars.col(o).list.drop_nulls() for o in col_out_)
    return df
