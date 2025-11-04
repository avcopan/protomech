"""Functions acting on reactions DataFrames."""

import functools
import itertools
from collections.abc import Collection, Mapping, Sequence
from typing import Annotated

import autochem as ac
import automol
import more_itertools as mit
import polars
import pydantic
from autochem.util import chemkin
from pandera import polars as pa
from pydantic_core import core_schema

from . import species
from .species import Species
from .util import c_, df_, pandera_
from .util.pandera_ import Model

DEFAULT_REAGENT_SEPARATOR = " + "


class Reaction(Model):
    """Core reaction table."""

    reactants: list[str]
    products: list[str]
    formula: Annotated[
        polars.Struct,
        {
            "H": polars.Int64,
            "He": polars.Int64,
            "Li": polars.Int64,
            "Be": polars.Int64,
            "B": polars.Int64,
            "C": polars.Int64,
            "N": polars.Int64,
            "O": polars.Int64,
            "F": polars.Int64,
            "Ne": polars.Int64,
            "Na": polars.Int64,
            "Mg": polars.Int64,
            "Al": polars.Int64,
            "Si": polars.Int64,
            "P": polars.Int64,
            "S": polars.Int64,
            "Cl": polars.Int64,
            "Ar": polars.Int64,
        },
    ] = pa.Field(coerce=True)


class ReactionRate(Model):
    """Reaction table with rate."""

    reversible: bool
    rate: polars.Struct


class ReactionRateExtra(Model):
    """Reaction table with rate."""

    rate_data: polars.Struct
    branch_frac: polars.Struct
    rev_rate: polars.Struct
    rev_rate_data: polars.Struct
    rev_branch_frac: polars.Struct
    well_skipping: bool
    cleared: bool
    partially_cleared: bool


assert all(
    f in pandera_.columns([Reaction, ReactionRate])
    for f in ac.rate.Reaction.model_fields
), "Make sure field names match autochem."


class ReactionSorted(Model):
    """Reaction table with sort information."""

    pes: int
    subpes: int
    channel: int


class ReactionStereo(Model):
    """Stereo-expanded reaction table."""

    amchi: str
    canon: bool
    orig_reactants: str
    orig_products: str


class ReactionError(Model):
    """Reaction error table."""

    is_missing_species: bool
    has_unbalanced_formula: bool


class ReactionMerged(Model):
    """Reaction table with unstable species."""

    replacements: polars.Struct


# validation
def validate(
    df: polars.DataFrame, model_: Model | Sequence[Model] = (), sort: bool = True
) -> polars.DataFrame:
    """Validate reactions DataFrame against model(s).

    :param df: DataFrame
    :param model_: Model(s)
    :return: DataFrame
    """
    models = [Reaction, *pandera_.normalize_model_input(model_)]
    if df.is_empty():
        return pandera_.empty(models)
    df = pandera_.validate(models, df)
    return df


ReactionDataFrame_ = Annotated[
    pydantic.SkipValidation[polars.DataFrame],
    pydantic.BeforeValidator(polars.DataFrame),
    pydantic.AfterValidator(validate),
    pydantic.PlainSerializer(lambda x: polars.DataFrame(x).to_dict(as_series=False)),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(dict[str, list]))
    ),
]


# properties
def has_rates(rxn_df: polars.DataFrame) -> bool:
    """Determine whether a reactions DataFrame has rates.

    :param rxn_df: Reactions DataFrame
    :return: `True` if it does, `False` if not
    """
    return ReactionRate.rate in rxn_df and df_.has_values(
        rxn_df.get_column(ReactionRate.rate).struct.unnest()
    )


def reagents(rxn_df: polars.DataFrame) -> list[list[str]]:
    """Get reagents as lists.

    :param rxn_df: A reactions DataFrame
    :return: The reagents
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(rcts + prds))


def species_names(rxn_df: polars.DataFrame) -> list[str]:
    """Get species in reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Species names
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(itertools.chain.from_iterable(rcts + prds)))


def reactant_names(rxn_df: polars.DataFrame) -> list[str]:
    """Get reatants in reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Species names
    """
    rcts = rxn_df.get_column(Reaction.reactants).to_list()
    return sorted(mit.unique_everseen(itertools.chain.from_iterable(rcts)))


def product_names(rxn_df: polars.DataFrame) -> list[str]:
    """Get reatants in reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Species names
    """
    prds = rxn_df.get_column(Reaction.products).to_list()
    return sorted(mit.unique_everseen(itertools.chain.from_iterable(prds)))


def reagent_strings(
    rxn_df: polars.DataFrame, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> list[str]:
    """Get reagents as strings.

    :param rxn_df: A reactions DataFrame
    :param sep: The separator for joining reagent strings
    :return: The reagents as strings
    """
    return [sep.join(r) for r in reagents(rxn_df)]


def reaction_rate_objects(rxn_df: polars.DataFrame, eq: str) -> list[ac.rate.Reaction]:
    """Get rate objects associated with one reaction.

    :param rxn_df: Reaction DataFrame
    :param eq: Equation
    :return: Rate objects
    """
    tmp_col = c_.temp()
    rxn_df = with_equation_match_column(rxn_df, col=tmp_col, eqs=[eq])
    rxn_df = rxn_df.filter(polars.col(tmp_col)).drop(tmp_col)
    rxn_df = rxn_df.filter(
        polars.col(ReactionRate.rate).is_not_null()
        & polars.col(ReactionRate.reversible).is_not_null()
    )
    rxn_df = with_rate_object_column(rxn_df, col=tmp_col)
    return rxn_df.get_column(tmp_col).to_list()


# binary operations
def update(
    rxn_df1: polars.DataFrame,
    rxn_df2: polars.DataFrame,
    drop_orig: bool = True,
    how: str = "full",
) -> polars.DataFrame:
    """Update reaction data by reaction key.

    Note: Reactants and products are updated as well, so there is no inconsistency due
    to reversing reaction direction relative to the rate constant.

    :param rxn_df1: Reaction DataFrame
    :param rxn_df2: Reaction DataFrame to update from
    :param drop_orig: Whether to drop original column values
    :param how: Polars join strategy
    :return: Reaction DataFrame
    """
    # Add reaction keys
    tmp_col = c_.temp()
    rxn_df1 = with_key(rxn_df1, tmp_col, reversible=True)
    rxn_df2 = with_key(rxn_df2, tmp_col, reversible=True)

    # Update
    rxn_df1 = df_.update(rxn_df1, rxn_df2, col_=tmp_col, drop_orig=drop_orig, how=how)
    return rxn_df1.drop(tmp_col)


def left_update(
    rxn_df1: polars.DataFrame, rxn_df2: polars.DataFrame, drop_orig: bool = True
) -> polars.DataFrame:
    """Left-update reaction data by reaction key.

    Note: Reactants and products are updated as well, so there is no inconsistency due
    to reversing reaction direction relative to the rate constant.

    :param rxn_df1: Reaction DataFrame
    :param rxn_df2: Reaction DataFrame to update from
    :param drop_orig: Whether to drop original column values
    :return: Reaction DataFrame
    """
    return update(rxn_df1, rxn_df2, drop_orig=drop_orig, how="left")


def difference(
    rxn_df1: polars.DataFrame,
    rxn_df2: polars.DataFrame,
    spc_df1: polars.DataFrame | None = None,
    spc_df2: polars.DataFrame | None = None,
    reversible: bool = False,
    stereo: bool = True,
) -> polars.DataFrame:
    """Get reaction data set difference.

    :param rxn_df1: Reaction DataFrame
    :param rxn_df2: Reaction DataFrame to update from
    :param spc_df1: Optional species DataFrame for using unique species IDs
    :param spc_df2: Optional species DataFrame for using unique species IDs
    :param reversible: Whether to treat reactions as reversible
    :param stereo: Whether to include stereochemistry
    :return: Reaction DataFrame
    """
    key_col = c_.temp()
    assert not (spc_df1 is None) ^ (spc_df2 is None), (
        "Requires species data for each reaction set."
    )
    rxn_df1 = with_key(
        rxn_df1, col=key_col, spc_df=spc_df1, reversible=reversible, stereo=stereo
    )
    rxn_df2 = with_key(
        rxn_df2, col=key_col, spc_df=spc_df2, reversible=reversible, stereo=stereo
    )
    return rxn_df1.filter(~polars.col(key_col).is_in(rxn_df2[key_col])).drop(key_col)


# add/remove rows
def drop_self_reactions(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Drop self-reactions from reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :return: Reactions DataFrame
    """
    rcol0 = Reaction.reactants
    pcol0 = Reaction.products
    rcol, pcol = c_.prefix((rcol0, pcol0), c_.temp())
    rxn_df = with_sorted_reagents(
        rxn_df, col_=(rcol0, pcol0), col_out_=(rcol, pcol), reversible=False
    )
    rxn_df = rxn_df.filter(polars.col(rcol) != polars.col(pcol))
    return rxn_df.drop(rcol, pcol)


def drop_noncanonical_enantiomers(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Drop non-canonical enantiomer reactions.

    :param rxn_df: Reactions DataFrame
    :return: Reactions DataFrame
    """
    return rxn_df.filter(polars.col(ReactionStereo.canon))


# add/remove columns
def with_key(
    rxn_df: polars.DataFrame,
    col: str = "key",
    spc_df: polars.DataFrame | None = None,
    reversible: bool | str = False,
    stereo: bool = True,
) -> polars.DataFrame:
    """Add a key for identifying unique reactions to this DataFrame.

    The key is formed by sorting within and across reactants and products to form a
    reaction ID and then joining to form a concatenated string.

    If a species DataFrame is passed in, the reaction keys will be name-agnostic.

    :param rxn_df: Reactions DataFrame
    :param col: Column name
    :param spc_df: Optional species DataFrame, for using unique species IDs
    :param reversibe: Whether reactions are reversible, in which case the reagents will
        be cross-sorted to a canonical direction. Can be specified by a Boolean column
        indicating which reactions are reversible.
    :param stereo: Whether to include stereochemistry
    :return: A reactions DataFrame with this key as a new column
    """
    rct_col0 = Reaction.reactants
    prd_col0 = Reaction.products
    rct_col = c_.temp()
    prd_col = c_.temp()

    # If requested, use species keys instead of names
    if spc_df is not None:
        id_col = c_.temp()
        spc_df = species.with_key(spc_df, id_col, stereo=stereo)
        rxn_df = translate_reagents(
            rxn_df,
            spc_df[Species.name],
            spc_df[id_col],
            rct_col=rct_col,
            prd_col=prd_col,
        )
        rct_col0 = rct_col
        prd_col0 = prd_col

    # Sort reagents
    rxn_df = with_sorted_reagents(
        rxn_df,
        col_=[rct_col0, prd_col0],
        col_out_=[rct_col, prd_col],
        reversible=reversible,
    )

    # Concatenate
    rxn_df = df_.with_concat_string_column(
        rxn_df, col, col_=[rct_col, prd_col], col_sep="=", list_sep="+"
    )
    return rxn_df.drop(rct_col, prd_col)


def with_duplicate_column(rxn_df: polars.DataFrame, col: str) -> polars.DataFrame:
    """Generate a column indicating which reactions are duplicate.

    :param rxn_df: Reactions DataFrame
    :param col: Duplicate column
    :return: Reactions DataFrame
    """
    tmp_col = c_.temp()
    reversible = ReactionRate.reversible if ReactionRate.reversible in rxn_df else True
    rxn_df = with_key(rxn_df, col=tmp_col, reversible=reversible)
    rxn_df = rxn_df.with_columns(polars.col(tmp_col).is_duplicated().alias(col))
    return rxn_df.drop(tmp_col)


def with_sorted_reagents(
    rxn_df: polars,
    col_: Sequence[str] = (Reaction.reactants, Reaction.products),
    col_out_: Sequence[str] | None = None,
    reversible: bool | str = False,
) -> polars.DataFrame:
    """Generate sorted reagents columns.

    :param rxn_df: Reactions DataFrame
    :param col_: Reactant and product column(s)
    :param col_out_: Output reactant and product column(s), if different from input
    :param reversibe: Whether reactions are reversible, in which case the reagents will
        be cross-sorted to a canonical direction. Can be specified by a Boolean column
        indicating which reactions are reversible.
    :return: Reactions DataFrame
    """
    col_out_ = col_ if col_out_ is None else col_out_
    assert len(col_) == 2, f"len({col_}) != 2"
    assert len(col_out_) == 2, f"len({col_out_}) != 2"
    return df_.with_sorted_columns(
        rxn_df, col_=col_, col_out_=col_out_, cross_sort=reversible
    )


def with_rate_object_columns(
    rxn_df: polars.DataFrame, obj_mapping: Mapping[str, str]
) -> polars.DataFrame:
    """Add rate object columns.

    :param rxn_df: Reaction DataFrame
    :param obj_mapping: Mapping of rate data to object columns
    :return: Reaction DataFrame
    """
    cols = [
        Reaction.reactants,
        Reaction.products,
        ReactionRate.reversible,
    ]
    fields = [
        Reaction.reactants,
        Reaction.products,
        ReactionRate.reversible,
        ReactionRate.rate,
    ]
    return rxn_df.with_columns(
        polars.struct([*cols, rate_col])
        .struct.rename_fields(fields)
        .map_elements(ac.rate.Reaction.model_validate, return_dtype=polars.Object)
        .alias(obj_col)
        for rate_col, obj_col in obj_mapping.items()
    )


def update_rate_data_from_object_columns(
    rxn_df: polars.DataFrame, obj_mapping: Mapping[str, str]
) -> polars.DataFrame:
    """Add rate object columns.

    :param rxn_df: Reaction DataFrame
    :param obj_mapping: Mapping of rate data to object columns
    :return: Reaction DataFrame
    """
    rate_cols = list(obj_mapping.keys())
    rxn_df = rxn_df.drop(rate_cols).with_columns(
        polars.Series(
            rate_col,
            [o.rate.model_dump() for o in rxn_df.get_column(obj_col)],
            strict=False,
        )
        for rate_col, obj_col in obj_mapping.items()
    )
    return rxn_df


def with_rate_object_column(
    rxn_df: polars.DataFrame,
    col: str,
    fill: bool = False,
    rate_col: str = ReactionRate.rate,  # type: ignore
    reverse: bool = False,
) -> polars.DataFrame:
    """Add a column of reaction rate objects.

    :param rxn_df: Reaction DataFrame
    :param col: Column
    :param fill: Whether to fill missing rates with dummy values
    :return: Reaction DataFrame
    """
    if fill:
        rxn_df = with_rates(rxn_df)

    cols = [
        Reaction.reactants if not reverse else Reaction.products,
        Reaction.products if not reverse else Reaction.reactants,
        ReactionRate.reversible,
        rate_col,
    ]
    fields = [
        Reaction.reactants,
        Reaction.products,
        ReactionRate.reversible,
        ReactionRate.rate,
    ]
    return rxn_df.with_columns(
        polars.struct(cols)
        .struct.rename_fields(fields)
        .map_elements(ac.rate.Reaction.model_validate, return_dtype=polars.Object)
        .alias(col)
    )


def with_dummy_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Add placeholder rates to this DataFrame, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    rev0 = True
    rate0 = ac.rate.ArrheniusRateFit().model_dump()
    rxn_df = rxn_df.drop(ReactionRate.rate, ReactionRate.reversible, strict=False)
    rxn_df = rxn_df.with_columns(polars.lit(rev0).alias(ReactionRate.reversible))
    rxn_df = rxn_df.with_columns(polars.lit(rate0).alias(ReactionRate.rate))
    return rxn_df


def with_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Add placeholder rates to this DataFrame, if missing.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    rev0 = True
    rate0 = ac.rate.ArrheniusRateFit().model_dump()

    if ReactionRate.reversible not in rxn_df:
        rxn_df = rxn_df.with_columns(polars.lit(rev0).alias(ReactionRate.reversible))

    if ReactionRate.rate not in rxn_df:
        rxn_df = rxn_df.with_columns(polars.lit(rate0).alias(ReactionRate.rate))

    rev0_lit = polars.lit(rev0, dtype=df_.dtype(rxn_df, ReactionRate.reversible))
    rate0_lit = polars.lit(rate0, dtype=df_.dtype(rxn_df, ReactionRate.rate))
    rxn_df = rxn_df.with_columns(
        polars.col(ReactionRate.reversible).fill_null(rev0_lit)
    )
    rxn_df = rxn_df.with_columns(polars.col(ReactionRate.rate).fill_null(rate0_lit))
    return rxn_df


def without_rates(rxn_df: polars.DataFrame) -> polars.DataFrame:
    """Remove rate data from this DataFrame, if present.

    This is mainly needed for ChemKin mechanism writing.

    :param rxn_df: Reaction DataFrame
    :return: Reaction DataFrame
    """
    return rxn_df.drop(ReactionRate.rate, ReactionRate.reversible, strict=False)


def with_species_presence_column(
    rxn_df: polars.DataFrame, col: str, species_names: Sequence[str]
) -> polars.DataFrame:
    """Add a column indicating the presence of one or more species.

    :param rxn_df: A reactions DataFrame
    :param col: The column name
    :param species_names: Species names
    :return: The modified reactions DataFrame
    """
    return rxn_df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(species_names))
        .list.any()
        .alias(col)
    )


def with_equation_match_column(
    rxn_df: polars.DataFrame, col: str, eqs: Sequence[str]
) -> polars.DataFrame:
    """Add a column indicating which equations match those in a list.

    :param rxn_df: Reactions DataFrame
    :param col: Column
    :param eqs: Reaction equations
    :return: Reaction DataFrame
    """
    rxns = list(map(chemkin.read_equation_reagents, eqs))
    expr = reactions_match_expression(rxns)
    return rxn_df.with_columns(expr.alias(col))


def with_reagent_strings_column(
    rxn_df: polars.DataFrame, col: str, sep: str = DEFAULT_REAGENT_SEPARATOR
) -> polars.DataFrame:
    """Add a column containing the reagent strings on either side of the reaction.

    e.g. ["C2H6 + OH", "C2H5 + H2O"]

    :param rxn_df: A reactions DataFrame
    :param col: The column name
    :param sep: The separator for joining reagent strings
    :return: The reactions DataFrame with this extra column
    """
    return rxn_df.with_columns(
        polars.concat_list(
            polars.col(Reaction.reactants).list.join(sep),
            polars.col(Reaction.products).list.join(sep),
        ).alias(col)
    )


def rename(
    rxn_df: polars.DataFrame,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    orig_prefix: str | None = None,
) -> polars.DataFrame:
    """Rename species in a reactions DataFrame.

    :param rxn_df: Reactions DataFrame
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :return: Reactions DataFrame
    """
    col_dct = c_.to_([Reaction.reactants, Reaction.products], orig_prefix or c_.temp())
    rxn_df = rxn_df.with_columns(polars.col(c0).alias(c) for c0, c in col_dct.items())
    rxn_df = translate_reagents(rxn_df=rxn_df, trans=names, trans_into=new_names)
    if not orig_prefix:
        rxn_df = rxn_df.drop(col_dct.values())
    return rxn_df


def translate_reagents(
    rxn_df: polars.DataFrame,
    trans: Sequence[object] | Mapping[object, object],
    trans_into: Sequence[object] | None = None,
    rct_col: str = Reaction.reactants,
    prd_col: str = Reaction.products,
) -> polars.DataFrame:
    """Translate the reagent names in a reactions DataFrame.

    :param rxn_df: A reactions DataFrame
    :param trans: A translation mapping or a sequence of values to replace
    :param trans_into: If `trans` is a sequence, a sequence of values to replace by,
        defaults to None
    :param rct_col: The column name to use for the reactants
    :param prd_col: The column name to use for the products
    :return: The updated reactions DataFrame
    """
    expr = (
        polars.element().replace(trans)
        if trans_into is None
        else polars.element().replace(trans, trans_into)
    )

    return rxn_df.with_columns(
        polars.col(Reaction.reactants).list.eval(expr).alias(rct_col),
        polars.col(Reaction.products).list.eval(expr).alias(prd_col),
    )


def select_pes(
    rxn_df: polars.DataFrame,
    formula_: str | dict | Sequence[str | dict],
    exclude: bool = False,
) -> polars.DataFrame:
    """Select (or exclude) PES by formula(s).

    :param rxn_df: Reaction DataFrame
    :param formula_: PES formula(s) to include or exclude
    :param exclude: Whether to exclude or include the formula(s)
    :return: Reaction DataFrame
    """
    formula_ = [formula_] if isinstance(formula_, str | dict) else formula_
    fmls = [automol.form.from_string(f) if isinstance(f, str) else f for f in formula_]

    def _match(fml: dict[str, int]) -> bool:
        return any(automol.form.match(fml, f) for f in fmls)

    col_tmp = c_.temp()
    rxn_df = df_.map_(rxn_df, Reaction.formula, col_tmp, _match)
    match_expr = polars.col(col_tmp)
    rxn_df = rxn_df.filter(~match_expr if exclude else match_expr)
    rxn_df = rxn_df.drop(col_tmp)
    return rxn_df


# Bootstrapping function
def bootstrap(
    data: dict[str, Sequence[object]] | polars.DataFrame,
    name_dct: dict[str, str] | None = None,
    spc_df: polars.DataFrame | None = None,
) -> polars.DataFrame:
    """Bootstrap species DataFrame from minimal data.

    :param data: Data
    :param name_dct: Rename reactants and products
    :param spc_df: Species DataFrame
    :return: DataFrame
    """
    # 0. Make dataframe from given data
    df = polars.DataFrame(data, strict=False)
    df = df.rename({c: str.lower(c) for c in df.columns})
    df = pandera_.impose_schema(Reaction, df)

    # If empty, return early
    if df.is_empty():
        df = pandera_.add_missing_columns(Reaction, df)
        return validate(df)

    if name_dct is not None:
        df = translate_reagents(df, name_dct)

    if spc_df is None:
        dtype = pandera_.dtype(Reaction, Reaction.formula)
        df = df.with_columns(polars.lit(None, dtype=dtype).alias(Reaction.formula))
    else:
        df = with_formula(df, spc_df=spc_df)

    return validate(df)


def sanitize(
    df: polars.DataFrame, spc_df: polars.DataFrame
) -> tuple[polars.DataFrame, polars.DataFrame]:
    """Remove invalid data from reaction DataFrame.

    :param rxn_df: Reaction DataFrame
    :param spc_df: Species DataFrame
    :return: Reaction DataFrame and error DataFrame
    """
    # 1. Check for missing species errors
    names = spc_df.get_column(Species.name).implode()
    err_col = ReactionError.is_missing_species
    df = df.with_columns(
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(names))
        .list.all()
        .not_()
        .alias(err_col)
    )

    # 2. Check for unbalanced formula errors
    rcol, pcol = c_.temp(), c_.temp()
    df = with_formula(df, spc_df=spc_df, col_in=Reaction.reactants, col_out=rcol)
    df = with_formula(df, spc_df=spc_df, col_in=Reaction.products, col_out=pcol)
    err_col = ReactionError.has_unbalanced_formula
    df = df.with_columns((polars.col(rcol) != polars.col(pcol)).alias(err_col))
    df = df.drop(pcol)

    # 3. Separate wheat from chaff
    err_cols = pandera_.columns(ReactionError)
    has_err = polars.concat_list(err_cols).list.any()
    err_df = df.filter(has_err)
    df = df.filter(~has_err).drop(err_cols)
    return validate(df), err_df


def with_formula(
    df: polars.DataFrame,
    spc_df: polars.DataFrame,
    col_in: str = Reaction.reactants,
    col_out: str = Reaction.formula,
) -> polars.DataFrame:
    """Determine reaction formulas from their reagents (reactants or products).

    :param df: The DataFrame
    :param spc_df: Optionally, pass in a species DataFrame for determining formulas
    :param col_in: A column with lists of species names (reactants or products), used to
        determine the overall formula
    :param col_out: The name of the new formula column
    :return: The reaction DataFrame with the new formula column
    """
    # If the column already exists and we haven't passed in a species dataframe, make
    # sure we don't wipe it out
    if Reaction.formula in df and spc_df is None:
        return df

    tmp_col = c_.temp()
    dtype = pandera_.dtype(Species, Species.formula)
    names = spc_df[Species.name]
    formulas = spc_df[Species.formula]
    expr = polars.element().replace_strict(names, formulas, default={"H": None})
    df = df.with_columns(polars.col(col_in).list.eval(expr).alias(tmp_col))
    df = df_.map_(df, tmp_col, col_out, automol.form.join_sequence, dtype_=dtype)
    df = df.drop(tmp_col)
    return df


# helpers
def reactions_match_expression(
    rxns: Sequence[tuple[Sequence[str], Sequence[str]]],
    cols: tuple[str, str] = (Reaction.reactants, Reaction.products),
) -> polars.Expr:
    """Expression for matching a list of reagents.

    :param rxn: Reactans and products list
    :param cols: Reactant and product columns
    :return: Expression
    """
    exprs = [reaction_match_expression(r, cols) for r in rxns]
    return functools.reduce(polars.Expr.or_, exprs)


def reaction_match_expression(
    rxn: tuple[Sequence[str], Sequence[str]],
    cols: tuple[str, str] = (Reaction.reactants, Reaction.products),
) -> polars.Expr:
    """Expression for matching a list of reagents.

    :param rxn: Reactans and products
    :param cols: Reactant and product columns
    :return: Expression
    """
    assert len(rxn) == len(cols) == 2
    rcts, prds = rxn
    rcol, pcol = cols
    return reagents_match_expression(rcts, rcol) & reagents_match_expression(prds, pcol)


def reagents_match_expression(
    rgts: Collection[str],
    col: str = Reaction.reactants,  # type: ignore
) -> polars.Expr:
    """Expression for matching a list of reagents.

    :param rgts: Reagents
    :param col: Column
    :return: Expression
    """
    return polars.col(col).list.set_symmetric_difference(rgts).list.len() == 0
