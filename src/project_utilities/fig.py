import functools
import itertools
import operator
from collections.abc import Sequence

import altair as alt
import autochem as ac
import numpy as np
import polars as pl

from automech import Mechanism, combine_enantiomers, consumption_mechanism, reaction
from automech.util import c_
from automech.util.pandera_ import Model

polars = pl


class RateData(Model):
    reactants: list[str]
    products: list[str]
    # Forward data:
    rate: polars.Struct
    rate_data: polars.Struct
    branch_frac: polars.Struct
    # Forward objects:
    rate_obj: polars.Object
    rate_data_obj: polars.Object
    branch_frac_obj: polars.Object
    # Reverse data:
    rev_rate: polars.Struct
    rev_rate_data: polars.Struct
    rev_branch_frac: polars.Struct
    # Descriptors:
    well_skipping: bool
    cleared: bool
    partially_cleared: bool
    # Plotting:
    label: str
    color: str


RATE_COLS: tuple[str, ...] = [
    RateData.rate,
    RateData.rate_data,
    RateData.branch_frac,
]  # type: ignore
REV_RATE_COLS: tuple[str, ...] = [
    RateData.rev_rate,
    RateData.rev_rate_data,
    RateData.rev_branch_frac,
]  # type: ignore
REV_MAPPING: dict[str, str] = {
    RateData.rate: RateData.rev_rate,
    RateData.rate_data: RateData.rev_rate_data,
    RateData.branch_frac: RateData.rev_branch_frac,
}  # type: ignore
OBJ_MAPPING: dict[str, str] = {
    RateData.rate: RateData.rate_obj,
    RateData.rate_data: RateData.rate_data_obj,
    RateData.branch_frac: RateData.branch_frac_obj,
}  # type: ignore


def rate_data(
    mech: Mechanism,
    reactants: Sequence[str],
    *,
    min_branch_frac: float | None = None,
) -> pl.DataFrame:
    """Display reactant branching fractions

    :param mech: Mechanism
    :param reactants: Reactants
    :param T_range: Temperature range
    :param P_range: Pressure range
    :param T: Temperature
    :param P: Pressure
    """
    mech = consumption_mechanism(mech, reactants=reactants, rev_mapping=REV_MAPPING)
    mech = combine_enantiomers(mech, rate_cols=RATE_COLS)
    rate_df = reaction.with_rate_object_columns(mech.reactions, OBJ_MAPPING)

    # Sort by median branching fraction
    median_branch_frac = c_.temp()
    rate_df = rate_df.with_columns(
        pl.col(RateData.branch_frac_obj)
        .map_elements(lambda o: np.nanmedian(o.rate.k_data), return_dtype=pl.Float64)
        .alias(median_branch_frac)
    )
    rate_df = rate_df.sort(pl.col(median_branch_frac), descending=True)
    if min_branch_frac is not None:
        rate_df = rate_df.filter(pl.col(median_branch_frac) > min_branch_frac)

    # Add labels
    rate_df = rate_df.with_columns(
        pl.col(RateData.products).list.join("+").alias(RateData.label)
    )

    # Add colors
    nrates = rate_df.height
    colors = list(
        itertools.islice(itertools.cycle(ac.util.plot.LINE_COLOR_CYCLE), nrates)
    )
    rate_df = rate_df.with_columns(pl.Series(RateData.color, colors, dtype=pl.String))
    return rate_df


def branching_fraction_chart(
    rate_df: pl.DataFrame,
    P_range: tuple[float, float],
    T: float,
    *,
    label: bool = True,
    total: bool = False,
) -> alt.Chart:
    objs = [r.rate for r in rate_df.get_column(RateData.branch_frac_obj).to_list()]  # type: ignore
    labels = rate_df.get_column(RateData.label).to_list()
    colors = rate_df.get_column(RateData.color).to_list()

    if total:
        objs.insert(0, functools.reduce(operator.add, objs))
        labels.insert(0, "total")
        colors.insert(0, ac.util.plot.Color.black)

    chart = ac.rate.data.display_p(
        objs,
        T=T,
        P_range=P_range,
        label=labels if label else None,
        color=colors,
        y_label="𝑓",
        y_unit="",
    )
    return chart


def rate_chart(
    rate_df: pl.DataFrame,
    T_range: tuple[float, float],
    P: float,
    *,
    label: bool = True,
    total: bool = False,
    T_drop: Sequence[float] = (),
    extra_rates: dict[str, ac.rate.RateFit] | None = None,
    extra_colors: dict[str, str] | None = None,
) -> alt.Chart:
    # Rates
    data_objs = [r.rate for r in rate_df.get_column(RateData.rate_data_obj).to_list()]  # type: ignore
    data_objs = [r.drop_temperatures(T_drop) for r in data_objs]
    fit_objs = [r.rate for r in rate_df.get_column(RateData.rate_obj).to_list()]  # type: ignore
    objs = [*data_objs, *fit_objs]
    labels = rate_df.get_column(RateData.label).to_list() * 2
    colors = rate_df.get_column(RateData.color).to_list() * 2
    if total:
        objs.insert(0, functools.reduce(operator.add, data_objs))
        labels.insert(0, "Total")
        colors.insert(0, ac.util.plot.Color.black)

    if extra_rates is not None:
        nrates = rate_df.height
        nextra = len(extra_rates)
        color_cycle = itertools.islice(
            itertools.cycle(ac.util.plot.LINE_COLOR_CYCLE), nrates, nrates + nextra
        )
        color_dct = (
            {label: next(color_cycle) for label in extra_rates}
            if extra_colors is None
            else extra_colors
        )
        for extra_label, extra_rate in extra_rates.items():
            objs.append(extra_rate)
            labels.append(extra_label)
            colors.append(color_dct[extra_label])

    chart = ac.rate.data.display(
        objs, T_range=T_range, P=P, label=labels if label else None, color=colors
    )
    return chart
