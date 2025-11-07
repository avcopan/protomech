import functools
import itertools
import operator
from collections.abc import Sequence

import altair as alt
import autochem as ac
import numpy as np
import polars as pl
from autochem import unit_
from autochem.unit_ import UNITS, Units, UnitsData
from autochem.util import plot

from automech import Mechanism, combine_enantiomers, consumption_mechanism, reaction
from automech.util import c_
from automech.util.pandera_ import Model


class RateData(Model):
    reactants: list[str]
    products: list[str]
    # Forward data:
    rate: pl.Struct
    rate_data: pl.Struct
    branch_frac: pl.Struct
    # Forward objects:
    rate_obj: pl.Object
    rate_data_obj: pl.Object
    branch_frac_obj: pl.Object
    # Reverse data:
    rev_rate: pl.Struct
    rev_rate_data: pl.Struct
    rev_branch_frac: pl.Struct
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
    colors: Sequence[str] | None = None,
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
    colors = colors or list(
        itertools.islice(itertools.cycle(ac.util.plot.LINE_COLOR_CYCLE), nrates)
    )
    rate_df = rate_df.with_columns(pl.Series(RateData.color, colors, dtype=pl.String))
    return rate_df


def branching_fraction_xy_data(
    rate_df: pl.DataFrame,
    P_range: tuple[float, float],
    T: float,
    *,
    total: bool = False,
    units: UnitsData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    units = UNITS if units is None else Units.model_validate(units)
    objs = [r.rate for r in rate_df.get_column(RateData.branch_frac_obj).to_list()]  # type: ignore
    if total:
        objs.insert(0, functools.reduce(operator.add, objs))

    x_data = np.linspace(*P_range, num=1000)
    y_data = []
    for obj in objs:
        x_data_, y_data_ = obj.plot_data(T=T, P=P_range, units=units)
        y_ = plot.transformed_spline_interpolator(x_data_, y_data_, x_trans=np.log10)
        y_data.append(y_(x_data))

    return x_data, np.asarray(y_data)


def branching_fraction_chart(
    rate_df: pl.DataFrame,
    P_range: tuple[float, float],
    T: float,
    *,
    total: bool = False,
    legend: bool = True,
    mark_kwargs: dict | None = None,
    units: UnitsData | None = None,
    y_range: tuple[float, float] | None = None,
) -> alt.Chart:
    units = UNITS if units is None else Units.model_validate(units)
    x_data, y_data = branching_fraction_xy_data(
        rate_df=rate_df, P_range=P_range, T=T, total=total, units=units
    )
    labels = rate_df.get_column(RateData.label).to_list()
    colors = rate_df.get_column(RateData.color).to_list()

    if total:
        labels.insert(0, "total")
        colors.insert(0, ac.util.plot.Color.black)

    x_unit = unit_.pretty_string(units.pressure)
    x_label = f"pressure ({x_unit})"
    y_label = "branching fraction"
    y_range = y_range or (0, 1)
    chart = plot.general(
        y_data=y_data,
        x_data=x_data,
        labels=labels,
        colors=colors,
        x_label=x_label,
        y_label=y_label,
        x_scale=plot.log_scale(P_range),
        x_axis=plot.log_scale_axis(P_range),
        y_scale=plot.regular_scale(y_range),
        y_axis=plot.regular_scale_axis(y_range),
        mark=plot.Mark.line,
        mark_kwargs=mark_kwargs,
        legend=legend,
    )
    return chart


def rate_xy_data(
    rate_df: pl.DataFrame,
    T_range: tuple[float, float],
    P: float,
    *,
    total: bool = False,
    units: UnitsData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    units = UNITS if units is None else Units.model_validate(units)
    # Rates
    objs = [r.rate for r in rate_df.get_column(RateData.rate_obj).to_list()]  # type: ignore
    x_data = np.linspace(*T_range, num=1000)
    y_data = []
    for obj in objs:
        y_data.append(obj(x_data, P, units=units))

    if total:
        y_data.insert(0, sum(y_data))

    y_data = np.where(np.greater(y_data, 0), y_data, np.nan)
    return x_data, y_data


def rate_chart(
    rate_df: pl.DataFrame,
    T_range: tuple[float, float],
    P: float,
    *,
    total: bool = False,
    legend: bool = True,
    mark_kwargs: dict | None = None,
    units: UnitsData | None = None,
    y_range: tuple[float, float] | None = None,
) -> alt.Chart:
    units = UNITS if units is None else Units.model_validate(units)
    x_data, y_data = rate_xy_data(
        rate_df=rate_df, T_range=T_range, P=P, total=total, units=units
    )
    labels = rate_df.get_column(RateData.label).to_list()
    colors = rate_df.get_column(RateData.color).to_list()
    if total:
        labels.insert(0, "Total")
        colors.insert(0, ac.util.plot.Color.black)

    y_data = np.where(np.greater(y_data, 0), y_data, np.nan)
    y_range = y_range or (np.nanmin(y_data), np.nanmax(y_data))

    x_unit = unit_.pretty_string(units.temperature)
    x_label = f"temperature ({x_unit})"
    obj = rate_df.select(pl.col(RateData.rate_obj).first()).item()
    y_unit = unit_.pretty_string(units.rate_constant(order=obj.rate.order))
    y_label = f"rate constant ({y_unit})"
    chart = plot.general(
        y_data=y_data,
        x_data=x_data,
        labels=labels,
        colors=colors,
        x_label=x_label,
        y_label=y_label,
        x_scale=plot.regular_scale(T_range),
        x_axis=plot.regular_scale_axis(T_range),
        y_scale=plot.log_scale(y_range),
        y_axis=plot.log_scale_axis(y_range),
        mark=plot.Mark.line,
        mark_kwargs=mark_kwargs,
        legend=legend,
    )
    return chart
