"""Rate constant models."""

import abc
import warnings
from collections.abc import Mapping
from typing import Annotated, ClassVar

import altair
import more_itertools as mit
import numpy
import pint
import pydantic
import xarray
from numpy.polynomial import chebyshev
from numpy.typing import ArrayLike, NDArray
from pydantic import BeforeValidator
from pydantic_core import core_schema

from .. import unit_
from ..unit_ import UNITS, C, D, Dimension, UnitManager, Units, UnitsData, const
from ..util import chemkin, mess, plot
from ..util.type_ import Frozen, NDArray_, Scalable, Scalers, SubclassTyped
from . import blend
from .blend import BlendingFunction_


class Key:
    # Independent variables
    T = "T"
    P = "P"
    # Dependent variables
    k = "k"


class BaseRate(UnitManager, Frozen, Scalable, SubclassTyped, abc.ABC):
    """Abstract base class for rate constants."""

    order: int = 1

    @property
    def unit(self) -> pint.Unit:
        return UNITS.rate_constant(self.order)

    def __init__(self, units: UnitsData | None = None, **kwargs):
        super().__init__(units=units, **kwargs)

    @abc.abstractmethod
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant.

        :param T: Temperature(s)
        :param P: Pressure(s)
        :param units: Input units and desired output units
        :return: Value(s)
        """
        pass

    def process_input(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
    ) -> tuple[NDArray[numpy.float128], NDArray[numpy.float128]]:
        """Normalize rate constant input.

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Temperature(s) and pressure(s)
        """
        T = numpy.array(T, dtype=numpy.float128)
        P = numpy.array(P, dtype=numpy.float128)
        T, P = numpy.meshgrid(T, P)
        return T, P

    def process_output(
        self,
        kTP: ArrayLike,  # noqa: N803
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
    ) -> NDArray[numpy.float128]:
        """Normalize rate constant output, clipping unphyiscal negative values.

        :param ktp: Rate constant values
        :return: Rate constant values
        """
        kTP = numpy.reshape(kTP, numpy.shape(T) + numpy.shape(P))
        return numpy.where(numpy.less_equal(kTP, 0), numpy.nan, kTP)

    @property
    def _plot_mark(self) -> str:
        return plot.Mark.line

    def _plot_data(
        self,
        T_range: tuple[float, float] = (400, 1250),  # noqa: N803
        P: float = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """Display as an Arrhenius plot.

        :param T_range: Temperature range
        :param P: Pressure
        :param units: Units
        :return: Chart
        """
        T = numpy.linspace(*T_range, 1000)
        k = self(T=T, P=P, units=units)
        return T, k

    def display(
        self,
        T_range: tuple[float, float] = (400, 1250),  # noqa: N803
        P: float = 1,  # noqa: N803
        units: UnitsData | None = None,
        label: str | None = None,
        color: str | None = None,
        x_label: str = "1000/𝑇",  # noqa: RUF001
        y_label: str = "𝑘",
    ) -> altair.Chart:
        """Display as an Arrhenius plot.

        :param T_range: Temperature range
        :param P: Pressure
        :param units: Units
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :return: Chart
        """
        T, k = self._plot_data(T_range=T_range, P=P, units=units)
        return plot.arrhenius(
            ks=[k],
            T=T,
            order=self.order,
            units=units,
            labels=[label] if label else None,
            colors=[color] if color else None,
            x_label=x_label,
            y_label=y_label,
            mark=self._plot_mark,
        )


def nan_array_to_none(arr: ArrayLike | None) -> NDArray | None:
    """Replace an array of NaNs with None.

    :param arr: Array or None
    :return: Array or None
    """
    return None if arr is None or numpy.all(numpy.isnan(arr)) else arr


def negative_rates_to_none(arr: ArrayLike | None) -> NDArray | None:
    """Replace negative rates with None.

    :param arr: Array or None
    :return: Array or None
    """
    if arr is None:
        return None
    arr = numpy.array(arr, copy=True)
    arr[arr < 0] = numpy.nan
    return arr


class Rate(BaseRate):
    T: list[float]
    P: list[float]
    k_data: Annotated[NDArray_, BeforeValidator(negative_rates_to_none)]
    k_high: Annotated[
        list[float] | None,
        BeforeValidator(nan_array_to_none),
        BeforeValidator(negative_rates_to_none),
    ] = None

    # Private attributes
    type_: ClassVar[str] = "data"
    _scalers: ClassVar[Scalers] = {"k_data": numpy.multiply, "k_high": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "T": D.temperature,
        "P": D.pressure,
        "k_data": D.rate_constant,
        "k_high": D.rate_constant,
    }

    @property
    def data_array(self):
        """Return data as an xarray.DataArray."""
        P = self.P
        k_data = self.k_data
        if self.k_high is not None:
            P = numpy.append(self.P, numpy.inf)
            k_data = numpy.vstack((self.k_data, self.k_high))
        return xarray.DataArray(data=k_data, coords={Key.P: P, Key.T: self.T})

    @property
    def _plot_mark(self) -> str:
        return plot.Mark.point

    def _plot_data(
        self,
        T_range: tuple[float, float] = (400, 1250),  # noqa: N803
        P: float = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """Display as an Arrhenius plot.

        :param T_range: Temperature range
        :param P: Pressure
        :param units: Units
        :return: Chart
        """
        T_min, T_max = T_range
        k = self(T=self.T, P=P, units=units)
        (ix,) = numpy.where(
            numpy.greater_equal(self.T, T_min) & numpy.less_equal(self.T, T_max)
        )
        return numpy.take(self.T, ix), numpy.take(k, ix)

    @unit_.manage_units([D.temperature, D.pressure], D.rate_constant)
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant."""
        kTP: NDArray[numpy.float128] = self.data_array.sel(
            {Key.T: T, Key.P: P}, method="ffill"
        ).data
        return self.process_output(kTP, T, P)

    def __add__(self, other: "Rate") -> "Rate":
        """Add rates."""
        assert self.order == other.order, f"{self} !~ {other}"
        T, ixT1, ixT2 = numpy.intersect1d(self.T, other.T, return_indices=True)
        P, ixP1, ixP2 = numpy.intersect1d(self.P, other.P, return_indices=True)
        k_data1 = self.k_data[numpy.ix_(ixP1, ixT1)]
        k_data2 = other.k_data[numpy.ix_(ixP2, ixT2)]
        k_data = numpy.add(k_data1, k_data2)

        k_high = None
        if self.k_high is not None and other.k_high is not None:
            k_high1 = numpy.array(self.k_high)[ixT1]
            k_high2 = numpy.array(other.k_high)[ixT2]
            k_high = numpy.add(k_high1, k_high2)

        return self.__class__(order=self.order, T=T, P=P, k_data=k_data, k_high=k_high)

    def without_nan(self) -> "Rate":
        """Return a copy of the rate without temperatures giving NaNs.

        :return: Rate
        """
        k_data = self.k_data
        k_all = k_data if self.k_high is None else numpy.vstack((k_data, self.k_high))
        not_nan = numpy.all(numpy.isfinite(k_all), axis=0)
        return self.__class__(
            order=self.order,
            T=numpy.array(self.T)[not_nan],
            P=self.P,
            k_data=self.k_data[:, not_nan],
        )


class RateFit(BaseRate):
    """Abstract base class for parametrized rate constants."""

    efficiencies: dict[str, float] = pydantic.Field(default_factory=dict)

    @property
    def third_body(self) -> str | None:
        """Get third body."""
        eff = self.efficiencies

        if eff.get("M") == 1:
            return "M"

        return next((c for c, e in eff.items() if e == 1.0), None)

    @property
    def is_pressure_dependent(self) -> bool:
        """Determine if the rate is pressure dependent."""
        return True

    @pydantic.field_validator("efficiencies", mode="before")
    @classmethod
    def sanitize_efficiencies(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {k: v for k, v in value.items() if v is not None}
        return value


class ArrheniusRateFit(RateFit):
    A: float = 1.0
    b: float = 0.0
    E: float = 0.0

    # Private attributes
    type_: ClassVar[str] = "arrhenius"
    _scalers: ClassVar[Scalers] = {"A": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "A": D.rate_constant,
        "E": D.energy_per_substance,
    }

    @property
    def is_pressure_dependent(self) -> bool:
        """Determine if the rate is pressure dependent."""
        return False

    @unit_.manage_units([D.temperature, D.pressure], D.rate_constant)
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant."""
        T_, P_ = self.process_input(T, P)
        R = const.value(C.gas, UNITS)
        kTP = self.A * (T_**self.b) * numpy.exp(-self.E / (R * T_))
        return self.process_output(kTP, T, P)

    @classmethod
    @unit_.manage_units([D.temperature, D.rate_constant])
    def fit(
        cls,
        T: ArrayLike,  # noqa: N803
        k: ArrayLike,
        order: int = 1,
        units: UnitsData | None = None,
    ) -> "ArrheniusRateFit":
        """Fit data to Arrhenius rate fit.

        :param T: Temperatures
        :param k: Rates
        :return: Rate fit
        """
        T = numpy.array(T, dtype=numpy.float64)
        _1 = numpy.ones_like(T)

        R = unit_.const.value(C.gas, UNITS)
        M = numpy.column_stack([_1, numpy.log(T), -1 / (R * T)])
        v = numpy.log(k)

        ok = numpy.isfinite(v)
        M = M[ok, :]
        v = v[ok]

        (lnA, b, E), *_ = numpy.linalg.lstsq(M, v, rcond=1e-24)
        return cls(order=order, A=numpy.exp(lnA), b=b, E=E)


class FalloffRateFit(RateFit, abc.ABC):  # type: ignore[misc]
    A_high: float
    b_high: float
    E_high: float
    A_low: float
    b_low: float
    E_low: float
    function: BlendingFunction_
    activated: bool = False

    # Private attributes
    type_: ClassVar[str] = "falloff"
    _scalers: ClassVar[Scalers] = {"A_high": numpy.multiply, "A_low": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "A_high": D.rate_constant,
        "E_high": D.energy_per_substance,
        "A_low": D.rate_constant,
        "E_low": D.energy_per_substance,
    }

    @unit_.manage_units([D.temperature, D.pressure], D.rate_constant)
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant."""
        T_, P_ = self.process_input(T, P)
        P_r = self.effective_reduced_pressure(T_, P_)
        if self.activated:
            k_low, _ = self.arrhenius_functions
            kTP = k_low(T_) / (1 + P_r) * self.function(T_, P_r)
        else:
            _, k_high = self.arrhenius_functions
            kTP = k_high(T_) * P_r / (1 + P_r) * self.function(T_, P_r)
        return self.process_output(kTP, T, P)

    @property
    def arrhenius_functions(
        self,
    ) -> tuple[ArrheniusRateFit, ArrheniusRateFit]:
        k_low = ArrheniusRateFit(
            A=self.A_high, b=self.b_high, E=self.E_high, order=self.order
        )
        k_high = ArrheniusRateFit(
            A=self.A_high, b=self.b_high, E=self.E_high, order=self.order
        )
        return k_low, k_high

    def effective_concentration(
        self,
        T: NDArray[numpy.float128],  # noqa: N803
        P: NDArray[numpy.float128],  # noqa: N803
    ) -> NDArray[numpy.float128]:
        """Get effective concentration(s) from temperature(s) and pressure(s).

        effective [M] = P / R T  (ideal gas law)

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Effective concentration(s)
        """
        # Evaluate, using pint to handle units
        R_ = const.quantity(C.gas)
        T_ = pint.Quantity(T, UNITS.temperature)
        P_ = pint.Quantity(P, UNITS.pressure)
        m_ = P_ / (R_ * T_)

        # Return value in concentration units
        return m_.m_as(UNITS.concentration)

    def effective_reduced_pressure(
        self,
        T: NDArray[numpy.float128],  # noqa: N803
        P: NDArray[numpy.float128],  # noqa: N803
    ) -> NDArray[numpy.float128]:
        """Get effective concentration(s) from temperature(s) and pressure(s).

        effective P_r = k_low [M] / k_high  (ideal gas law)

        :param T: Temperature(s)
        :param P: Pressure(s)
        :return: Effective reduced pressure(s)
        """
        m_eff = self.effective_concentration(T, P)
        k_low, k_high = self.arrhenius_functions
        return k_low(T) * m_eff / k_high(T)


class PlogRateFit(RateFit):
    As: list[float]
    bs: list[float]
    Es: list[float]
    Ps: list[float]

    # Private attributes
    type_: ClassVar[str] = "plog"
    _scalers: ClassVar[Scalers] = {"As": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "As": D.rate_constant,
        "Es": D.energy_per_substance,
        "Ps": D.pressure,
    }

    @unit_.manage_units([D.temperature, D.pressure], D.rate_constant)
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant for a single pressure."""
        T_, P_ = self.process_input(T, P)
        P0 = self.nearest_pressure(P_, which=0)
        P1 = self.nearest_pressure(P_, which=1)
        kT0 = self.nearest_arrhenius_values(T_, P_, which=0)
        kT1 = self.nearest_arrhenius_values(T_, P_, which=1)

        # Evaluate intermediate pressures
        log_P, log_P0, log_P1 = map(numpy.log, (P_, P0, P1))
        kTP = kT0 + (kT1 - kT0) * (log_P - log_P0) / (log_P1 - log_P0)

        # Evaluate on-boundary pressures (needed to fill in last pressure value)
        if numpy.ndim(P_) > 0:
            kTP[..., numpy.equal(P_, P0)] = kT0[..., numpy.equal(P_, P0)]
        elif P_ == P0:
            kTP = kT0

        return self.process_output(kTP, T, P)

    @property
    def arrhenius_functions(self) -> list[ArrheniusRateFit]:
        return [
            ArrheniusRateFit(A=A, b=b, E=E, order=self.order)
            for A, b, E in zip(self.As, self.bs, self.Es, strict=True)
        ]

    @property
    def pressures(self) -> NDArray[numpy.float128]:
        """Pressures."""
        return numpy.array(self.Ps, dtype=numpy.float128)

    @property
    def pressure_indices(self) -> list[int]:
        """Pressure indices."""
        return list(range(self.pressures.shape[0]))

    def nearest_arrhenius_values(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.float128]:
        """Get nearest lower or higher pressure.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest function(s)
        """
        iP = self.nearest_index(P, which=which)
        kTs = [k(T) for k in self.arrhenius_functions]
        return numpy.where(
            numpy.isin(iP, self.pressure_indices),
            numpy.choose(iP, kTs, mode="clip"),
            numpy.nan,
        )

    def nearest_pressure(
        self,
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.float128]:
        """Get nearest lower or higher pressure.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest pressure(s)
        """
        iP = self.nearest_index(P, which=which)
        ps = self.pressures
        return numpy.where(
            numpy.isin(iP, self.pressure_indices),
            numpy.take(ps, iP, mode="clip"),
            numpy.nan,
        )

    def nearest_index(
        self,
        P: ArrayLike,  # noqa: N803
        which: int = 0,
    ) -> NDArray[numpy.int_]:
        """Get nearest lower or higher index.

        :param P: Pressure(s)
        :param which: 0=lower, 1=higher
        :return: Nearest index (indices)
        """
        return numpy.searchsorted(self.Ps, P, side="right") - 1 + which

    @classmethod
    @unit_.manage_units([D.temperature, D.rate_constant])
    def fit(
        cls,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike,  # noqa: N803
        k_data: ArrayLike,
        k_high: ArrayLike | None = None,
        order: int = 1,
        units: UnitsData | None = None,
    ) -> "PlogRateFit":
        """Fit data to Plog rate fit.

        :param T: Temperatures
        :param P: Temperatures
        :param k_data: Finite pressure rates
        :param k_high: High-pressure-limit rates
        :return: Rate fit
        """
        k_data_fits = [
            ArrheniusRateFit.fit(T=T, k=k, order=order, units=units) for k in k_data
        ]
        k_high_fit = None
        if k_high is not None:
            k_high_fit = ArrheniusRateFit.fit(T=T, k=k_high, order=order, units=units)
            warnings.warn(f"Currently not fitting high-pressure limit {k_high_fit}")

        return cls(
            order=order,
            As=[f.A for f in k_data_fits],
            bs=[f.b for f in k_data_fits],
            Es=[f.E for f in k_data_fits],
            Ps=P,
        )


class ChebRateFit(RateFit):
    coeffs: NDArray_
    T_range: tuple[float, float]
    P_range: tuple[float, float]

    # Private attributes
    type_: ClassVar[str] = "cheb"
    _scalers: ClassVar[Scalers] = {"coeffs": numpy.multiply}
    _dimensions: ClassVar[dict[str, Dimension]] = {
        "coeffs": D.rate_constant,
        "T_range": D.temperature,
        "P_range": D.pressure,
    }

    @unit_.manage_units([D.temperature, D.pressure], D.rate_constant)
    def __call__(
        self,
        T: ArrayLike,  # noqa: N803
        P: ArrayLike = 1,  # noqa: N803
        units: UnitsData | None = None,
    ) -> NDArray[numpy.float128]:
        """Evaluate rate constant for a single pressure."""
        # Skip input processing, since chebgrid2d automatically forms the grid
        T0, T1 = self.T_range
        P0, P1 = self.P_range

        inv_ = numpy.reciprocal
        log_ = numpy.log10

        T_r = (2 * inv_(T) - inv_(T0) - inv_(T1)) / (inv_(T1) - inv_(T0))
        P_r = (2 * log_(P) - log_(P0) - log_(P1)) / (log_(P1) - log_(P0))

        # AVC: I don't understand why I need to transpose the coefficient matrix here
        kTP = chebyshev.chebgrid2d(T_r, P_r, self.coeffs.T)
        return self.process_output(kTP, T, P)


Rate_ = Annotated[
    pydantic.SkipValidation[BaseRate],
    pydantic.BeforeValidator(lambda x: BaseRate.model_validate(x)),
    pydantic.PlainSerializer(lambda x: BaseRate.model_validate(x).model_dump()),
    pydantic.GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(pydantic.BaseModel))
    ),
]


# Conversions
def chemkin_string(rate_const: RateFit, eq_width: int = 0) -> str:
    """Write Chemkin rate to a string.

    :param rate_const: Rate constant
    :param eq_width: Equation width for alignment
    :return: Chemkin rate string
    """
    # Generate dummy head line
    head_line = chemkin.write_numbers([1.0, 0.0, 0.0])

    # Calculate the total width of the top line for alignment
    head_width = eq_width + len(head_line) + 1

    # Generate auxiliary lines and replace head line, if appropriate
    match rate_const:
        case ArrheniusRateFit():
            head_params = [rate_const.A, rate_const.b, rate_const.E]
            head_line = chemkin.write_numbers(head_params)
            aux_lines = []
        case FalloffRateFit(activated=False):
            high_params = [rate_const.A_high, rate_const.b_high, rate_const.E_high]
            low_params = [rate_const.A_low, rate_const.b_low, rate_const.E_low]
            head_line = chemkin.write_numbers(high_params)
            aux_lines = [chemkin.write_aux("LOW", low_params, head_width=head_width)]
        case FalloffRateFit(activated=True):
            high_params = [rate_const.A_high, rate_const.b_high, rate_const.E_high]
            low_params = [rate_const.A_low, rate_const.b_low, rate_const.E_low]
            head_line = chemkin.write_numbers(low_params)
            aux_lines = [chemkin.write_aux("HIGH", high_params, head_width=head_width)]
        case PlogRateFit():
            plog_params = [rate_const.Ps, rate_const.As, rate_const.bs, rate_const.Es]
            aux_lines = [
                chemkin.write_aux("PLOG", row) for row in zip(*plog_params, strict=True)
            ]
        case ChebRateFit():
            shape = numpy.shape(rate_const.coeffs)
            aux_lines = [
                chemkin.write_aux("TCHEB", rate_const.T_range, head_width=head_width),
                chemkin.write_aux("PCHEB", rate_const.P_range, head_width=head_width),
                chemkin.write_aux("CHEB", shape, head_width=head_width, as_int=True),
                *(
                    chemkin.write_aux("CHEB", row, head_width=head_width)
                    for row in rate_const.coeffs.tolist()
                ),
            ]
        case _:
            raise ValueError(
                f"Rate constant has unknown type {type(rate_const)}:\n{rate_const}"
            )

    if isinstance(rate_const, FalloffRateFit):
        aux_lines.extend(
            blend.chemkin_aux_lines(rate_const.function, head_width=head_width)
        )

    eff_line = chemkin.write_efficiencies(
        rate_const.efficiencies, third_body=rate_const.third_body
    )
    if eff_line is not None:
        aux_lines.append(eff_line)

    lines = [head_line, *aux_lines]
    return "\n".join(lines)


def from_mess_channel_output(mess_chan_out: str, order: int) -> Rate:
    """Extract rate data from MESS output.

    :param mess_chan_out: MESS output channel string
    :param order: Order
    :return: Rate data
    """
    res = mess.parse_output_channel(mess_chan_out)
    return from_mess_channel_output_parse_results(res, order=order)


# Parse helpers
def from_mess_channel_output_parse_results(
    res: mess.MessOutputChannelParseResults, order: int
) -> Rate:
    """Extract rate data from MESS output parse results.

    :param res: MESS output parse results
    :param order: Order
    :return: Rate data
    """
    return Rate(
        order=order,
        T=res.T,
        P=res.P,
        k_data=res.k_data,
        k_high=res.k_high,
        units={"substance": "molec"},
    )


def from_chemkin_parse_results(
    res: chemkin.ChemkinRateParseResults, units: UnitsData | None = None
) -> RateFit:
    """Extract rate data from Chemkin parse results.

    Chemkin parse results are modified in-place

    :param res: Chemkin rate parse results
    :return: Rate data
    """
    # Extract efficiencies
    efficiencies = res.efficiencies.copy()
    res.efficiencies.clear()

    # Determine reaction order
    order = len(res.reactants) + bool(efficiencies)

    # Determine units
    units = Units() if units is None else Units.model_validate(units)

    if "CHEB" in res.aux_numbers:
        # Read coefficients
        cheb: list[float] = res.aux_numbers.pop("CHEB")
        shape = tuple(map(int, cheb[:2]))
        coeffs = numpy.reshape(cheb[2:], shape)
        # Read ranges
        t_range = res.aux_numbers.pop("TCHEB")
        p_range = res.aux_numbers.pop("PCHEB")
        return ChebRateFit(
            coeffs=coeffs,
            T_range=t_range,
            P_range=p_range,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "PLOG" in res.aux_numbers:
        ps, As, bs, Es = zip(
            *mit.chunked(res.aux_numbers.pop("PLOG"), 4, strict=True), strict=True
        )
        return PlogRateFit(
            As=As,
            bs=bs,
            Es=Es,
            Ps=ps,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "LOW" in res.aux_numbers:
        A_high, b_high, E_high = res.arrhenius
        A_low, b_low, E_low = res.aux_numbers.pop("LOW")
        function = blend.from_chemkin_parse_results(res)
        return FalloffRateFit(
            A_low=A_low,
            b_low=b_low,
            E_low=E_low,
            A_high=A_high,
            b_high=b_high,
            E_high=E_high,
            function=function,
            activated=False,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    if "HIGH" in res.aux_numbers:
        A_low, b_low, E_low = res.arrhenius
        A_high, b_high, E_high = res.aux_numbers.pop("HIGH")
        function = blend.from_chemkin_parse_results(res)
        return FalloffRateFit(
            A_low=A_low,
            b_low=b_low,
            E_low=E_low,
            A_high=A_high,
            b_high=b_high,
            E_high=E_high,
            function=function,
            activated=True,
            efficiencies=efficiencies,
            order=order,
            units=units,
        )

    A, b, E = res.arrhenius
    return ArrheniusRateFit(
        A=A, b=b, E=E, efficiencies=efficiencies, order=order, units=units
    )
