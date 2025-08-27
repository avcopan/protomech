"""Cantera reactors."""

from collections.abc import Mapping
from numbers import Number

Solution = object  # cantera.Solution
ReactorNet = object  # cantera.ReactorNet


def jsr(
    model: Solution,
    temp: Number,
    pres: Number,
    tau: Number,
    vol: Number,
    conc: Mapping[str, Number],
) -> ReactorNet:
    """Run a jet-stirred reactor simulation.

    :param model: Chemical kinetics model
    :param temp: Temperature
    :param pres: Pressure
    :param tau: Residence time
    :param conc: Starting concentrations
    :return: Solved simulation reactor network
    """
    import cantera as ct

    # Use concentrations from the previous iteration to speed up convergence
    model.TPX = temp, pres, conc

    # Set up JSR: inlet -> flow control -> reactor -> pressure control -> exhaust
    reactor = ct.IdealGasReactor(model, energy="off", volume=vol)
    exhaust = ct.Reservoir(model)
    inlet = ct.Reservoir(model)
    ct.PressureController(
        upstream=reactor,
        downstream=exhaust,
        K=1e-3,
        primary=ct.MassFlowController(
            upstream=inlet,
            downstream=reactor,
            mdot=reactor.mass / tau,
        ),
    )
    reactor_net = ct.ReactorNet([reactor])
    reactor_net.advance_to_steady_state(max_steps=100000)
    return reactor
