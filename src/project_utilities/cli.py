"""Project workflows."""

from pathlib import Path

import click

from . import workflow


@click.group()
def main():
    """Project workflows."""
    pass


@main.group("simulate")
def simulate():
    """Run simulation."""
    pass


@simulate.command("o2")
@click.argument("tag")
@click.argument("root_path")
@click.option("-e", "--gather_every", default=1, help="Run every n concentrations.")
@click.option(
    "-c", "--control", is_flag=True, default=False, help="Run the control model."
)
def simulate_o2(
    tag: str,
    root_path: str | Path,
    gather_every: int = 1,
    control: bool = False,
) -> None:
    """Simulate model (JSR).

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param temp_k: Temperature in K
    :param pres_atm: Pressure in atm
    :param tau_s: Residence time in s
    :param vol_cm3: Volume in cm^3
    :param gather_every: Gather every nth point
    """
    workflow.run_o2_simulation(
        tag=tag, root_path=root_path, gather_every=gather_every, control=control
    )


@simulate.command("t")
@click.argument("tag")
@click.argument("root_path")
@click.option("-e", "--gather_every", default=1, help="Run every n concentrations.")
def simulate_t(
    tag: str,
    root_path: str | Path,
    gather_every: int = 1,
) -> None:
    """Simulate model (JSR).

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param temp_k: Temperature in K
    :param pres_atm: Pressure in atm
    :param tau_s: Residence time in s
    :param vol_cm3: Volume in cm^3
    :param gather_every: Gather every nth point
    """
    workflow.run_t_simulation(tag=tag, root_path=root_path, gather_every=gather_every)


if __name__ == "__main__":
    main()
