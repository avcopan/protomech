"""Project workflows."""

import functools
import signal
import time
from collections.abc import Sequence
from numbers import Number
from pathlib import Path

import altair
import cantera
import polars as pl
from cantera.ck2yaml import Parser

import automech
from automech import Mechanism
from automech.reaction import Reaction, ReactionRate, ReactionRateType, ReactionSorted
from automech.species import Species
from automech.util import c_
from protomech import mess

from . import p_, reactors
from .util import previous_tag, previous_tags


# Workflows
def write_parent_mechanism(mech: Mechanism, root_path: str | Path) -> None:
    """Write parent mechanism.

    :param mech: Parent mechanism
    :param root_path: Project root directory
    """
    mech_path = p_.parent_mechanism(ext="json", path=p_.data(root_path))
    return automech.io.write(mech, mech_path)


def read_parent_mechanism(root_path: str | Path) -> Mechanism:
    """Read parent mechanism.

    :param root_path: Project root directory
    :return: Parent mechanism
    """
    mech_path = p_.parent_mechanism(ext="json", path=p_.data(root_path))
    return automech.io.read(mech_path)


def previous_version_species(tag: str, root_path: str | Path) -> Mechanism:
    """Generate a base mechanism with species from previous version.

    Includes only the used species in the previous mechanism.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :return: Mechanism with used species in previous mechanism
    """
    tag0 = previous_tag(tag)
    gen_mech0 = automech.io.read(
        p_.generated_mechanism(tag0, "json", path=p_.data(root_path))
    )
    spc_mech0 = automech.without_unused_species(gen_mech0)
    spc_mech0 = automech.without_reactions(spc_mech0)
    return spc_mech0


def expand_stereo(
    mech: Mechanism,
    tag: str,
    root_path: str | Path,
    enant: bool = True,
) -> tuple[Mechanism, Mechanism, Mechanism]:
    """Prepare mechanism for calculation.

    :param mech: Mechanism
    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param enant: Whether to include all enantiomers
    :param fake_sort: Whether to do a fake sort, splitting up all reactions
    """
    gen_mech = automech.drop_duplicate_reactions(mech)

    # Expand and sort
    print("\nExpanding stereochemistry...")
    ste_mech, err_mech = automech.expand_stereo(
        gen_mech, enant=enant, distinct_ts=False
    )

    # Display
    print("\nStereoexpansion errors:")
    automech.display_reactions(err_mech)

    return ste_mech, err_mech, gen_mech


def update_previous_version(
    gen_mech: Mechanism, ste_mech, tag: str, root_path: str | Path
) -> Mechanism:
    """Update previous version with new species/reactions.

    :param gen_mech: Generated mechanism
    :param ste_mech: Stereo-expanded mechanism
    :param tag: Mechanism tag
    :param root_path: Project root directory
    :return: Mechanism with used species in previous mechanism
    """
    tag0 = previous_tag(tag)
    gen_mech0 = automech.io.read(
        p_.generated_mechanism(tag0, "json", path=p_.data(root_path))
    )
    ste_mech0 = automech.io.read(
        p_.stereo_mechanism(tag0, "json", path=p_.data(root_path))
    )
    gen_mech = automech.update(gen_mech0, gen_mech)
    ste_mech = automech.update(ste_mech0, ste_mech)
    gen_mech = automech.drop_duplicate_reactions(gen_mech)
    ste_mech = automech.drop_duplicate_reactions(ste_mech)
    ste_mech = automech.without_sort_data(ste_mech)
    return gen_mech, ste_mech


def augment_calculation(
    gen_mech: Mechanism,
    ste_mech: Mechanism,
    tag: str,
    root_path: str | Path,
) -> None:
    """Augment an existing mechanism with new species/reactions.

    :param mech: Mechanism
    :param tag: Mechanism tag
    :param root_path: Project root directory
    """
    gen_mech = automech.drop_duplicate_reactions(gen_mech)
    ste_mech = automech.drop_duplicate_reactions(ste_mech)

    # Read in the existing mechanism
    gen_mech0 = automech.io.read(
        p_.generated_mechanism(tag, "json", path=p_.data(root_path))
    )
    ste_mech0 = automech.io.read(
        p_.stereo_mechanism(tag, "json", path=p_.data(root_path))
    )

    # Differentiate reactions
    gen_mech_ = automech.reaction_difference(
        gen_mech, gen_mech0, reversible=True, stereo=False, drop_species=True
    )
    ste_mech_ = automech.reaction_difference(
        ste_mech, ste_mech0, reversible=True, stereo=True, drop_species=True
    )

    # Differentiate species
    gen_spc_ = automech.species.difference(
        gen_mech.species, gen_mech0.species, stereo=False
    )
    ste_spc_ = automech.species.difference(
        ste_mech.species, ste_mech0.species, stereo=True
    )

    # Sort
    offset = max(ste_mech0.reactions.get_column(ReactionSorted.pes, default=[0]))
    ste_mech_ = automech.with_fake_sort_data(ste_mech_, offset=offset)

    # Augment
    gen_mech.reactions = pl.concat(
        [gen_mech0.reactions, gen_mech_.reactions], how="diagonal_relaxed"
    )
    gen_mech.species = pl.concat([gen_mech0.species, gen_spc_], how="diagonal_relaxed")
    ste_mech.reactions = pl.concat(
        [ste_mech0.reactions, ste_mech_.reactions], how="diagonal_relaxed"
    )
    ste_mech.species = pl.concat([ste_mech0.species, ste_spc_], how="diagonal_relaxed")

    # Write
    print("\nWriting mechanism...")
    gen_path = p_.generated_mechanism(tag, ext="json", path=p_.data(root_path))
    ste_path = p_.stereo_mechanism(tag, ext="json", path=p_.data(root_path))
    ste_rxn_path = p_.stereo_mechanism(tag, ext="dat", path=p_.mechanalyzer(root_path))
    ste_spc_path = p_.stereo_mechanism(tag, ext="csv", path=p_.mechanalyzer(root_path))
    print(gen_path)
    automech.io.write(gen_mech, gen_path)
    print(ste_path)
    automech.io.write(ste_mech, ste_path)
    print(ste_rxn_path)
    print(ste_spc_path)
    automech.io.mechanalyzer.write.mechanism(
        ste_mech, rxn_out=ste_rxn_path, spc_out=ste_spc_path
    )


def prepare_calculation(
    gen_mech: Mechanism,
    ste_mech: Mechanism,
    tag: str,
    root_path: str | Path,
    fake_sort: bool = False,
) -> None:
    """Prepare mechanism for calculation.

    :param gen_mech: Generated mechanism
    :param ste_mech: Stereo-expanded mechanism
    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param enant: Whether to include all enantiomers
    :param fake_sort: Whether to do a fake sort, splitting up all reactions
    """
    gen_mech = automech.drop_duplicate_reactions(gen_mech)
    ste_mech = automech.drop_duplicate_reactions(ste_mech)
    gen_mech = automech.with_sorted_reagents(gen_mech)
    ste_mech = automech.with_sorted_reagents(ste_mech)

    # Sort
    print("\nSorting mechanism...")
    sorter_ = automech.with_fake_sort_data if fake_sort else automech.with_sort_data
    ste_mech = sorter_(ste_mech)

    # Write
    print("\nWriting mechanism...")
    gen_path = p_.generated_mechanism(tag, ext="json", path=p_.data(root_path))
    ste_path = p_.stereo_mechanism(tag, ext="json", path=p_.data(root_path))
    ste_rxn_path = p_.stereo_mechanism(tag, ext="dat", path=p_.mechanalyzer(root_path))
    ste_spc_path = p_.stereo_mechanism(tag, ext="csv", path=p_.mechanalyzer(root_path))
    print(gen_path)
    automech.io.write(gen_mech, gen_path)
    print(ste_path)
    automech.io.write(ste_mech, ste_path)
    print(ste_rxn_path)
    print(ste_spc_path)
    automech.io.mechanalyzer.write.mechanism(
        ste_mech, rxn_out=ste_rxn_path, spc_out=ste_spc_path
    )


def prepare_comparison(tag: str, root_path: str | Path) -> None:
    """Prepare comparisons to the parent mechanism.
    Update previous version with new species/reactions.

    :param gen_mech: Generated mechanism
    :param ste_mech: Stereo-expanded mechanism
    :param tag: Mechanism tag
    :param root_path: Project root directory
    :return: Mechanism with used species in previous mechanism
    """
    print("\nReading parent mechanism...")
    par_mech = read_parent_mechanism(root_path)

    print("\nReading stereo mechanism...")
    ste_path = p_.stereo_mechanism(tag, ext="json", path=p_.data(root_path))
    print(ste_path)
    ste_mech = automech.io.read(ste_path)

    # Add temp prefix to reactants, products, orig_reactants, orig_products
    pre = c_.temp()
    cols = [Reaction.reactants, Reaction.products]
    cols0 = c_.orig(cols)
    pre_cols = c_.prefix(cols, pre=pre)
    pre_cols0 = c_.prefix(cols0, pre=pre)
    to_pre = c_.to_([*cols, *cols0], pre=pre)
    ste_rxn_df = ste_mech.reactions.rename(to_pre)

    # Copy prefixed orig_reactants, orig_products to reactants, products
    to_arg = dict(zip(pre_cols0, cols, strict=True))
    ste_rxn_df = ste_rxn_df.with_columns(
        pl.col(c0).alias(c) for c0, c in to_arg.items()
    )

    # Left-update to pull in data and directions
    ste_rxn_df = automech.reaction.left_update(ste_rxn_df, par_mech.reactions)

    # Move the results to the orig_ columns, since they are missing stereo
    ste_rxn_df = ste_rxn_df.rename(c_.to_orig(cols))

    # Determine final columns with stereo by comparison to see if the direction
    # needs to be reversed
    match = pl.when(
        (pl.col(cols0[0]) == pl.col(pre_cols0[0]))
        & (pl.col(cols0[1]) == pl.col(pre_cols0[1]))
    )
    ste_rxn_df = ste_rxn_df.with_columns(
        match.then(pre_cols[0]).otherwise(pre_cols[1]).alias(cols[0])
    )
    ste_rxn_df = ste_rxn_df.with_columns(
        match.then(pre_cols[1]).otherwise(pre_cols[0]).alias(cols[1])
    )
    ste_rxn_df = ste_rxn_df.drop(to_pre.values())

    # Replace the original reactions dataframe
    comp_mech = ste_mech.model_copy()
    comp_mech.reactions = ste_rxn_df

    # Write
    print("\nWriting mechanism...")
    comp_path = p_.comparison_mechanism(tag, ext="json", path=p_.data(root_path))
    print(comp_path)
    automech.io.write(comp_mech, comp_path)


def gather_statistics(tag: str, root_path: str | Path) -> None:
    """Gather statistics on the number of species/reactions.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    """
    par_mech = automech.io.read(p_.parent_mechanism("json", path=p_.data(root_path)))
    gen_mech = automech.io.read(
        p_.generated_mechanism(tag, "json", path=p_.data(root_path))
    )
    ste_mech = automech.io.read(
        p_.stereo_mechanism(tag, "json", path=p_.data(root_path))
    )
    gen_mech.reactions = automech.reaction.with_dummy_rates(gen_mech.reactions)
    ste_mech.reactions = automech.reaction.with_dummy_rates(ste_mech.reactions)
    gen_diff_mech = automech.full_difference(
        gen_mech, par_mech, reversible=True, stereo=False
    )
    ste_diff_mech = automech.full_difference(
        ste_mech, par_mech, reversible=True, stereo=False
    )

    labels = ["Parent", "Generated", "Stereo", "Generated - Parent", "Stereo - Parent"]
    tables = [par_mech, gen_mech, ste_mech, gen_diff_mech, ste_diff_mech]
    data = {
        "Mechanism": labels,
        "Species Count": list(map(automech.species_count, tables)),
        "Reaction Count": list(map(automech.reaction_count, tables)),
    }

    # Print added species and reactions
    print("New species and reactions:\n")
    diff_ckin_str = automech.io.chemkin.write.mechanism(
        gen_diff_mech, fill_rates=True, elem=False, therm=False
    )
    print(diff_ckin_str)
    print("\n\n")

    # Print statistics summary
    print("Statistics summary:\n")
    stat_df = pl.DataFrame(data)
    print(stat_df)


def prepare_simulation(
    tag: str,
    stoich: str,
    therm_tag: str,
    root_path: str | Path,
    *,
    clear_nodes: Sequence[str] = (),
    clear_edges: Sequence[tuple[str, str]] = (),
    T_range: tuple[float, float] = (300, 1200),
    T_vals=(600, 700, 800, 900, 1000, 1100, 1200),
    P_vals=(0.1, 1, 10, 100),
    T_drop=(300, 400),
    A_fill=1e-20,
) -> None:
    """Prepare simulation from MESS output.

    :param tag: Mechanism tag
    :param stoich: Stoichiometry
    :param tag0: Source mechanism tag
    :param root_path: Project root directory
    """
    print(f"Reading in source mechanism and selecting {stoich} PES...")
    mech_path = p_.stereo_mechanism(tag, "json", p_.data(root_path))
    mech = automech.io.read(mech_path)
    mech.reactions = automech.reaction.select_pes(mech.reactions, stoich)
    mech = automech.without_unused_species(mech)

    print("Adding thermochemistry data...")
    therm_path = p_.ckin(root_path, therm_tag)
    therm_file = therm_path / "all_therm.ckin_1"
    mech = automech.io.chemkin.update.thermo(mech, therm_file)

    print("Post-processing rate data...")
    print(" - Reading in MESS input...")
    mess_path = p_.mess_final(root_path) / stoich
    mess_inp = mess_path / "mess.inp"
    surf0 = mess.surf.from_mess_input(mess_inp)

    print(" - Reading in MESS output...")
    mess_out = mess_path / "rate.out"
    surf0 = mess.surf.with_mess_output_rates(surf0, mess_out=mess_out)

    print(" - Integrating over fake nodes...")
    surf = mess.surf.absorb_fake_nodes(surf0)

    print(" - Enforcing equal enantiomer rates...")
    surf = mess.surf.enforce_equal_enantiomer_rates(surf, tol=0.1)

    if clear_nodes:
        print(f" - Clearing rates for nodes {clear_nodes}...")
        clear_node_keys = [
            mess.surf.node_key_from_label(surf, label) for label in clear_nodes
        ]
        surf = mess.surf.clear_node_rates(surf, keys=clear_node_keys)

    if clear_edges:
        print(f" - Clearing rates for edges {clear_edges}...")
        clear_edge_keys = [
            mess.surf.edge_key_from_labels(surf, labels) for labels in clear_edges
        ]
        surf = mess.surf.clear_edge_rates(surf, keys=clear_edge_keys)

    print(" - Clearing unfittable pressure ranges...")
    surf = mess.surf.clear_unfittable_pressure_ranges(surf)

    print(" - Removing unfittable well-skipping rates...")
    unfit_skip_rate_keys = mess.surf.unfittable_rate_keys(surf, direct=False)
    surf = mess.surf.remove_well_skipping_rates(surf, unfit_skip_rate_keys)

    print(" - Removing irrelevant well-skipping rates...")
    irrel_skip_rate_keys = mess.surf.irrelevant_rate_keys(
        surf, T=T_vals, P=P_vals, direct=False, min_branch_frac=0.01
    )
    surf = mess.surf.remove_well_skipping_rates(surf, irrel_skip_rate_keys)

    print(" - Removing isolated nodes...")
    surf = mess.surf.remove_isolates(surf)

    print(" - Checking for bad pressure-independence well-skipping rates...")
    bad_skip_rate_keys = mess.surf.bad_pressure_independence_rate_keys(
        surf, T_vals=T_vals, direct=False
    )
    if bad_skip_rate_keys:
        msg = f"Bad pressure-independent well-skipping rates: {bad_skip_rate_keys}"
        raise ValueError(msg)

    print(" - Checking for non-irrelevant bad pressure-independence direct rates...")
    all_bad_rate_keys = mess.surf.bad_pressure_independence_rate_keys(
        surf, T_vals=T_vals, well_skipping=False
    )
    irrel_rate_keys = mess.surf.irrelevant_rate_keys(
        surf, T=T_vals, P=P_vals, well_skipping=False, min_branch_frac=0.01
    )
    bad_rate_keys = set(all_bad_rate_keys) - set(irrel_rate_keys)
    if bad_rate_keys:
        msg = f"Bad pressure-independent well-skipping rates: {bad_skip_rate_keys}"
        raise ValueError(msg)

    print(" - Clearing bad pressure-independence direct rates that are irrelevant...")
    irrel_bad_rate_keys = set(all_bad_rate_keys) & set(irrel_rate_keys)
    surf = mess.surf.clear_rates(surf, irrel_bad_rate_keys)

    print(" - Determining irrelevant pressures...")
    irrel_pressure_dct = mess.surf.irrelevant_rate_pressures(
        surf, T=T_vals, P=P_vals, min_branch_frac=0.01
    )

    print(" - Fitting rates...")
    surf = mess.surf.fit_rates(
        surf,
        T_drop=T_drop,
        A_fill=A_fill,
        bad_fit="raise",
        bad_fit_fill_pressures_dct=irrel_pressure_dct,
    )

    print("Adding rate data...")
    print(" - Matching rates to reaction directions...")
    surf = mess.surf.match_rate_directions(surf, mech)

    print(" - Building direct reactions DataFrame...")
    rate_data = []
    for rate_key in mess.surf.rate_keys(surf, well_skipping=False):
        key1, key2 = rate_key
        node1 = mess.surf.node_object(surf, key1)
        node2 = mess.surf.node_object(surf, key2)
        rate = surf.rate_fits[rate_key]
        rate_data.append(
            {
                Reaction.reactants: node1.names_list,  # type: ignore
                Reaction.products: node2.names_list,  # type: ignore
                ReactionRate.reversible: True,
                ReactionRate.rate: rate.model_dump(),
                ReactionRateType.well_skipping: False,
            }
        )

    rate_df = pl.DataFrame(rate_data, infer_schema_length=None)
    print(rate_df)


def prepare_simulation_old(tag: str, root_path: str | Path) -> None:
    """Read calculation results and prepare simulation.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    """
    # Read mechanisms
    print("\nReading mechanisms...")
    par_mech = automech.io.read(p_.parent_mechanism("json", path=p_.data(root_path)))
    sub_mech = automech.io.read(
        p_.stereo_mechanism(tag, "json", path=p_.data(root_path))
    )

    # Add calculated thermo to mechanism object
    print("\nAdding calculated thermo...")
    ckin_path = p_.ckin(root_path, tag)
    *_, therm_file = ckin_path.glob("all_therm.ckin*")
    cal_sub_mech = automech.io.chemkin.update.thermo(sub_mech, therm_file)

    # Add calculated rates to mechanism object (use units of parent)
    print("\nAdding calculated rates...")
    rate_files = list(ckin_path.glob("*.ckin"))
    cal_sub_mech = automech.io.chemkin.update.rates(cal_sub_mech, rate_files)

    # Merge updated rates and thermo into parent mechanism
    print("\nExpanding and updating parent...")
    con_mech = automech.expand_parent_stereo(par_mech, cal_sub_mech)
    cal_mech = automech.update(con_mech, cal_sub_mech)

    # Write
    print("\nWriting mechanism to JSON...")
    automech.io.write(
        cal_sub_mech, p_.calculated_mechanism(tag, "json", path=p_.data(root_path))
    )
    automech.io.write(
        con_mech, p_.full_control_mechanism(tag, "json", path=p_.data(root_path))
    )
    automech.io.write(
        cal_mech, p_.full_calculated_mechanism(tag, "json", path=p_.data(root_path))
    )

    print("\nWriting mechanism to Chemkin...")
    con_path = p_.full_control_mechanism(tag, "dat", path=p_.chemkin(root_path))
    print(f"Control: {con_path}")
    automech.io.chemkin.write.mechanism(con_mech, con_path)
    cal_path = p_.full_calculated_mechanism(tag, "dat", path=p_.chemkin(root_path))
    print(f"Calculated: {cal_path}")
    automech.io.chemkin.write.mechanism(cal_mech, cal_path)

    print("\nConverting ChemKin mechanism to Cantera YAML...")
    Parser.convert_mech(
        con_path,
        out_name=p_.full_control_mechanism(tag, "yaml", path=p_.cantera(root_path)),
    )
    Parser.convert_mech(
        cal_path,
        out_name=p_.full_calculated_mechanism(tag, "yaml", path=p_.cantera(root_path)),
    )

    print("\nValidating Cantera model...")
    cantera.Solution(
        p_.full_calculated_mechanism(tag, "yaml", path=p_.cantera(root_path))
    )


def plot_rates(tag: str, root_path: str | Path) -> None:
    """Plot calculated rates.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    """
    # Read mechanisms
    print("\nReading mechanisms...")
    par_mech = automech.io.read(p_.parent_mechanism("json", path=p_.data(root_path)))
    cal_sub_mech = automech.io.read(
        p_.calculated_mechanism(tag, "json", path=p_.data(root_path))
    )

    # Compare calculated to parent mechanism
    print("\nCompare calculated mechanism to parent mechanism...")
    tags0 = previous_tags(tag)
    cal_paths0 = [
        p_.calculated_mechanism(t, "json", path=p_.data(root_path)) for t in tags0
    ]
    cal_mechs0 = list(map(automech.io.read, cal_paths0))
    trues = [True] * len(tags0)
    automech.display_reactions(
        cal_sub_mech,
        comp_mechs=[par_mech, *cal_mechs0],
        comp_labels=["Hill", *tags0],
        comp_stereo=[False, *trues],
    )


def prepare_simulation_species(tag: str, root_path: str | Path) -> None:
    """Simulate model (JSR).

    :param tag: Mechanism tag
    :param root_path: Project root directory
    """
    data_path = p_.data(root_path)
    mech = automech.io.read(p_.full_calculated_mechanism(tag, "json", path=data_path))

    # Read in data and rename species to match simulation
    print("\nReading in species...")
    name_df0 = pl.read_csv(data_path / "hill" / "species.csv")

    # Form join columns with original (stereo-free) names
    tmp_col = c_.temp()
    name_col = Species.name
    name_col0 = c_.orig(name_col)
    name_df0 = name_df0.with_columns(pl.col(name_col0).alias(tmp_col))
    name_df = mech.species.with_columns(
        pl.col(name_col0).fill_null(pl.col(name_col)).alias(tmp_col)
    ).select(name_col, tmp_col)

    # Join to add updated names
    name_df = name_df0.join(name_df, on=tmp_col, how="left").drop(tmp_col)
    assert name_df.select(name_col).null_count().item() == 0

    # Write names to CSV
    print("\nWriting simulation species names to CSV...")
    name_path = p_.simulation_species(tag, path=p_.cantera(root_path))
    print(name_path)
    name_df.write_csv(name_path)
    return name_df


def run_o2_simulation(
    tag: str,
    root_path: str | Path,
    temp_k: Number = 825,
    pres_atm: Number = 1.1,
    tau_s: Number = 4,
    vol_cm3: Number = 1,
    gather_every: int = 1,
    max_time: int = 300,
    control: bool = False,
) -> None:
    """Simulate JSR O2 sweep with model.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param temp_k: Temperature in K
    :param pres_atm: Pressure in atm
    :param tau_s: Residence time in s
    :param vol_cm3: Volume in cm^3
    :param gather_every: Gather every nth point
    :param max_time: Time limit per simulation
    :param control: Whether to run the control mechanism
    """
    data_path = p_.data(root_path)

    # Read in data and rename species to match simulation
    print("\nReading in species...")
    name_col = Species.name
    name_col0 = c_.orig(name_col)
    name_df0 = pl.read_csv(data_path / "hill" / "species.csv")
    name_df = prepare_simulation_species(tag, root_path)

    # Read in concentration data
    print("\nReading in concentrations...")
    conc_df = pl.read_csv(data_path / "hill" / "concentration.csv")
    conc_df = conc_df.gather_every(gather_every)
    print(conc_df)

    # Determine concentrations for each point
    print("\nDetermining concentrations for each point...")
    spc_dct = dict(name_df0.select("concentration", name_col0).drop_nulls().iter_rows())
    concs = conc_df.rename(spc_dct).select("CPT(563)", "N2", "O2(6)").rows(named=True)
    print(concs)

    # Load mechanism and set initial conditions
    print("\nDefining model and conditions...")
    model = (
        cantera.Solution(
            p_.full_control_mechanism(tag, "yaml", path=p_.cantera(root_path))
        )
        if control
        else cantera.Solution(
            p_.full_calculated_mechanism(tag, "yaml", path=p_.cantera(root_path))
        )
    )
    pres_atm *= cantera.one_atm  # convert to Pa from atm
    vol_cm3 *= (1e-2) ** 3  # convert to m^3 from cm^3

    # Run simulations for each point and store the results in an array
    print("\nRunning simulations...")
    sim_dct = dict.fromkeys(name_df[name_col], ())
    for conc in concs:
        print(f"Starting simulation for {conc}")
        time0 = time.time()
        try:
            reactor = timeout(reactors.jsr, max_time)(
                model=model,
                temp=temp_k,
                pres=pres_atm,
                tau=tau_s,
                vol=vol_cm3,
                conc=conc,
            )
            x_dct = reactor.thermo.mole_fraction_dict()
            sim_dct = {s: (*x, x_dct.get(s)) for s, x in sim_dct.items()}
            print(f"Finished in {time.time() - time0} s")
        except TimeoutError:
            sim_dct = {s: (*x, None) for s, x in sim_dct.items()}
            print(f"Timed out after {time.time() - time0} s")
        except Exception as e:
            print(f"Error: {e}")

    print("\nExtracting results...")
    sim_df = conc_df.with_columns(pl.Series(s, xs) * 10**6 for s, xs in sim_dct.items())
    print(sim_df)

    print("\nWriting results to CSV...")
    sim_path = (
        p_.full_control_mechanism(tag, "csv", path=p_.cantera_o2(root_path))
        if control
        else p_.full_calculated_mechanism(tag, "csv", path=p_.cantera_o2(root_path))
    )
    print(sim_path)
    sim_df.write_csv(sim_path)


def run_t_simulation(
    tag: str,
    root_path: str | Path,
    pres_atm: Number = 1.1,
    tau_s: Number = 4,
    vol_cm3: Number = 1,
    gather_every: int = 1,
    max_time: int = 300,
    control: bool = False,
) -> None:
    """Simulate JSR O2 sweep with model.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param temps_k: Temperatures in K
    :param pres_atm: Pressure in atm
    :param tau_s: Residence time in s
    :param vol_cm3: Volume in cm^3
    :param gather_every: Gather every nth point
    :param max_time: Time limit per simulation
    :param control: Whether to run the control mechanism
    """
    data_path = p_.data(root_path)

    # Read in data and rename species to match simulation
    print("\nReading in species...")
    name_col = Species.name
    name_col0 = c_.orig(name_col)
    name_df0 = pl.read_csv(data_path / "hill" / "species.csv")
    name_df = prepare_simulation_species(tag, root_path)

    # Read in concentration data
    print("\nReading in concentrations...")
    conc_df = pl.read_csv(data_path / "hill" / "concentration.csv")
    conc_df = conc_df.filter(pl.col("phi") == 1)
    print(conc_df)

    # Determine concentrations for each point
    print("\nDetermining concentrations for each point...")
    spc_dct = dict(name_df0.select("concentration", name_col0).drop_nulls().iter_rows())
    conc0 = conc_df.rename(spc_dct).select("CPT(563)", "N2", "O2(6)").row(0, named=True)
    print(conc0)

    # Read in temperature data
    print("\nReading in temperatures...")
    temp_df = pl.read_csv(data_path / "hill" / "T" / "temperature.csv")
    temp_df = temp_df.gather_every(gather_every)
    temps = temp_df.get_column("temperature").to_list()
    print(temps)

    # Load mechanism and set initial conditions
    print("\nDefining model and conditions...")
    model = (
        cantera.Solution(
            p_.full_control_mechanism(tag, "yaml", path=p_.cantera(root_path))
        )
        if control
        else cantera.Solution(
            p_.full_calculated_mechanism(tag, "yaml", path=p_.cantera(root_path))
        )
    )
    pres_atm *= cantera.one_atm  # convert to Pa from atm
    vol_cm3 *= (1e-2) ** 3  # convert to m^3 from cm^3

    # Run simulations for each point and store the results in an array
    conc = conc0
    print(f"\nRunning simulations for {conc}...")
    sim_dct = dict.fromkeys(name_df[name_col], ())
    for temp in temps:
        print(f"Starting simulation at T={temp} K")
        time0 = time.time()
        try:
            reactor = timeout(reactors.jsr, max_time)(
                model=model,
                temp=temp,
                pres=pres_atm,
                tau=tau_s,
                vol=vol_cm3,
                conc=conc,
            )
            x_dct = reactor.thermo.mole_fraction_dict()
            sim_dct = {s: (*x, x_dct.get(s)) for s, x in sim_dct.items()}
            print(f"Calculated: Finished in {time.time() - time0} s")
        except TimeoutError:
            sim_dct = {s: (*x, None) for s, x in sim_dct.items()}
            print(f"Calculated: Timed out after {time.time() - time0} s")
        except Exception as e:
            print(f"Error: {e}")

    print("\nExtracting results...")
    sim_df = temp_df.with_columns(pl.Series(s, xs) * 10**6 for s, xs in sim_dct.items())
    print(sim_df)

    print("\nWriting results to CSV...")
    sim_path = (
        p_.full_control_mechanism(tag, "csv", path=p_.cantera_t(root_path))
        if control
        else p_.full_calculated_mechanism(tag, "csv", path=p_.cantera_t(root_path))
    )
    print(sim_path)
    sim_df.write_csv(sim_path)


def plot_o2_simulation(
    tag: str,
    root_path: str | Path,
    x_col: str,
    x_title: str = "O₂ (molec/cm³) ⋅ 10⁻¹⁸",
    y_title: str = "concentration (ppm)",
    control: bool = True,
    line_source_: str | Sequence[str] | None = None,
    point_source: str | None = None,
    my_work_label: str = "This work",
    control_label: str = "Control",
) -> dict[str, altair.Chart]:
    """Plot simulation results.

    :param tag: Mechanism tag
    :param root_path: Project root directory
    :param control: Whether to include the control line
    :param line_source_: Extra data source(s) to plot as line(s)
    :param point_source: Extra data source to plot as points
    :return: Altair chart
    """
    line_source_ = [] if line_source_ is None else line_source_
    line_sources = [line_source_] if isinstance(line_source_, str) else line_source_
    sources = [*line_sources, point_source] if point_source else line_sources

    # Read in simulation results
    name_df = pl.read_csv(p_.simulation_species(tag, path=p_.cantera(root_path)))
    name_df = name_df.filter(pl.col("Experiment").is_not_null())

    sim_df = pl.read_csv(
        p_.full_calculated_mechanism(tag, "csv", path=p_.cantera_o2(root_path))
    )
    sim_df0 = pl.read_csv(
        p_.full_control_mechanism(tag, "csv", path=p_.cantera_o2(root_path))
    )

    data_path = p_.data(root_path)
    data_dct = {s: pl.read_csv(data_path / "hill" / f"{s}.csv") for s in sources}
    data_dct = {s: rename_columns(s, d, name_df) for s, d in data_dct.items()}
    data_dct[my_work_label] = sim_df
    line_sources.insert(0, my_work_label)

    # Add the control line, if requested
    if control:
        data_dct[control_label] = sim_df0
        line_sources.append(control_label)

    chart_dct = {
        name: make_chart(
            data_dct,
            x_col,
            name,
            line_sources=line_sources,
            point_source=point_source,
            x_title=x_title,
            y_title=y_title,
        )
        for name in name_df.get_column(Species.name).to_list()
    }
    return chart_dct


def rename_columns(
    source: str, data_df: pl.DataFrame, name_df: pl.DataFrame
) -> pl.DataFrame:
    """Preprocess data for plotting.

    :param source: Data source
    :param data_df: Simulation data
    :param name_df: Species names
    :return: Preprocessed data
    """
    col_dct = dict(name_df.select(source, Species.name).drop_nulls().iter_rows())
    col_dct.update({f"{n0}_err": f"{n}_err" for n0, n in col_dct.items()})
    col_dct = {n0: n for n0, n in col_dct.items() if n0 in data_df.columns}
    return data_df.rename(col_dct)


def isolate_xy_columns(
    source: str, data_df: pl.DataFrame | None, x_col: str, y_col: str
) -> pl.DataFrame:
    """Isolate x and y columns.

    :param source: Data source
    :param data_df: Data
    :param x_col: X-axis column
    :param y_col: Y-axis column
    :return: Isolated data
    """
    if data_df is None:
        return None

    return data_df.select(x_col, y_col).rename({y_col: source})


COLOR_SEQUENCE = [
    "#ff0000",  # red
    "#0066ff",  # blue
    "#1ab73a",  # green
    "#ef7810",  # orange
    "#8533ff",  # purple
    "#d0009a",  # pink
    "#ffcd00",  # yellow
    "#916e6e",  # brown
]


def make_chart(
    data_dct: dict[str, pl.DataFrame],
    x_col: str,
    y_col: str,
    line_sources: Sequence[str],
    point_source: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> altair.Chart:
    """Make an altair chart for one variable.

    :param data_dct: Data sources
    :param x_col: X-axis column
    :param y_col: Y-axis column
    :param line_sources: Data sources to plot as lines
    :param point_source: Extra data source to plot as points
    :return: Joined data
    """
    x_title = x_title or x_col
    y_title = y_title or y_col
    line_dfs = [
        isolate_xy_columns(s, data_dct.get(s), x_col, y_col) for s in line_sources
    ]
    point_df = isolate_xy_columns(
        point_source, data_dct.get(point_source), x_col, y_col
    )

    line_df = functools.reduce(lambda x, y: x.join(y, on=x_col), line_dfs)

    line_colors = altair.Scale(
        domain=line_sources, range=COLOR_SEQUENCE[: len(line_sources)]
    )

    point_chart = (
        altair.Chart(point_df)
        .mark_circle()
        .encode(x=x_col, y=point_source, color=altair.value("black"))
    )
    line_chart = (
        altair.Chart(line_df)
        .mark_line()
        .transform_fold(fold=line_sources)
        .encode(
            x=altair.Y(x_col, title=x_title),
            y=altair.Y("value:Q", title=y_title),
            color=altair.Color("key:N", scale=line_colors),
        )
    )
    return line_chart + point_chart


# Helpers
def timeout(func, seconds=10):
    def raise_timeout_error(signum, frame):
        raise TimeoutError()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        signal.signal(signal.SIGALRM, raise_timeout_error)
        signal.alarm(seconds)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
        return result

    return wrapper
