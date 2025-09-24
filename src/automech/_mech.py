"""Definition and core functionality of mechanism data structure."""

import functools
import itertools
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TypeAlias

import autochem as ac
import automol
import more_itertools as mit
import polars
import pydantic
from automol.graph import enum
from IPython.display import display as ipy_display

from . import net as net_
from . import reaction, species
from .reaction import (
    Reaction,
    ReactionDataFrame_,
    ReactionRate,
    ReactionSorted,
    ReactionStereo,
    ReactionUnstable,
)
from .species import Species, SpeciesDataFrame_, SpeciesStereo
from .util import c_, df_, pandera_


class Mechanism(pydantic.BaseModel):
    """Chemical kinetic mechanism."""

    reactions: ReactionDataFrame_
    species: SpeciesDataFrame_


def from_network(net: net_.Network) -> Mechanism:
    """Generate mechanism from reaction network.

    :param net: Reaction network
    :return: Mechanism
    """
    spc_data = list(
        itertools.chain(*(d[net_.Key.species] for *_, d in net.nodes.data()))
    )
    rxn_data = [d for *_, d in net.edges.data()]

    spc_df = pandera_.empty(Species) if not spc_data else (polars.DataFrame(spc_data))
    rxn_df = pandera_.empty(Reaction) if not rxn_data else (polars.DataFrame(rxn_data))

    def _postprocess(df: polars.DataFrame) -> polars.DataFrame:
        col = net_.Key.id
        if col not in df:
            return df
        return df.sort(col).unique(col, maintain_order=True).drop(col, strict=False)

    spc_df = _postprocess(spc_df)
    rxn_df = _postprocess(rxn_df)
    return Mechanism(reactions=rxn_df, species=spc_df)


def from_smiles(
    spc_smis: Sequence[str] = (),
    rxn_smis: Sequence[str] = (),
    name_dct: dict[str, str] | None = None,
    src_mech: Mechanism | None = None,
) -> Mechanism:
    """Generate mechanism using SMILES strings for species names.

    If `name_dct` is `None`, CHEMKIN names will be auto-generated.

    :param spc_smis: Species SMILES strings
    :param rxn_smis: Optionally, reaction SMILES strings
    :param name_dct: Optionally, specify name for some molecules
    :param spin_dct: Optionally, specify spin state (2S) for some molecules
    :param charge_dct: Optionally, specify charge for some molecules
    :param src_mech: Optional source mechanism for species names
    :return: Mechanism
    """
    # Add in any missing species from reaction SMILES
    rct_smis = list(map(automol.smiles.reaction_reactants, rxn_smis))
    prd_smis = list(map(automol.smiles.reaction_products, rxn_smis))
    spc_smis = list(
        mit.unique_everseen(itertools.chain(spc_smis, *rct_smis, *prd_smis))
    )

    # Build species dataframe
    spc_df = species.bootstrap(
        {Species.smiles: spc_smis}, name_dct=name_dct, key=Species.smiles
    )

    # Left-update by species key, if source mechanism was provided
    if src_mech is not None:
        spc_df = species.left_update(spc_df, src_mech.species, drop_orig=True)

    # Build reactions dataframe
    trans_dct = df_.lookup_dict(spc_df, Species.amchi, Species.name)
    rct_chis = [list(map(automol.smiles.amchi, rs)) for rs in rct_smis]
    prd_chis = [list(map(automol.smiles.amchi, rs)) for rs in prd_smis]
    data = {Reaction.reactants: rct_chis, Reaction.products: prd_chis}
    rxn_df = reaction.bootstrap(data, name_dct=trans_dct, spc_df=spc_df)

    mech = Mechanism(reactions=rxn_df, species=spc_df)
    return mech if src_mech is None else left_update(mech, src_mech)


# properties
def species_count(mech: Mechanism) -> int:
    """Get number of species in mechanism.

    :param mech: Mechanism
    :return: Number of species
    """
    return df_.count(mech.species)


def reaction_count(mech: Mechanism) -> int:
    """Get number of reactions in mechanism.

    :param mech: Mechanism
    :return: Number of reactions
    """
    return df_.count(mech.reactions)


def reagents(mech: Mechanism) -> list[list[str]]:
    """Get sets of reagents in mechanism.

    :param mech: Mechanism
    :return: Sets of reagents
    """
    return reaction.reagents(mech.reactions)


def reaction_rate_objects(mech: Mechanism, eq: str) -> list[ac.rate.Reaction]:
    """Get rate objects associated with one reaction.

    :param mech: Mechanism
    :param eq: Equation
    :return: Rate objects
    """
    return reaction.reaction_rate_objects(mech.reactions, eq=eq)


def species_names(
    mech: Mechanism,
    rxn_only: bool = False,
    formulas: Sequence[str] | None = None,
    exclude_formulas: Sequence[str] = (),
) -> list[str]:
    """Get names of species in mechanism.

    :param mech: Mechanism
    :param rxn_only: Only include species that are involved in reactions?
    :param formulas: Formula strings of species to include, using * for wildcard
        stoichiometry
    :param exclude_formulas: Formula strings of species to exclude, using * for wildcard
        stoichiometry
    :return: Species names
    """

    def _formula_matcher(fml_strs):
        """Determine whether a species is excluded."""
        fmls = list(map(automol.form.from_string, fml_strs))

        def _matches_formula(chi):
            fml = automol.amchi.formula(chi)
            return any(automol.form.match(fml, e) for e in fmls)

        return _matches_formula

    spc_df = mech.species

    if formulas is not None:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(formulas), dtype_=bool
        )
        spc_df = spc_df.filter(polars.col("match"))

    if exclude_formulas:
        spc_df = df_.map_(
            spc_df, Species.amchi, "match", _formula_matcher(exclude_formulas)
        )
        spc_df = spc_df.filter(~polars.col("match"))

    spc_names = spc_df[Species.name].to_list()

    if rxn_only:
        rxn_df = mech.reactions
        rxn_spc_names = reaction.species_names(rxn_df)
        spc_names = [n for n in spc_names if n in rxn_spc_names]

    return spc_names


def unstable_species_names(mech: Mechanism) -> list[str]:
    """Get names of unstable species in mechanism.

    :param mech: Mechanism
    :return: Species names
    """
    instab_mech = without_reactions(mech)
    instab_mech = enumerate_reactions(instab_mech, enum.ReactionSmarts.qooh_instability)
    return reaction.reactant_names(instab_mech.reactions)


def rename_dict(mech1: Mechanism, mech2: Mechanism) -> tuple[dict[str, str], list[str]]:
    """Generate dictionary for renaming species names from one mechanism to another.

    :param mech1: Mechanism with original names
    :param mech2: Mechanism with desired names
    :return: Dictionary mapping names from `mech1` to those in `mech2`, and list
        of names from `mech1` that are missing in `mech2`
    """
    match_cols = [Species.amchi, Species.spin, Species.charge]

    # Read in species and names
    col_dct = c_.to_(Species.name, c_.temp())
    spc1_df = mech1.species.rename(col_dct)
    spc2_df = mech2.species.select([Species.name, *match_cols])

    # Get names from first mechanism that are included/excluded in second
    incl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="inner")
    excl_spc_df = spc1_df.join(spc2_df, on=match_cols, how="anti")

    (orig_col,) = col_dct.values()
    name_dct = df_.lookup_dict(incl_spc_df, orig_col, Species.name)
    missing_names = excl_spc_df.get_column(orig_col).to_list()
    return name_dct, missing_names


def network(mech: Mechanism) -> net_.Network:
    """Generate reaction network representation of mechanism.

    :param mech: Mechanism
    :return: Reaction network
    """
    spc_df = mech.species
    rxn_df = mech.reactions

    # Double-check that reagents are sorted
    rxn_df = reaction.with_sorted_reagents(rxn_df, reversible=False)

    # Add species and reaction indices
    spc_df = df_.with_index(spc_df, net_.Key.id)
    rxn_df = df_.with_index(rxn_df, net_.Key.id)

    # Exluded species
    rgt_names = list(
        itertools.chain(
            *rxn_df[Reaction.reactants].to_list(), *rxn_df[Reaction.products].to_list()
        )
    )
    excl_spc_df = spc_df.filter(~polars.col(Species.name).is_in(rgt_names))

    # Get dataframe of reagents
    rgt_col = "reagents"
    rgt_exprs = [
        rxn_df.select(polars.col(Reaction.reactants).alias(rgt_col), Reaction.formula),
        rxn_df.select(polars.col(Reaction.products).alias(rgt_col), Reaction.formula),
        excl_spc_df.select(
            polars.concat_list(Species.name).alias(rgt_col), Species.formula
        ),
    ]
    rgt_df = polars.concat(rgt_exprs, how="vertical_relaxed").group_by(rgt_col).first()

    # Append species data to reagents dataframe
    names = spc_df[Species.name]
    datas = spc_df.to_struct()
    expr = polars.element().replace_strict(names, datas, default=None)
    rgt_df = rgt_df.with_columns(
        polars.col(rgt_col).list.eval(expr).alias(net_.Key.species)
    )

    # Build network object
    def _node_data_from_dict(dct: dict[str, object]):
        key = tuple(dct.get(rgt_col))
        return (key, dct)

    def _edge_data_from_dict(dct: dict[str, object]):
        key1 = tuple(dct.get(Reaction.reactants))
        key2 = tuple(dct.get(Reaction.products))
        return (key1, key2, dct)

    return net_.from_data(
        node_data=list(map(_node_data_from_dict, rgt_df.to_dicts())),
        edge_data=list(map(_edge_data_from_dict, rxn_df.to_dicts())),
    )


def apply_network_function(
    mech: Mechanism, func: Callable, *args, **kwargs
) -> Mechanism:
    """Apply network function to mechanism.

    :param mech: Mechanism
    :param func: Function
    :param *args: Function arguments
    :param **kwargs: Function keyword arguments
    :return: Mechanism
    """
    mech_ = mech.model_copy()

    col_idx = c_.temp()
    mech_.species = df_.with_index(mech_.species, col=col_idx)
    mech_.reactions = df_.with_index(mech_.reactions, col=col_idx)
    net = network(mech_)
    net = func(net, *args, **kwargs)
    spc_idxs = net_.species_values(net, col_idx)
    rxn_idxs = net_.edge_values(net, col_idx)
    mech_.species = mech_.species.filter(polars.col(col_idx).is_in(spc_idxs)).drop(
        col_idx
    )
    mech_.reactions = mech_.reactions.filter(polars.col(col_idx).is_in(rxn_idxs)).drop(
        col_idx
    )
    return mech_


# transformations
def rename(
    mech: Mechanism,
    names: Sequence[str] | Mapping[str, str],
    new_names: Sequence[str] | None = None,
    orig_prefix: str | None = None,
    drop_missing: bool = False,
) -> Mechanism:
    """Rename species in mechanism.

    :param mech: Mechanism
    :param names: A list of names or mapping from current to new names
    :param new_names: A list of new names
    :param drop_orig: Whether to drop the original names, or include them as `orig`
    :param drop_missing: Whether to drop missing species or keep them
    :return: Mechanism with updated species names
    """
    mech = mech.model_copy()

    if drop_missing:
        mech = with_species(mech, list(names), strict=drop_missing)

    mech.species = species.rename(
        mech.species, names=names, new_names=new_names, orig_prefix=orig_prefix
    )
    mech.reactions = reaction.rename(
        mech.reactions, names=names, new_names=new_names, orig_prefix=orig_prefix
    )
    return mech


def neighborhood(
    mech: Mechanism, species_names: Sequence[str], radius: int = 1
) -> Mechanism:
    """Determine neighborhood of set of species.

    :param mech: Mechanism
    :param species_names: Names of species
    :param radius: Maximum distance of neighbors to include, defaults to 1
    :return: Neighborhood mechanism
    """
    return apply_network_function(
        mech, net_.neighborhood, species_names=species_names, radius=radius
    )


# drop/add reactions
def drop_duplicate_reactions(mech: Mechanism) -> Mechanism:
    """Drop duplicate reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism without duplicate reactions
    """
    mech = mech.model_copy()

    col_tmp = c_.temp()
    mech.reactions = reaction.with_key(mech.reactions, col=col_tmp, reversible=True)
    mech.reactions = mech.reactions.unique(col_tmp, maintain_order=True)
    mech.reactions = mech.reactions.drop(col_tmp)
    return mech


def drop_unstable_product_reactions(mech: Mechanism) -> Mechanism:
    """Drop instability from mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    instab_mech = without_reactions(mech)
    instab_mech = enumerate_reactions(instab_mech, enum.ReactionSmarts.qooh_instability)
    return reaction_difference(
        mech, instab_mech, reversible=True, stereo=False, drop_species=False
    )


def drop_self_reactions(mech: Mechanism) -> Mechanism:
    """Drop self-reactions from mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()
    mech.reactions = reaction.drop_self_reactions(mech.reactions)
    return mech


def drop_reactions_by_smiles(
    mech: Mechanism, rxn_smis: Sequence[str] = (), stereo: bool = True
) -> Mechanism:
    """Drop species and reactions by SMILES strings.

    :param spc_smis: Species SMILES strings
    :param rxn_smis: Optionally, reaction SMILES strings
    :param src_mech: Optional source mechanism for species names
    :param stereo: Whether to include stereochemistry in matching
    :return: Mechanism
    """
    mech = mech.model_copy()

    drop_mech = from_smiles(rxn_smis=rxn_smis, src_mech=mech)

    tmp_col = c_.temp()
    mech = with_key(mech, col=tmp_col, stereo=stereo, reversible=True)
    drop_mech = with_key(drop_mech, col=tmp_col, stereo=stereo, reversible=True)
    drop_keys = drop_mech.reactions.get_column(tmp_col).implode()
    mech.reactions = mech.reactions.filter(~polars.col(tmp_col).is_in(drop_keys))
    mech.reactions = mech.reactions.drop(tmp_col)
    mech.species = mech.species.drop(tmp_col)
    return mech


def drop_noncanonical_enantiomers(mech: Mechanism) -> Mechanism:
    """Drop non-canonical enantiomer reactions from a mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()

    if mech.reactions.is_empty():
        mech.species = species.drop_noncanonical_enantiomers(mech.species)
        return mech

    mech.reactions = reaction.drop_noncanonical_enantiomers(mech.reactions)
    mech = without_unused_species(mech)
    return mech


def with_species(
    mech: Mechanism, spc_names: Sequence[str] = (), strict: bool = False
) -> Mechanism:
    """Extract submechanism including species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    return _with_or_without_species(
        mech=mech, spc_names=spc_names, without=False, strict=strict
    )


def without_species(mech: Mechanism, spc_names: Sequence[str] = ()) -> Mechanism:
    """Extract submechanism excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be excluded
    :return: Submechanism
    """
    return _with_or_without_species(mech=mech, spc_names=spc_names, without=True)


def _with_or_without_species(
    mech: Mechanism,
    spc_names: Sequence[str] = (),
    without: bool = False,
    strict: bool = False,
) -> Mechanism:
    """Extract submechanism containing or excluding species names from list.

    :param mech: Mechanism
    :param spc_names: Names of species to be included or excluded
    :param without: Extract submechanism *without* these species?
    :param strict: Strictly include these species and no others?
    :return: Submechanism
    """
    mech = mech.model_copy()

    # Build appropriate filtering expression
    expr = polars.concat_list(Reaction.reactants, Reaction.products).list.eval(
        polars.element().is_in(spc_names)
    )
    expr = expr.list.all() if strict else expr.list.any()
    expr = expr.not_() if without else expr

    # Temporary workaround for bug https://github.com/pola-rs/polars/issues/23300:
    # Should be able to just do mech.reactions.filter(expr)
    tmp_col = c_.temp()
    mech.reactions = (
        mech.reactions.with_columns(expr.alias(tmp_col))
        .filter(polars.col(tmp_col))
        .drop(tmp_col)
    )
    return without_unused_species(mech)


def without_reactions(mech: Mechanism) -> Mechanism:
    """Remove all reactions from the mechanism.

    :param mech: Mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()
    mech.reactions = mech.reactions.clear()
    return mech


def without_unused_species(mech: Mechanism) -> Mechanism:
    """Remove unused species from mechanism.

    :param mech: Mechanism
    :return: Mechanism without unused species
    """
    mech = mech.model_copy()
    used_names = species_names(mech, rxn_only=True)
    mech.species = mech.species.filter(polars.col(Species.name).is_in(used_names))
    return mech


def with_key(
    mech: Mechanism, col: str = "key", stereo: bool = True, reversible: bool = False
) -> Mechanism:
    """Add match key column for species and reactions.

    :param mech: Mechanism
    :param col: Key column identifying common species and reactions
    :param stereo: Whether to include stereochemistry
    :param reversibe: Whether reactions are reversible, in which case the reagents will
        be cross-sorted to a canonical direction. Can be specified by a Boolean column
        indicating which reactions are reversible.
    :return: First and second Mechanisms with intersection columns
    """
    mech = mech.model_copy()
    mech.species = species.with_key(mech.species, col=col, stereo=stereo)
    mech.reactions = reaction.with_key(
        mech.reactions, col, spc_df=mech.species, stereo=stereo, reversible=reversible
    )
    return mech


def with_rate_objects(
    mech: Mechanism,
    col: str,
    comp_mechs: Sequence[Mechanism] = (),
    comp_cols: Sequence[str] = (),
    comp_stereo: bool | Sequence[bool] = True,
    fill: bool = False,
) -> Mechanism:
    """Add rate objects.

    :param mech: Mechanism
    :param col: Column
    :param comp_mechs: Other mechanisms by rate object column
    :param comp_stereo: Whether to include stereo in matching reactions
        Use sequence to specify for each mechanism
    :param fill: Whether to fill in missing rates
    :return: Mechanism
    """
    mech = mech.model_copy()
    mech.reactions = reaction.with_rate_objects(mech.reactions, col=col, fill=fill)
    if comp_mechs is not None:
        mech = with_comparison_rate_objects(
            mech, comp_mechs=comp_mechs, comp_cols=comp_cols, stereo=comp_stereo
        )
    return mech


def with_comparison_rate_objects(
    mech: Mechanism,
    comp_mechs: Sequence[Mechanism],
    comp_cols: Sequence[str],
    stereo: bool | Sequence[bool] = True,
) -> Mechanism:
    """Add rate objects from other mechanisms for comparison.

    :param mech: Mechanism
    :param comp_mechs: Comparison mechanisms
    :param comp_cols: Comparison columns
    :param stereo: Whether to include stereo in matching reactions
        Use sequence to specify for each mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()

    comp_stereos = [stereo] * len(comp_mechs) if isinstance(stereo, bool) else stereo

    key_col_dct = {s: c_.temp() for s in set(comp_stereos)}
    for comp_stereo, key_col in key_col_dct.items():
        mech.reactions = reaction.with_key(
            mech.reactions,
            col=key_col,
            spc_df=mech.species,
            stereo=comp_stereo,
            reversible=False,
        )

    for comp_col, comp_mech, comp_stereo in zip(
        comp_cols, comp_mechs, comp_stereos, strict=True
    ):
        key_col = key_col_dct.get(comp_stereo)
        spc_df = comp_mech.species
        rxn_df = comp_mech.reactions
        rxn_df = reaction.with_key(
            rxn_df, col=key_col, spc_df=spc_df, stereo=comp_stereo, reversible=False
        )
        rxn_df = reaction.with_rate_objects(rxn_df, col=comp_col)
        rxn_df = rxn_df.select(key_col, comp_col)
        mech.reactions = mech.reactions.join(rxn_df, on=key_col, how="left")

    mech.reactions = mech.reactions.drop(list(key_col_dct.values()))
    return mech


def expand_stereo(
    mech: Mechanism,
    enant: bool = True,
    strained: bool = False,
    distinct_ts: bool = True,
) -> tuple[Mechanism, Mechanism]:
    """Expand stereochemistry for mechanism.

    Note: The ugliness of this function is unavoidable until we can include AutoMol
    objects/structs in Polars columns. This will require a refactor on the AutoMol side,
    to turn *both* Graph *and* Reaction objects into Pydantic models.

    Note: Setting `enant` to `False` results in a mechanism containing only canonical
    enantiomer *reactions*. Non-canonical enantiomer *species* will only be dropped if
    they are not present in any canonical reaction.

    :param mech: Mechanism
    :param enant: Distinguish between enantiomers?, defaults to True
    :param strained: Include strained stereoisomers?
    :param distinct_ts: Include duplicate reactions for distinct TSs?
    :return: Mechanism with classified reactions, and one with unclassified
    """
    mech = mech.model_copy()
    err_mech = mech.model_copy()
    species0 = mech.species

    # Do species expansion
    mech.species = species.expand_stereo(mech.species, enant=True, strained=strained)

    if not reaction_count(mech):
        mech.species = (
            mech.species
            if enant
            else mech.species.filter(polars.col(SpeciesStereo.canon))
        )
        return mech, mech.model_copy()

    # Add reactant and product AMChIs
    rct_col = Reaction.reactants
    prd_col = Reaction.products
    temp_dct = c_.to_([rct_col, prd_col], c_.temp())
    mech.reactions = reaction.translate_reagents(
        mech.reactions,
        trans=species0[Species.name],
        trans_into=species0[Species.amchi],
        rct_col=temp_dct.get(rct_col),
        prd_col=temp_dct.get(prd_col),
    )

    # Add "orig" prefix to current reactant and product columns
    orig_dct = c_.to_orig([rct_col, prd_col])
    mech.reactions = mech.reactions.drop(orig_dct.values(), strict=False)
    mech.reactions = mech.reactions.rename(orig_dct)

    # Define expansion function
    name_dct: dict = df_.lookup_dict(
        mech.species, (c_.orig(Species.name), Species.amchi), Species.name
    )

    def _expand_reaction(rchi0s, pchi0s, rname0s, pname0s):
        """Classify reaction and return reaction objects."""
        objs = automol.reac.from_amchis(rchi0s, pchi0s, stereo=False)
        rnames_lst = []
        pnames_lst = []
        ts_amchis = []
        canons = []
        for obj in objs:
            sobjs = automol.reac.expand_stereo(obj, enant=True, strained=strained)
            for sobj in sobjs:
                # Determine AMChI
                ts_amchi = automol.reac.ts_amchi(sobj)
                # Determine updated equation
                rchis, pchis = automol.reac.amchis(sobj)
                rnames = tuple(map(name_dct.get, zip(rname0s, rchis, strict=True)))
                pnames = tuple(map(name_dct.get, zip(pname0s, pchis, strict=True)))
                canon = automol.amchi.is_canonical_enantiomer_reaction(rchis, pchis)
                if not all(isinstance(n, str) for n in rnames + pnames):
                    return ([], [], [], [])

                rnames_lst.append(rnames)
                pnames_lst.append(pnames)
                ts_amchis.append(ts_amchi)
                canons.append(canon)
        return rnames_lst, pnames_lst, ts_amchis, canons

    # Do expansion
    cols_in = [*temp_dct.values(), *orig_dct.values()]
    cols_out = (
        Reaction.reactants,
        Reaction.products,
        ReactionStereo.amchi,
        ReactionStereo.canon,
    )
    mech.reactions = df_.map_(
        mech.reactions, cols_in, cols_out, _expand_reaction, bar=True
    )

    # Separate out error cases
    err_mech.reactions = mech.reactions.filter(polars.col(rct_col).list.len() == 0)
    mech.reactions = mech.reactions.filter(polars.col(rct_col).list.len() != 0)

    # Expand table by stereoisomers
    err_mech.reactions = err_mech.reactions.drop(
        ReactionStereo.amchi, ReactionStereo.canon, *orig_dct.keys()
    ).rename(dict(map(reversed, orig_dct.items())))
    mech.reactions = mech.reactions.explode(
        Reaction.reactants,
        Reaction.products,
        ReactionStereo.amchi,
        ReactionStereo.canon,
    )
    mech.reactions = mech.reactions.drop(temp_dct.values())

    if not distinct_ts:
        mech = drop_duplicate_reactions(mech)

    if not enant:
        mech.reactions = mech.reactions.filter(polars.col(ReactionStereo.canon))
        names = reaction.species_names(mech.reactions)
        mech.species = mech.species.filter(
            polars.col(SpeciesStereo.canon) | polars.col(Species.name).is_in(names)
        )

    return mech, err_mech


# binary operations
def update(
    mech1: Mechanism, mech2: Mechanism, drop_orig: bool = True, how: str = "full"
) -> Mechanism:
    """Left-update mechanism data by species and reaction key.

    Any overlapping species or reactions will be replaced with those of the second
    mechanism.

    Warning: Not all join strategies will result in a consistent mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param drop_orig: Whether to drop the original column values
    :param how: Polars join strategy
    :return: Mechanism
    """
    mech = mech1.model_copy()
    name_dct, *_ = rename_dict(mech1, mech2)
    mech = rename(mech, name_dct)

    mech.species = species.update(
        mech.species, mech2.species, drop_orig=drop_orig, how=how
    )
    mech.reactions = reaction.update(
        mech.reactions, mech2.reactions, drop_orig=drop_orig, how=how
    )
    return mech


def left_update(
    mech1: Mechanism, mech2: Mechanism, drop_orig: bool = True
) -> Mechanism:
    """Left-update mechanism data by species and reaction key.

    Any overlapping species or reactions will be replaced with those of the second
    mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param drop_orig: Whether to drop the original column values
    :return: Mechanism
    """
    return update(mech1, mech2, drop_orig=drop_orig, how="left")


def species_difference(
    mech1: Mechanism,
    mech2: Mechanism,
    stereo: bool = True,
) -> Mechanism:
    """Get mechanism with species not included in another mechanism (drops reactions).

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param stereo: Whether to include stereochemistry
    :return: Mechanism
    """
    mech = without_reactions(mech1)
    mech.species = species.difference(mech1.species, mech2.species, stereo=stereo)
    return mech


def reaction_difference(
    mech1: Mechanism,
    mech2: Mechanism,
    reversible: bool = False,
    stereo: bool = True,
    drop_species: bool = True,
) -> Mechanism:
    """Get mechanism with reactions not included in another mechanism.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param reversible: Whether to treat reactions as reversible
    :param stereo: Whether to include stereochemistry
    :param drop_species: Whether to drop unused species
    :return: Mechanism
    """
    mech = mech1.model_copy()
    mech.reactions = reaction.difference(
        mech1.reactions,
        mech2.reactions,
        spc_df1=mech1.species,
        spc_df2=mech2.species,
        reversible=reversible,
        stereo=stereo,
    )
    if drop_species:
        mech = without_unused_species(mech)
    return mech


def full_difference(
    mech1: Mechanism,
    mech2: Mechanism,
    reversible: bool = False,
    stereo: bool = True,
) -> Mechanism:
    """Get mechanism with species and reactions not included in another mechanism.

    Purely for the convenient counting/statistics, as this generally results in an
    inconsistent mechanism that is missing some species that are involved in its
    reactions.

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :param reversible: Whether to treat reactions as reversible
    :param stereo: Whether to include stereochemistry
    :return: Mechanism
    """
    mech = mech1.model_copy()
    mech.reactions = reaction.difference(
        mech1.reactions,
        mech2.reactions,
        spc_df1=mech1.species,
        spc_df2=mech2.species,
        reversible=reversible,
        stereo=stereo,
    )
    mech.species = species.difference(mech1.species, mech2.species, stereo=stereo)
    return mech


# sequence operations
def combine_all(mechs: Sequence[Mechanism]) -> Mechanism:
    """Combine mechanisms into one.

    :param mechs: Mechanisms
    :return: Mechanism
    """
    return functools.reduce(update, mechs)


# parent
def expand_parent_stereo(mech: Mechanism, sub_mech: Mechanism) -> Mechanism:
    """Apply stereoexpansion of submechanism to parent mechanism.

    Produces equivalent of parent mechanism, containing distinct
    stereoisomers of submechanism. Expansion is completely naive, with no
    consideration of stereospecificity, and is simply designed to allow merging of
    stereo-expanded submechanism into parent mechanism.

    :param par_mech: Parent mechanism
    :param sub_mech: Stereo-expanded sub-mechanism
    :return: Equivalent parent mechanism, with distinct stereoisomers from
        sub-mechanism
    """
    mech = mech.model_copy()
    sub_mech = sub_mech.model_copy()

    # 1. Species table
    #   a. Add stereo columns to par_mech species table
    col_dct = c_.to_orig([Species.name, Species.smiles, Species.amchi])
    mech.species = mech.species.rename(col_dct)

    #   b. Group by original names and isolate expanded stereoisomers
    name_col = Species.name
    name_col0 = c_.orig(Species.name)
    sub_mech.species = species.validate(sub_mech.species, SpeciesStereo)
    sub_mech.species = sub_mech.species.select(*col_dct.keys(), *col_dct.values())
    sub_mech.species = sub_mech.species.group_by(name_col0).agg(polars.all())

    #   c. Form species expansion dictionary, to be used for reaction expansion
    exp_dct: dict[str, list[str]] = df_.lookup_dict(
        sub_mech.species, name_col0, name_col
    )

    #   d. Join on original names, explode, and fill in non-stereoisomer columns
    mech.species = mech.species.join(sub_mech.species, how="left", on=name_col0)
    mech.species = mech.species.drop(polars.selectors.ends_with("_right"))
    mech.species = mech.species.explode(*col_dct.keys())
    mech.species = mech.species.with_columns(
        *(polars.col(k).fill_null(polars.col(v)) for k, v in col_dct.items())
    )

    # 2. Reaction table
    #   a. Identify subset of reactions to be expanded
    has_rate = ReactionRate.rate in mech.reactions
    mech.reactions = reaction.with_rates(mech.reactions)

    mech.reactions = mech.reactions.with_columns(
        **c_.from_orig([Reaction.reactants, Reaction.products, ReactionRate.rate])
    )
    needs_exp = (
        polars.concat_list(Reaction.reactants, Reaction.products)
        .list.eval(polars.element().is_in(list(exp_dct.keys())))
        .list.any()
    )
    # Temporary workaround for bug https://github.com/pola-rs/polars/issues/23300:
    # Should be able to just do mech.reactions.filter(expr)
    tmp_col = c_.temp()
    exp_rxn_df = (
        mech.reactions.with_columns(needs_exp.alias(tmp_col))
        .filter(polars.col(tmp_col))
        .drop(tmp_col)
    )
    rem_rxn_df = (
        mech.reactions.with_columns((~needs_exp).alias(tmp_col))
        .filter(polars.col(tmp_col))
        .drop(tmp_col)
    )

    #   b. Expand and dump to dictionary
    def exp_(rate: ac.rate.Reaction) -> list[dict[str, object]]:
        rates = ac.rate.expand_lumped(rate, exp_dct=exp_dct)
        return (
            [r.reactants for r in rates],
            [r.products for r in rates],
            [r.rate.model_dump() for r in rates],
        )

    obj_col = c_.temp()
    cols = [Reaction.reactants, Reaction.products, ReactionRate.rate]
    dtypes = list(map(polars.List, map(exp_rxn_df.schema.get, cols)))
    exp_rxn_df = reaction.with_rate_objects(exp_rxn_df, col=obj_col)
    exp_rxn_df = df_.map_(exp_rxn_df, obj_col, cols, exp_, dtype_=dtypes, bar=True)
    exp_rxn_df = exp_rxn_df.explode(cols)
    mech.reactions = polars.concat([rem_rxn_df, exp_rxn_df.drop(obj_col)])

    if not has_rate:
        mech.reactions = reaction.without_rates(mech.reactions)
        mech.reactions = mech.reactions.drop(c_.orig(ReactionRate.rate))

    return mech


# building
ReagentValue_ = str | Sequence[str] | None


def enumerate_reactions(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    excl_rcts: Sequence[str] = (),
    src_mech: Mechanism | None = None,
    repeat: int = 1,
    drop_self_rxns: bool = True,
    match_src: bool = True,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Does not include stereochemistry! (Run before stereoexpansion.)

    Reactants are listed by position in the SMARTS template. If a sequence of reactants
    is provided, reactions will be enumerated for each of them. If `None` is provided,
    reactions will be enumerated for all species currently in the mechanism.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration (see above)
    :param spc_col_: Species column(s) for identifying reactants and products
    :param src_mech: Optional source mechanism for species names and data
    :param repeat: Number of times to repeat the enumeration
    :param drop_self_rxns: Whether to drop self-reactions
    :param match_src: Whether to match the direction and add data from the source mechanism
    :return: Mechanism with enumerated reactions
    """
    for _ in range(repeat):
        mech = _enumerate_reactions(
            mech,
            smarts,
            rcts_=rcts_,
            spc_col_=spc_col_,
            src_mech=src_mech,
            excl_rcts=excl_rcts,
            match_src=match_src,
        )

    if drop_self_rxns:
        mech = drop_self_reactions(mech)

    return mech


def enumerate_products(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: Mechanism | None = None,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Does not include stereochemistry! (Run before stereoexpansion.)

    Reactants are listed by position in the SMARTS template. If a sequence of reactants
    is provided, reactions will be enumerated for each of them. If `None` is provided,
    reactions will be enumerated for all species currently in the mechanism.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration (see above)
    :param spc_col_: Species column(s) for identifying reactants and products
    :param src_mech: Optional source mechanism for species names and data
    :param repeat: Number of times to repeat the enumeration
    :param drop_self_rxns: Whether to drop self-reactions
    :return: Mechanism with enumerated reactions
    """
    return _enumerate_reactions(
        mech,
        smarts,
        rcts_=rcts_,
        spc_col_=spc_col_,
        src_mech=src_mech,
        skip_rxn_update=True,
    )


def _enumerate_reactions(
    mech: Mechanism,
    smarts: str,
    rcts_: Sequence[ReagentValue_] | None = None,
    excl_rcts: Sequence[ReagentValue_] = (),
    spc_col_: str | Sequence[str] = Species.name,
    src_mech: Mechanism | None = None,
    skip_rxn_update: bool = False,
    match_src: bool = True,
) -> Mechanism:
    """Enumerate reactions for mechanism based on SMARTS reaction template.

    Does not include stereochemistry! (Run before stereoexpansion.)

    Reactants are listed by position in the SMARTS template. If a sequence of reactants
    is provided, reactions will be enumerated for each of them. If `None` is provided,
    reactions will be enumerated for all species currently in the mechanism.

    :param mech: Mechanism
    :param smarts: SMARTS reaction template
    :param rcts_: Reactants to be used in enumeration (see above)
    :param excl_rcts: Reactants to be excluded from enumeration (see above)
    :param spc_col_: Species column(s) for identifying reactants and products
    :param src_mech: Optional source mechanism for species names and data
    :param skip_rxn_update: Whether to skip the reaction update, only adding products of
        reactions as species
    :param skip_rxn_left_update: Whether to skip the left update of reactions, which
        will align their direction to and add data from `src_mech`
    :return: Mechanism with enumerated reactions
    """
    # Check reactants argument
    nrcts = automol.smarts.reactant_count(smarts)
    rcts_ = [None] * nrcts if rcts_ is None else rcts_
    assert len(rcts_) == nrcts, f"Reactant count mismatch for {smarts}:\n{rcts_}"

    # Process reactants argument
    mech = mech.model_copy()
    pool = df_.values(mech.species, spc_col_)
    rcts_vals_ = [
        pool if r is None else [r] if isinstance(r, str) else r for r in rcts_
    ]

    excl_rcts = [[r] if isinstance(r, str) else r for r in excl_rcts]
    excl_chis = species.amchis(mech.species, vals_=excl_rcts, col_=spc_col_, fill=True)

    # Enumerate reactions
    rxn_chis = []
    for rct_vals_ in itertools.product(*rcts_vals_):
        rct_chis = species.amchis(
            mech.species, vals_=rct_vals_, col_=spc_col_, fill=True
        )

        # Skip if excluded reactants are present
        if any(c in excl_chis for c in rct_chis):
            continue

        for rxn in automol.reac.enum.from_amchis(smarts, rct_chis):
            _, prd_chis = automol.reac.amchis(rxn, stereo=False)
            rxn_chis.append((rct_chis, prd_chis))

    # Form the updated species DataFrame
    chis = list(itertools.chain.from_iterable([*r, *p] for r, p in rxn_chis))
    chis = list(mit.unique_everseen(chis))
    spc_df = species.bootstrap({Species.amchi: chis})
    mech.species = species.update(spc_df, mech.species)
    if src_mech is not None:
        mech.species = species.left_update(mech.species, src_mech.species)

    if skip_rxn_update:
        return mech

    # Form the updated reactions DataFrame
    if not rxn_chis:
        return mech

    rct_chis, prd_chis = zip(*rxn_chis, strict=True)
    name_dct = df_.lookup_dict(mech.species, Species.amchi, Species.name)
    rxn_df = reaction.bootstrap(
        {Reaction.reactants: rct_chis, Reaction.products: prd_chis},
        name_dct=name_dct,
        spc_df=mech.species,
    )
    mech.reactions = reaction.update(rxn_df, mech.reactions)
    if match_src and src_mech is not None:
        mech.reactions = reaction.left_update(mech.reactions, src_mech.reactions)
    return drop_duplicate_reactions(mech)


def replace_unstable_products(
    mech: Mechanism, src_mech: Mechanism | None = None
) -> tuple[Mechanism, Mechanism]:
    """Replace unstable products with their stable counterparts.

    :param mech: Mechanism
    :return: Mechanism without unstable species; mechanism of unstable species
    """
    mech = mech.model_copy()

    # Enumerate instability reactions
    uns_mech = without_reactions(mech)
    uns_mech = _enumerate_reactions(
        uns_mech,
        enum.ReactionSmarts.qooh_instability,
        src_mech=src_mech,
        match_src=True,  # Do not align direction to source mechanism
    )

    # Form dictionary mapping unstable products to stable ones
    name_col = c_.temp()
    uns_rxn_df = uns_mech.reactions
    uns_rxn_df = uns_rxn_df.with_columns(
        polars.col(Reaction.reactants).list.first().alias(name_col)
    )
    uns_dct = dict(uns_rxn_df.select(name_col, Reaction.products).iter_rows())

    # Replace unstable reaction products with stable ones
    #   1. Create columns of stable products for each unstable product
    mech.reactions = mech.reactions.with_columns(
        polars.when(polars.col(Reaction.products).list.contains(name))
        .then(ins)
        .otherwise([])
        .alias(name)
        for name, ins in uns_dct.items()
    )
    #   2. Remove unstable species from product list and concat with stable products
    mech.reactions = mech.reactions.with_columns(
        polars.concat_list(
            polars.col(Reaction.products).list.set_difference(uns_dct.keys()),
            *(polars.col(name) for name in uns_dct.keys()),
        )
    )
    #   3. Combine the replacement columns into a single struct column
    mech.reactions = mech.reactions.with_columns(
        polars.struct(polars.col(name) for name in uns_dct.keys()).alias(
            ReactionUnstable.replaced_unstable
        )
    ).drop(uns_dct.keys())
    assert not set(species_names(mech, rxn_only=True)) & set(uns_dct), mech

    # Remove unstable species and add stable ones
    stable_spc_df = uns_mech.species.filter(
        ~polars.col(Species.name).is_in(uns_dct.keys())
    )
    mech.species = mech.species.filter(~polars.col(Species.name).is_in(uns_dct.keys()))
    mech.species = species.update(mech.species, stable_spc_df)

    # Save mechanism containing only the unstable species
    uns_spc_mech = without_reactions(uns_mech)
    uns_spc_mech.species = uns_mech.species.filter(
        polars.col(Species.name).is_in(uns_dct.keys())
    )
    return mech, uns_spc_mech


# sorting
def with_sort_data(mech: Mechanism) -> Mechanism:
    """Add columns to sort mechanism by species and reactions.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    mech = mech.model_copy()
    mech = without_sort_data(mech)

    # Sort species by formula
    mech.species = species.sort_by_formula(mech.species)

    # Sort reactions by shape and by reagent names
    idx_col = c_.temp()
    mech.reactions = mech.reactions.sort(
        polars.col(Reaction.reactants).list.len(),
        polars.col(Reaction.products).list.len(),
        df_.list_to_struct_expression(mech.reactions, Reaction.reactants),
        df_.list_to_struct_expression(mech.reactions, Reaction.products),
    )
    mech.reactions = df_.with_index(mech.reactions, idx_col)

    # Generate sort data from network
    srt_dct = net_.sort_data(network(mech), idx_col)
    srt_data = [
        {
            idx_col: i,
            ReactionSorted.pes: p,
            ReactionSorted.subpes: s,
            ReactionSorted.channel: c,
        }
        for i, (p, s, c) in srt_dct.items()
    ]
    srt_schema = {idx_col: polars.UInt32, **pandera_.schema(ReactionSorted)}
    srt_df = polars.DataFrame(srt_data, schema=srt_schema)

    # Add sort data to reactions dataframe and sort
    mech.reactions = mech.reactions.join(srt_df, on=idx_col, how="left")
    mech.reactions = mech.reactions.drop(idx_col)
    mech.reactions = mech.reactions.sort(
        ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel
    )
    return mech


def with_fake_sort_data(mech: Mechanism, offset: int = 0) -> Mechanism:
    """Add fake sort columns.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    mech = mech.model_copy()
    mech = without_sort_data(mech)
    mech.reactions = df_.with_index(
        mech.reactions, ReactionSorted.pes, offset=offset + 1
    )
    mech.reactions = mech.reactions.with_columns(
        polars.lit(1).alias(ReactionSorted.subpes),
        polars.lit(1).alias(ReactionSorted.channel),
    )
    mech.reactions = pandera_.impose_schema(ReactionSorted, mech.reactions)
    mech.reactions = reaction.validate(mech.reactions, [Reaction, ReactionSorted])
    return mech


def without_sort_data(mech: Mechanism) -> Mechanism:
    """Add columns to sort mechanism by species and reactions.

    :param mech: Mechanism
    :return: Mechanism with sort columns
    """
    mech = mech.model_copy()
    mech.reactions = mech.reactions.drop(
        ReactionSorted.pes, ReactionSorted.subpes, ReactionSorted.channel, strict=False
    )
    return mech


def with_sorted_reagents(mech: Mechanism) -> Mechanism:
    """Sort reagents in the mechanism alphabetically.

    :param mech: Mechanism
    :return: Mechanism
    """
    mech = mech.model_copy()
    mech.reactions = reaction.with_sorted_reagents(mech.reactions, reversible=False)
    return mech


# comparison
def are_equivalent(mech1: Mechanism, mech2: Mechanism) -> bool:
    """Determine whether two mechanisms are equivalent.

    (Currently too strict -- need to figure out how to handle nested float comparisons
    in Struct columns.)

    Waiting on:
     - https://github.com/pola-rs/polars/issues/11067 (to be used with .unnest())
    and/or:
     - https://github.com/pola-rs/polars/issues/18936

    :param mech1: First mechanism
    :param mech2: Second mechanism
    :return: `True` if they are, `False` if they aren't
    """
    same_reactions = mech1.reactions.equals(mech2.reactions)
    same_species = mech1.species.equals(mech2.species)
    return same_reactions and same_species


# read/write
def string(mech: Mechanism) -> str:
    """Write mechanism to JSON string.

    :param mech: Mechanism
    :return: Mechanism JSON string
    """
    return mech.model_dump_json()


def from_string(mech_str: str) -> Mechanism:
    """Read mechanism from JSON string.

    :param mech_str: Mechanism JSON string
    :return: Mechanism
    """
    return Mechanism.model_validate_json(mech_str)


# display
def display(
    mech: Mechanism,
    stereo: bool = True,
    color_subpes: bool = True,
    species_centered: bool = False,
    exclude_formulas: Sequence[str] = net_.DEFAULT_EXCLUDE_FORMULAS,
    height: str = "1000px",
    out_name: str = "net.html",
    out_dir: str = ".automech",
    open_browser: bool = True,
) -> None:
    """Display mechanism as reaction network.

    :param mech: Mechanism
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param color_subpes: Add distinct colors to different PESs
    :param species_centered: Display as species-centered network?
    :param exclude_formulas: If species-centered, exclude these species from display
    :param height: Control height of frame
    :param out_name: Name of HTML file for network visualization
    :param out_dir: Name of directory for saving network visualization
    :param open_browser: Whether to open browser automatically
    """
    net_.display(
        net=network(mech),
        stereo=stereo,
        color_subpes=color_subpes,
        species_centered=species_centered,
        exclude_formulas=exclude_formulas,
        height=height,
        out_name=out_name,
        out_dir=out_dir,
        open_browser=open_browser,
    )


def display_species(
    mech: Mechanism,
    ids: Collection[str | int] | None = None,
    spc_vals_: Sequence[str] | None = None,
    spc_key_: str | Sequence[str] = Species.name,
    stereo: bool = True,
    keys: tuple[str, ...] = (
        Species.name,
        Species.smiles,
    ),
):
    """Display species in mechanism.

    :param mech: Mechanism
    :param vals_: Species column value(s) list for selection
    :param spc_key_: Species column key(s) for selection
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    """
    # Read in mechanism data
    spc_df = mech.species

    # Select the requested ids, if any
    if ids is not None:
        ids = list(map(int, ids))
        tmp_col = c_.temp()
        spc_df = spc_df.with_row_index(name=tmp_col, offset=1).filter(
            polars.col(tmp_col).is_in(ids)
        )

    if spc_vals_ is not None:
        spc_df = species.filter(spc_df, vals_=spc_vals_, col_=spc_key_)
        id_ = [spc_key_] if isinstance(spc_key_, str) else spc_key_
        keys = [*id_, *(k for k in keys if k not in id_)]

    def _display_species(chi, *vals):
        """Display a species."""
        # Print requested information
        for key, val in zip(keys, vals, strict=True):
            print(f"{key}: {val}")

        automol.amchi.display(chi, stereo=stereo)

    # Display requested reactions
    df_.map_(spc_df, (Species.amchi, *keys), None, _display_species)


Number: TypeAlias = float | int


def display_reactions(
    mech: Mechanism,
    eqs: Collection | None = None,
    chans: Collection | None = None,
    stereo: bool = True,
    spc_cols: Sequence[str] = (Species.smiles,),
    t_range: tuple[Number, Number] = (400, 1250),
    p: Number = 1,
    label: str = "This work",
    comp_mechs: Sequence[Mechanism] = (),
    comp_labels: Sequence[str] = (),
    comp_stereo: bool | Sequence[bool] = True,
):
    """Display reactions in mechanism.

    :param mech: Mechanism
    :param eqs: Optionally, specify specific equations to visualize
    :param chans: Optionally, specify specific channels to visualize, e.g. "1: 2"
    :param stereo: Include stereochemistry in species drawings?, defaults to True
    :param keys: Keys of extra columns to print
    :param spc_cols: Optionally, translate reactant and product names into these
        species dataframe values
    :param t_range: Range of temperatures for Arrhenius plot
    :param p: Pressure for Arrhenius plot
    :param label: Label for rate comparison
    :param comp_mechs: Mechanisms to compare rates with by label
    :param comp_stereo: Whether to include stereo in matching reactions
        Use sequence to specify for each mechanism
    """
    # Add rate objects with comparisons
    ncomps = len(comp_mechs)
    assert len(comp_labels) == ncomps, f"{comp_labels} !~ ncomps"
    obj_col = c_.temp()
    comp_cols = [c_.temp() for _ in range(ncomps)]
    mech = with_rate_objects(
        mech,
        col=obj_col,
        comp_mechs=comp_mechs,
        comp_cols=comp_cols,
        comp_stereo=comp_stereo,
        fill=True,
    )

    # Read in mechanism data
    spc_df = mech.species
    rxn_df = mech.reactions

    # Select the requested equations, if any
    if eqs is not None:
        tmp_col = c_.temp()
        rxn_df = reaction.with_equation_match_column(rxn_df, tmp_col, eqs)
        rxn_df = rxn_df.filter(tmp_col).drop(tmp_col)

    # Select the requested channels, if any
    if chans is not None:
        tmp_col = c_.temp()
        cols = [ReactionSorted.pes, ReactionSorted.channel]
        vals_lst = [tuple(map(int, chan.split(":"))) for chan in chans]
        assert all(c in rxn_df.columns for c in cols), f"{cols} !<= {rxn_df.columns}"
        rxn_df = df_.with_match_index_column(rxn_df, tmp_col, vals_=vals_lst, col_=cols)
        rxn_df = rxn_df.filter(polars.col(tmp_col).is_not_null()).drop(tmp_col)

    # 2. Add AMChI translation + others that were requested
    spc_cols_ = [Species.amchi, *spc_cols]
    rct_cols = [c_.temp() for _ in range(len(spc_cols_))]
    prd_cols = [c_.temp() for _ in range(len(spc_cols_))]
    names = spc_df.get_column(Species.name)
    for spc_col, rct_col, prd_col in zip(spc_cols_, rct_cols, prd_cols, strict=True):
        vals = spc_df.get_column(spc_col)
        rxn_df = reaction.translate_reagents(
            rxn_df, names, vals, rct_col=rct_col, prd_col=prd_col
        )

    def _display_reaction(rate: ac.rate.Reaction, *vals):
        comp_rates, vals = vals[:ncomps], vals[ncomps:]
        idxs = [i for i, r in enumerate(comp_rates) if r is not None]
        comp_rates_ = [comp_rates[i] for i in idxs]
        comp_labels_ = [comp_labels[i] for i in idxs]

        assert len(vals) % 2 == 0, "Expected even number of values"
        rxn_chis, *rxn_vals_lst = list(zip(*mit.divide(2, vals), strict=True))
        eq = ac.rate.chemkin_equation(rate)
        print()
        print("*********")
        print(f"Reaction: {eq}")

        # Print the requested translations
        print("Translations:")
        for spc_col, (rct_vals, prd_vals) in zip(spc_cols, rxn_vals_lst, strict=True):
            indent_print(f"{spc_col}:")
            indent_print(f"reactants = {rct_vals}", n=2)
            indent_print(f"products = {prd_vals}", n=2)

        print("Rate parameters:")
        indent_print(f"{label}:")
        indent_print(ac.rate.chemkin_string(rate), n=2)
        for comp_label, comp_rate in zip(comp_labels_, comp_rates_, strict=True):
            indent_print(f"{comp_label}:")
            indent_print(ac.rate.chemkin_string(comp_rate), n=2)

        # Display the reaction
        automol.amchi.display_reaction(*rxn_chis, stereo=stereo)

        # Display the Arrhenius plot
        ipy_display(
            ac.rate.display(
                [rate, *comp_rates_],
                label=["This work", *comp_labels_],
                T_range=t_range,
                P=p,
            )
        )

    # Sort by the number of non-null comparison columns
    rxn_df = rxn_df.sort([polars.col(c).is_null() for c in comp_cols])

    # Display requested reactions
    cols = [obj_col, *comp_cols, *rct_cols, *prd_cols]
    df_.map_(rxn_df, cols, None, _display_reaction)


# Helpers
def indent_print(text: str, n: int = 1) -> None:
    """Indent text by a number of spaces and print.

    :param text: Text
    :param n: Number of double-spaced indentations
    """
    return print(indent(text, n=n))


def indent(text: str, n: int = 1) -> str:
    """Indent text by a number of spaces.

    :param text: Text
    :param n: Number of double-spaced indentations
    :return: Text
    """
    return textwrap.indent(text, "  " * n)
