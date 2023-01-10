"""
Solves linear optimal power flow for a network iteratively while updating reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: ../img/results.png
        :scale: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the ``pyomo=False`` setting in the :func:`network.lopf` and  :func:`pypsa.linopf.ilopf` function.
Additionally, some extra constraints specified in :mod:`prepare_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors made through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, clean_pu_profiles
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.linopf import (
    define_constraints,
    define_variables,
    get_var,
    ilopf,
    join_exprs,
    linexpr,
    network_lopf,
)

from pypsa.linopt import (
    define_constraints,
    define_variables,
    get_con,
    get_var,
    join_exprs,
    linexpr,
    run_and_read_cbc,
    run_and_read_cplex,
    run_and_read_glpk,
    run_and_read_gurobi,
    run_and_read_highs,
    run_and_read_xpress,
    set_conref,
    write_bound,
    write_constraint,
    write_objective,
)

from pypsa.descriptors import (
    Dict,
    additional_linkports,
    expand_series,
    get_active_assets,
    get_activity_mask,
    get_bounds_pu,
    get_extendable_i,
    get_non_extendable_i,
    nominal_attrs,
)
idx = pd.IndexSlice
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):

    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)
    clean_pu_profiles(n)
    load_shedding = solve_opts.get("load_shedding")
    if load_shedding:
        n.add("Carrier", "Load")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            load_shedding = 1.0e5  # ZAR/MWh
        # intersect between macroeconomic and surveybased
        # willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full)
        # 1e2 is practical relevant, 8e3 good for debugging
        n.madd(
            "Generator",
            buses_i,
            " load_shedding",
            bus=buses_i,
            carrier="load_shedding",
            build_year=n.investment_periods[0],
            lifetime=100,
            #sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e5,#load_shedding,
            p_nom=1e6,  # MW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components(n.one_port_components):
            # TODO: uncomment out to and test noisy_cost (makes solution unique)
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    return n


def add_CCL_constraints(n, sns, config):
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    gen_country = n.generators.bus.map(n.buses.country)
    # cc means country and carrier
    p_nom_per_cc = (
        pd.DataFrame(
            {
                "p_nom": linexpr((1, get_var(n, "Generator", "p_nom"))),
                "country": gen_country,
                "carrier": n.generators.carrier,
            }
        )
        .dropna(subset=["p_nom"])
        .groupby(["country", "carrier"])
        .p_nom.apply(join_exprs)
    )
    minimum = agg_p_nom_minmax["min"].dropna()
    if not minimum.empty:
        minconstraint = define_constraints(
            n, p_nom_per_cc[minimum.index], ">=", minimum, "agg_p_nom", "min"
        )
    maximum = agg_p_nom_minmax["max"].dropna()
    if not maximum.empty:
        maxconstraint = define_constraints(
            n, p_nom_per_cc[maximum.index], "<=", maximum, "agg_p_nom", "max"
        )


def add_EQ_constraints(n, sns, o, scaling=1e-1):
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    lhs_gen = (
        linexpr(
            (n.snapshot_weightings.generators * scaling, get_var(n, "Generator", "p").T)
        )
        .T.groupby(ggrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = (
        linexpr(
            (
                -n.snapshot_weightings.stores * scaling,
                get_var(n, "StorageUnit", "spill").T,
            )
        )
        .T.groupby(sgrouper, axis=1)
        .apply(join_exprs)
    )
    lhs_spill = lhs_spill.reindex(lhs_gen.index).fillna("")
    lhs = lhs_gen + lhs_spill
    define_constraints(n, lhs, ">=", rhs, "equity", "min")


def add_BAU_constraints(n, sns, config):
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    lhs = (
        linexpr((1, get_var(n, "Generator", "p_nom")))
        .groupby(n.generators.carrier)
        .apply(join_exprs)
    )
    define_constraints(n, lhs, ">=", mincaps[lhs.index], "Carrier", "bau_mincaps")


def add_SAFE_constraints(n, sns, config):
    peakdemand = (
        1.0 + config["electricity"]["SAFE_reservemargin"]
    ) * n.loads_t.p_set.sum(axis=1).max()
    conv_techs = config["plotting"]["conv_techs"]
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs"
    ).p_nom.sum()
    ext_gens_i = n.generators.query("carrier in @conv_techs & p_nom_extendable").index
    lhs = linexpr((1, get_var(n, "Generator", "p_nom")[ext_gens_i])).sum()
    rhs = peakdemand - exist_conv_caps
    define_constraints(n, lhs, ">=", rhs, "Safe", "mintotalcap")


def add_operational_reserve_margin_constraint(n, sns, config):

    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    reserve = get_var(n, "Generator", "r")
    lhs = linexpr((1, reserve)).sum(1)

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        renewable_capacity_variables = get_var(n, "Generator", "p_nom")[
            vres_i.intersection(ext_i)
        ]
        lhs += linexpr(
            (-EPSILON_VRES * capacity_factor, renewable_capacity_variables)
        ).sum(1)

    # Total demand at t
    demand = n.loads_t.p.sum(1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    define_constraints(n, lhs, ">=", rhs, "Reserve margin")


def update_capacity_constraint(n,sns):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = get_var(n, "Generator", "p")
    reserve = get_var(n, "Generator", "r")

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = linexpr((1, dispatch), (1, reserve))

    if not ext_i.empty:
        capacity_variable = get_var(n, "Generator", "p_nom")
        lhs += linexpr((-p_max_pu[ext_i], capacity_variable)).reindex(
            columns=gen_i, fill_value=""
        )

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    define_constraints(n, lhs, "<=", rhs, "Generators", "updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.
    """

    define_variables(n, 0, np.inf, "Generator", "r", axes=[sns, n.generators.index])

    add_operational_reserve_margin_constraint(n, config)

    update_capacity_constraint(n)


def add_battery_constraints(n,sns):
    nodes = n.buses.index[n.buses.carrier == "battery"]
    if nodes.empty or ("Link", "p_nom") not in n.variables.index:
        return
    link_p_nom = get_var(n, "Link", "p_nom")
    lhs = linexpr(
        (1, link_p_nom[nodes + " charger"]),
        (
            -n.links.loc[nodes + " discharger", "efficiency"].values,
            link_p_nom[nodes + " discharger"].values,
        ),
    )
    define_constraints(n, lhs, "=", 0, "Link", "charger_ratio")

def min_capacity_factor(n,sns):
    for y in n.investment_periods:
        for carrier in snakemake.config["electricity"]["min_capacity_factor"]:
            # only apply to extendable generators for now
            cf = snakemake.config["electricity"]["min_capacity_factor"][carrier]
            for tech in n.generators[(n.generators.p_nom_extendable==True) & (n.generators.carrier==carrier)].index:
                tech_p_nom=get_var(n, 'Generator', 'p_nom')[tech]
                tech_p_nom=get_var(n, 'Generator', 'p_nom')[tech]
                tech_p=get_var(n, 'Generator', 'p')[tech].loc[y]
                lhs = linexpr((1,tech_p)).sum()+linexpr((-cf*8760,tech_p_nom))
                define_constraints(n, lhs, '>=',0, 'Generators', tech+'_y_'+str(y)+'_min_CF')

# Reserve requirement of 1GW for fast acting reserves from PHS or battery, and 2.2GW of total reserves
def reserves(n, sns):
    model_setup = (pd.read_excel(snakemake.input.model_file, 
                                sheet_name='model_setup',
                                index_col=[0])
                                .loc[snakemake.wildcards.model_file])
    
    reserve_requirements = (pd.read_excel(snakemake.input.model_file, 
                                sheet_name='projected_parameters',
                                index_col=[0,1])
                                .loc[model_setup['projected_parameters']]
                                .loc[['fast_reserves','total_reserves'],:].T)

    for reserve_type in ['fast','total']:
        carriers = snakemake.config["electricity"]["reserves"]["operating_reserves"][reserve_type]
        for y in n.investment_periods:
            lhs=0
            rhs=reserve_requirements.loc[y,reserve_type+'_reserves']
            # Generators
            for tech_type in ['Generator','StorageUnit']:
                active = get_active_assets(n,tech_type,y)
                tech_list = n.df(tech_type).query("carrier == @carriers").index.intersection(active[active].index)
                for tech in tech_list:
                    if tech_type=='Generator':
                        tech_p=get_var(n, tech_type, 'p')[tech].loc[y]
                    elif tech_type=='StorageUnit':
                        tech_p=get_var(n, tech_type, 'p_dispatch')[tech].loc[y]
                    p_max_pu = get_as_dense(n, tech_type, "p_max_pu")[tech].loc[y]
                    if type(lhs)==int:
                        lhs=linexpr((-1,tech_p))
                    else:
                        lhs+=linexpr((-1,tech_p))
                    if n.df(tech_type).p_nom_extendable[tech]==False:
                        tech_p_nom=n.df(tech_type).p_nom[tech]
                        rhs+=-tech_p_nom*p_max_pu
                    else:
                        tech_p_nom=get_var(n, tech_type, 'p_nom')[tech]
                        lhs+=linexpr((p_max_pu,tech_p_nom))
            lhs.index=pd.MultiIndex.from_arrays([lhs.index.year,lhs.index])  
            rhs.index=pd.MultiIndex.from_arrays([rhs.index.year,rhs.index])
            define_constraints(n, lhs, '>=',rhs, 'Reserves_'+str(y)+'_'+reserve_type)

def define_storage_global_constraints(n, sns):
    """
    Defines global constraints for the optimization. Possible types are.

    4. tech_capacity_expansion_limit - linopf only considers generation - so add in storage
        Use this to se a limit for the summed capacitiy of a carrier (e.g.
        'onwind') for each investment period at choosen nodes. This limit
        could e.g. represent land resource/ building restrictions for a
        technology in a certain region. Currently, only the
        capacities of extendable generators have to be below the set limit.
    """

    if n._multi_invest:
        period_weighting = n.investment_period_weightings["years"]
        weightings = n.snapshot_weightings.mul(period_weighting, level=0, axis=0).loc[
            sns
        ]
    else:
        weightings = n.snapshot_weightings.loc[sns]

    def get_period(n, glc, sns):
        period = slice(None)
        if n._multi_invest and not np.isnan(glc["investment_period"]):
            period = int(glc["investment_period"])
            if period not in sns.unique("period"):
                logger.warning(
                    "Optimized snapshots do not contain the investment "
                    f"period required for global constraint `{glc.name}`."
                )
        return period


    # (4) tech_capacity_expansion_limit
    # TODO: Generalize to carrier capacity expansion limit (i.e. also for stores etc.)
    #substr = lambda s: re.sub(r"[\[\]\(\)]", "", s)
    glcs = n.global_constraints.query("type == " '"tech_capacity_expansion_limit"')
    c, attr = "StorageUnit", "p_nom"

    for name, glc in glcs.iterrows():
        period = get_period(n, glc, sns)
        car = glc["carrier_attribute"]
        bus = str(glc.get("bus", ""))  # in pypsa buses are always strings
        ext_i = n.df(c).query("carrier == @car and p_nom_extendable").index
        if bus:
            ext_i = n.df(c).loc[ext_i].query("bus == @bus").index
        ext_i = ext_i[get_activity_mask(n, c, sns)[ext_i].loc[period].any()]

        if ext_i.empty:
            continue

        cap_vars = get_var(n, c, attr)[ext_i]

        lhs = join_exprs(linexpr((1, cap_vars)))
        rhs = glc.constant
        sense = glc.sense

        define_constraints(
            n,
            lhs,
            sense,
            rhs,
            "GlobalConstraint",
            "mu",
            axes=pd.Index([name]),
            spec=name,
        )


def add_local_max_capacity_constraint(n,snapshots):

    c, attr = 'Generator', 'p_nom'
    res = ['onwind', 'solar']
    ext_i = n.df(c)[(n.df(c)["carrier"].isin(res))
                    & (n.df(c)["p_nom_extendable"])].index
    time_valid = snapshots.levels[0]

    active_i = pd.concat([get_active_assets(n,c,inv_p,snapshots).rename(inv_p)
                          for inv_p in time_valid], axis=1).astype(int)

    ext_and_active = active_i.T[active_i.index.intersection(ext_i)]

    if ext_and_active.empty: return

    cap_vars = get_var(n, c, attr)[ext_and_active.columns]

    lhs = (linexpr((ext_and_active, cap_vars)).T
           .groupby([n.df(c).carrier, n.df(c).country]).sum(**agg_group_kwargs).T)

    p_nom_max_w = n.df(c).p_nom_max.div(n.df(c).weight).loc[ext_and_active.columns]
    p_nom_max_t = expand_series(p_nom_max_w, time_valid).T

    rhs = (p_nom_max_t.mul(ext_and_active)
           .groupby([n.df(c).carrier, n.df(c).country], axis=1)
           .max(**agg_group_kwargs))

    define_constraints(n, lhs, "<=", rhs, 'GlobalConstraint', 'res_limit')

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, snapshots, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, snapshots, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, snapshots,config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, snapshots, o)
    add_battery_constraints(n,snapshots)
    min_capacity_factor(n,snapshots)
    define_storage_global_constraints(n, snapshots)
    reserves(n,snapshots)

def solve_network(n, config, opts="", **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)
    multi_investment_periods=isinstance(n.snapshots, pd.MultiIndex)


    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if (snakemake.wildcards.regions=='RSA') | (cf_solving.get("skip_iterations", False)):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            multi_investment_periods=multi_investment_periods,
            extra_functionality=extra_functionality,
            **kwargs
        )
    else:
        ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            multi_investment_periods=multi_investment_periods,
            extra_functionality=extra_functionality,
            **kwargs
        )

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network', **{'model_file':'validation-4',
                            'regions':'RSA',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC-24H',
                            'attr':'p_nom'})
    configure_logging(snakemake)

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        n = pypsa.Network(snakemake.input[0])
        if snakemake.config["augmented_line_connection"].get("add_to_snakefile"):
            n.lines.loc[
                n.lines.index.str.contains("new"), "s_nom_min"
            ] = snakemake.config["augmented_line_connection"].get("min_expansion")
        n = prepare_network(n, solve_opts)
        n = solve_network(
            n,
            config=snakemake.config,
            opts=opts,
            solver_dir=tmpdir,
            solver_logfile=snakemake.log.solver,
            #keep_references=True, #only for debugging when needed
        )

        n.export_to_netcdf(snakemake.output[0])
    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
