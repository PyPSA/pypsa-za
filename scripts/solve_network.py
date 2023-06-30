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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development

logger = logging.getLogger(__name__)

def local_ilopf(
    n,
    snapshots=None,
    msq_threshold=0.05,
    min_iterations=1,
    max_iterations=100,
    track_iterations=False,
    **kwargs,
):
    """
    Iterative linear optimization updating the line parameters for passive AC
    and DC lines. This is helpful when line expansion is enabled. After each
    sucessful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, default 0.05
        Maximal mean square difference between optimized line capacity of
        the current and the previous iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the iterative optimization stops
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    **kwargs
        Keyword arguments of the lopf function which runs at each iteration
    """

    n.lines["carrier"] = n.lines.bus0.map(n.buses.carrier)
    ext_i = get_extendable_i(n, "Line")
    typed_i = n.lines.query('type != ""').index
    ext_untyped_i = ext_i.difference(typed_i)
    ext_typed_i = ext_i.intersection(typed_i)
    base_s_nom = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.bus0.map(n.buses.v_nom)
    )
    n.lines.loc[ext_typed_i, "num_parallel"] = (n.lines.s_nom / base_s_nom)[ext_typed_i]

    def update_line_params(n, s_nom_prev):
        factor = n.lines.s_nom_opt / s_nom_prev
        for attr, carrier in (("x", "AC"), ("r", "DC")):
            ln_i = n.lines.query("carrier == @carrier").index.intersection(
                ext_untyped_i
            )
            n.lines.loc[ln_i, attr] /= factor[ln_i]
        ln_i = ext_i.intersection(typed_i)
        n.lines.loc[ln_i, "num_parallel"] = (n.lines.s_nom_opt / base_s_nom)[ln_i]

    def msq_diff(n, s_nom_prev):
        lines_err = (
            np.sqrt((s_nom_prev - n.lines.s_nom_opt).pow(2).mean())
            / n.lines["s_nom_opt"].mean()
        )
        logger.info(
            f"Mean square difference after iteration {iteration} is " f"{lines_err}"
        )
        return lines_err

    def save_optimal_capacities(n, iteration, status):
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f"{attr}_opt_{iteration}"] = n.df(c)[f"{attr}_opt"]
        setattr(n, f"status_{iteration}", status)
        setattr(n, f"objective_{iteration}", n.objective)
        n.iteration = iteration
        n.global_constraints = n.global_constraints.rename(
            columns={"mu": f"mu_{iteration}"}
        )

    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[n.branch_components].items():
            n.df(c)[f"{attr}_opt_0"] = n.df(c)[f"{attr}"]
    iteration = 1
    kwargs["store_basis"] = True
    diff = msq_threshold
    while diff >= msq_threshold or iteration < min_iterations:
        if iteration > max_iterations:
            logger.info(
                f"Iteration {iteration} beyond max_iterations "
                f"{max_iterations}. Stopping ..."
            )
            break

        s_nom_prev = n.lines.s_nom_opt.copy() if iteration else n.lines.s_nom.copy()
        kwargs["warmstart"] = snakemake.config['solving']['options']['warmstart']#bool(iteration and ("basis_fn" in n.__dir__()))
        status, termination_condition = network_lopf(n, snapshots, **kwargs)
        assert status == "ok", (
            f"Optimization failed with status {status}"
            f"and termination {termination_condition}"
        )
        if track_iterations:
            save_optimal_capacities(n, iteration, status)
        update_line_params(n, s_nom_prev)
        diff = msq_diff(n, s_nom_prev)
        iteration += 1
    logger.info("Running last lopf with fixed branches (HVDC links and HVAC lines)")
    ext_dc_links_b = n.links.p_nom_extendable & (n.links.carrier == "DC")
    s_nom_orig = n.lines.s_nom.copy()
    p_nom_orig = n.links.p_nom.copy()

    n.lines.loc[ext_i, ["s_nom", "s_nom_extendable"]] = pd.DataFrame(
        {"s_nom": n.lines.loc[ext_i, "s_nom_opt"], "s_nom_extendable": False}
    )
    # n.lines.loc[ext_i, ["s_nom", "s_nom_extendable"]] = (
    #     n.lines.loc[ext_i, "s_nom_opt"],
    #     False,
    # )
    n.links.loc[ext_dc_links_b, ["p_nom", "p_nom_extendable"]] = pd.DataFrame(
        {"p_nom": n.links.loc[ext_dc_links_b, "p_nom_opt"], "p_nom_extendable": False}
    )    
    # n.links.loc[ext_dc_links_b, ["p_nom", "p_nom_extendable"]] = (
    #     n.links.loc[ext_dc_links_b, "p_nom_opt"],
    #     False,
    # )
    kwargs["warmstart"] = False
    network_lopf(n, snapshots, **kwargs)

    # n.lines.loc[ext_i, ["s_nom", "s_nom_extendable"]] = s_nom_orig.loc[ext_i], True
    n.lines.loc[ext_i, ["s_nom", "s_nom_extendable"]] = pd.DataFrame(
        {"s_nom": s_nom_orig.loc[ext_i], "s_nom_extendable": True}
    )
    # n.links.loc[ext_dc_links_b, ["p_nom", "p_nom_extendable"]] = (
    #     p_nom_orig.loc[ext_dc_links_b],
    #     True,
    # )
    n.links.loc[ext_dc_links_b, ["p_nom", "p_nom_extendable"]] = pd.DataFrame(
        {"p_nom": p_nom_orig.loc[ext_dc_links_b], "p_nom_extendable": True}
    )

    ## add costs of additional infrastructure to objective value of last iteration
    obj_links = (
        n.links[ext_dc_links_b].eval("capital_cost * (p_nom_opt - p_nom_min)").sum()
    )
    obj_lines = n.lines.eval("capital_cost * (s_nom_opt - s_nom_min)").sum()
    n.objective += obj_links + obj_lines
    n.objective_constant -= obj_links + obj_lines

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

# Reserve requirement of 1GW for spinning acting reserves from PHS or battery, and 2.2GW of total reserves
def reserves(n, sns):
    
    # Operating reserves    
    model_setup = pd.read_excel(
            snakemake.input.model_file, 
            sheet_name='model_setup',
            index_col=[0]
    ).loc[snakemake.wildcards.model_file]

    reserve_requirements =pd.read_excel(
        snakemake.input.model_file, sheet_name='projected_parameters', index_col=[0,1]
    )

    for reserve_type in ['spinning','total']:
        carriers = snakemake.config["electricity"]["operating_reserves"][reserve_type]
        for y in n.investment_periods:
            lhs=0
            rhs = reserve_requirements.loc[(model_setup['projected_parameters'],reserve_type+'_reserves'),y]
            
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

    ###################################################################################
    # Reserve margin above maximum peak demand in each year
    # The sum of res_margin_carriers multiplied by their assumed constribution factors 
    # must be higher than the maximum peak demand in each year by the reserve_margin value

    peakdemand = n.loads_t.p_set.sum(axis=1).groupby(n.snapshots.get_level_values(0)).max()
    res_margin_carriers = snakemake.config['electricity']['reserve_margin']

    for y in n.investment_periods:
        if reserve_requirements.loc[(model_setup['projected_parameters'],'reserve_margin_active'),y]:    
            active = (
                n.generators.index[n.get_active_assets('Generator',y)]
                .append(n.storage_units.index[n.get_active_assets('StorageUnit',y)])
            ).to_list() 

            exist_capacity=0
            for c in ['Generator','StorageUnit']:
                non_ext_gen_i = n.df(c).index[
                    (n.df(c).carrier.isin(res_margin_carriers)) & 
                    (n.df(c).p_nom_extendable==False) & 
                    (n.df(c).index.isin(active))
                ]
                exist_capacity += (
                    n.df(c).loc[non_ext_gen_i,'p_nom']
                    .mul(n.df(c).loc[non_ext_gen_i,'carrier'].map(res_margin_carriers))
                ).sum()

                ext_gen_i = n.df(c).index[
                    (n.df(c).carrier.isin(res_margin_carriers)) & 
                    (n.df(c).p_nom_extendable==True) & 
                    (n.df(c).index.isin(active))
                ]
                if c =='Generator':
                    lhs = linexpr(
                        (
                            n.df(c).loc[ext_gen_i,'carrier']
                            .map(res_margin_carriers), 
                            get_var(n, c, "p_nom")[ext_gen_i]
                        )
                    ).sum()
                else:
                    lhs += linexpr(
                        (
                            n.df(c).loc[ext_gen_i,'carrier']
                            .map(res_margin_carriers), 
                            get_var(n, c, "p_nom")[ext_gen_i]
                        )
                    ).sum()

            rhs = (peakdemand.loc[y]*(1+
                reserve_requirements.loc[(model_setup['projected_parameters'],'reserve_margin'),y]) 
                - exist_capacity
            )
            define_constraints(n, lhs, ">=", rhs, "reserve_margin", str(y))

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

    if (snakemake.wildcards.regions=='1-supply') | (cf_solving.get("skip_iterations", False)):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            multi_investment_periods=multi_investment_periods,
            extra_functionality=extra_functionality,
            **kwargs
        )
    else:
        local_ilopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
            track_iterations=track_iterations,
            msq_threshold=snakemake.config["solving"]["options"]["msq_threshold"],
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
        snakemake = mock_snakemake(
            'solve_network', 
            **{
                'model_file':'grid-2040',
                'regions':'11-supply',
                'resarea':'redz',
                'll':'copt',
                'opts':'LC',
                'attr':'p_nom'
            }
        )
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
