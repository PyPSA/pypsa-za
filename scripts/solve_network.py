# import numpy as np
# import pandas as pd
# import logging
# logging.basicConfig(filename=snakemake.log.python, level=logging.INFO)

# import pypsa

# if 'tmpdir' in snakemake.config['solving']:
#     # PYOMO should write its lp files into tmp here
#     tmpdir = snakemake.config['solving']['tmpdir']
#     import os
#     if not os.path.isdir(tmpdir):
#         os.mkdir(tmpdir)
#     from pyutilib.services import TempfileManager
#     TempfileManager.tempdir = tmpdir

# def prepare_network(n):
#     solve_opts = snakemake.config['solving']['options']
#     if 'clip_p_max_pu' in solve_opts:
#         for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
#             df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

#     if solve_opts.get('load_shedding'):
#         n.add("Carrier", "Load")
#         load_i = n.madd("Generator", n.buses.index, suffix=" Load",
#                         bus=n.buses.index,
#                         carrier='load',
#                         marginal_cost=1.0e5 * snakemake.config['costs']['EUR_to_ZAR'],
#                         # intersect between macroeconomic and surveybased
#                         # willingness to pay
#                         # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
#                         p_nom=1e6)

#         if 'SAFE' in snakemake.wildcards.opts.split('-'):
#             # there must be no load shedding in the extra hour introduced in the SAFE scenario
#             load_p_max_pu = pd.DataFrame(1., index=n.snapshots, columns=load_i)
#             load_p_max_pu.iloc[-1, :] = 0.

#             n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, load_p_max_pu], axis=1)

#     if solve_opts.get('noisy_costs'):
#         for t in n.iterate_components():
#             #if 'capital_cost' in t.df:
#             #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
#             if 'marginal_cost' in t.df:
#                 t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

#     if solve_opts.get('nhours'):
#         nhours = solve_opts['nhours']
#         n = n[:solve_opts['nhours'], :]
#         n.snapshot_weightings[:] = 8760./nhours

#     return n

# def apply_time_segmentation(n, segments, solver_name="cbc"):
#     logger.info(f"Aggregating time series to {segments} segments.")
#     try:
#         import tsam.timeseriesaggregation as tsam
#     except:
#         raise ModuleNotFoundError("Optional dependency 'tsam' not found."
#                                 "Install via 'pip install tsam'")

#     p_max_pu_norm = n.generators_t.p_max_pu.max()
#     p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

#     load_norm = n.loads_t.p_set.max()
#     load = n.loads_t.p_set / load_norm
    
#     inflow_norm = n.storage_units_t.inflow.max()
#     inflow = n.storage_units_t.inflow / inflow_norm

#     raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

#     agg = tsam.TimeSeriesAggregation(raw, hoursPerPeriod=len(raw),
#                                     noTypicalPeriods=1, noSegments=int(segments),
#                                     segmentation=True, solver=solver_name)

#     segmented = agg.createTypicalPeriods()

#     weightings = segmented.index.get_level_values("Segment Duration")
#     offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
#     snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

#     n.set_snapshots(pd.DatetimeIndex(snapshots, name='name'))
#     n.snapshot_weightings = pd.Series(weightings, index=snapshots, name="weightings", dtype="float64")
    
#     segmented.index = snapshots
#     n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
#     n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
#     n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

#     return n

# def solve_network(n):
    
#     def add_opts_constraints(n):
#         opts = snakemake.wildcards.opts.split('-')

#         if 'BAU' in opts:
#             mincaps = snakemake.config['electricity']['BAU_mincapacities']
#             def bau_mincapacities_rule(model, carrier):
#                 gens = n.generators.index[n.generators.p_nom_extendable & (n.generators.carrier == carrier)]
#                 return sum(model.generator_p_nom[gen] for gen in gens) >= mincaps[carrier]
#             n.model.bau_mincapacities = pypsa.opt.Constraint(list(mincaps), rule=bau_mincapacities_rule)

#     def fix_lines(n, lines_i=None, links_i=None): # , fix=True):
#         if lines_i is not None and len(lines_i) > 0:
#             s_nom = n.lines.s_nom.where(
#                 n.lines.type == '',
#                 np.sqrt(3) * n.lines.type.map(n.line_types.i_nom) * n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel
#             )
#             for l in lines_i:
#                 n.model.passive_branch_s_nom["Line", l].fix(s_nom.at[l])
#                 # n.model.passive_branch_s_nom[l].fixed = fix
#             if isinstance(n.opt, pypsa.opf.PersistentSolver):
#                 n.opt.update_var(n.model.passive_branch_s_nom)

#         if links_i is not None and len(links_i) > 0:
#             for l in links_i:
#                 n.model.link_p_nom[l].fix(n.links.at[l, 'p_nom'])
#                 # n.model.link_p_nom[l].fixed = fix
#             if isinstance(n.opt, pypsa.opf.PersistentSolver):
#                 n.opt.update_var(n.model.link_p_nom)

#     solve_opts = snakemake.config['solving']['options']

#     solver_options = snakemake.config['solving']['solver'].copy()
#     solver_options['logfile'] = snakemake.log.gurobi
#     solver_name = solver_options.pop('name')

#     def run_lopf(n, allow_warning_status=False, fix_zero_lines=False):
#         if not hasattr(n, 'opt') or not isinstance(n.opt, pypsa.opf.PersistentSolver):
#             pypsa.opf.network_lopf_build_model(n, formulation=solve_opts['formulation'])
#             add_opts_constraints(n)

#             pypsa.opf.network_lopf_prepare_solver(n, solver_name=solver_name)

#         if fix_zero_lines:
#             fix_lines_b = (n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable
#             n.lines.loc[fix_lines_b & (n.lines.type == ''), 's_nom'] = 0.
#             n.lines.loc[fix_lines_b & (n.lines.type != ''), 'num_parallel'] = 0.

#             fix_links_b = (n.links.p_nom_opt == 0.) & n.links.p_nom_extendable
#             n.links.loc[fix_links_b, 'p_nom'] = 0.

#             # WARNING: We are not unfixing these later
#             fix_lines(n, lines_i=n.lines.index[fix_lines_b], links_i=n.links.index[fix_links_b])

#         status, termination_condition = \
#         pypsa.opf.network_lopf_solve(n,
#                                      solver_options=solver_options,
#                                      formulation=solve_opts['formulation'])

#         assert status == "ok" or allow_warning_status and status == 'warning', \
#             ("network_lopf did abort with status={} "
#              "and termination_condition={}"
#              .format(status, termination_condition))

#         return status, termination_condition

#     lines_ext_b = n.lines.s_nom_extendable
#     if lines_ext_b.any():
#         # puh: ok, we need to iterate, since there is a relation
#         # between s/p_nom and r, x for branches.
#         msq_threshold = 0.01
#         lines = pd.DataFrame(n.lines[['r', 'x', 'type', 'num_parallel']])

#         lines['s_nom'] = (
#             np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) * n.lines.bus0.map(n.buses.v_nom) * n.lines.num_parallel
#         ).where(n.lines.type != '', n.lines['s_nom'])

#         lines_ext_typed_b = (n.lines.type != '') & lines_ext_b
#         lines_ext_untyped_b = (n.lines.type == '') & lines_ext_b

#         def update_line_parameters(n, zero_lines_below=10, fix_zero_lines=False):
#             if zero_lines_below > 0:
#                 n.lines.loc[n.lines.s_nom_opt < zero_lines_below, 's_nom_opt'] = 0.
#                 n.links.loc[n.links.p_nom_opt < zero_lines_below, 'p_nom_opt'] = 0.

#             if lines_ext_untyped_b.any():
#                 for attr in ('r', 'x'):
#                     n.lines.loc[lines_ext_untyped_b, attr] = (
#                         lines[attr].multiply(lines['s_nom']/n.lines['s_nom_opt'])
#                     )

#             if lines_ext_typed_b.any():
#                 n.lines.loc[lines_ext_typed_b, 'num_parallel'] = (
#                     lines['num_parallel'].multiply(n.lines['s_nom_opt']/lines['s_nom'])
#                 )
#                 logger.debug("lines.num_parallel={}".format(n.lines.loc[lines_ext_typed_b, 'num_parallel']))

#             if isinstance(n.opt, pypsa.opf.PersistentSolver):
#                 n.calculate_dependent_values()

#                 assert solve_opts['formulation'] == 'kirchhoff', \
#                     "Updating persistent solvers has only been implemented for the kirchhoff formulation for now"

#                 n.opt.remove_constraint(n.model.cycle_constraints)
#                 del n.model.cycle_constraints_index
#                 del n.model.cycle_constraints_index_0
#                 del n.model.cycle_constraints_index_1
#                 del n.model.cycle_constraints

#                 pypsa.opf.define_passive_branch_flows_with_kirchhoff(n, n.snapshots, skip_vars=True)
#                 n.opt.add_constraint(n.model.cycle_constraints)

#         iteration = 1

#         lines['s_nom_opt'] = lines['s_nom']
#         status, termination_condition = run_lopf(n, allow_warning_status=True)

#         def msq_diff(n):
#             lines_err = np.sqrt(((n.lines['s_nom_opt'] - lines['s_nom_opt'])**2).mean())/lines['s_nom_opt'].mean()
#             logger.info("Mean square difference after iteration {} is {}".format(iteration, lines_err))
#             return lines_err

#         min_iterations = solve_opts.get('min_iterations', 2)
#         max_iterations = solve_opts.get('max_iterations', 999)
#         while msq_diff(n) > msq_threshold or iteration < min_iterations:
#             if iteration >= max_iterations:
#                 logger.info("Iteration {} beyond max_iterations {}. Stopping ...".format(iteration, max_iterations))
#                 break

#             update_line_parameters(n)
#             lines['s_nom_opt'] = n.lines['s_nom_opt']
#             iteration += 1

#             # Not really needed, could also be taken out
#             n.export_to_netcdf(snakemake.output[0])

#             status, termination_condition = run_lopf(n, allow_warning_status=True)

#         update_line_parameters(n, zero_lines_below=500)

#     status, termination_condition = run_lopf(n, fix_zero_lines=True, allow_warning_status=True)

#     # Drop zero lines from network
#     zero_lines_i = n.lines.index[(n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable]
#     if len(zero_lines_i):
#         n.mremove("Line", zero_lines_i)
#     zero_links_i = n.links.index[(n.links.p_nom_opt == 0.) & n.links.p_nom_extendable]
#     if len(zero_links_i):
#         n.mremove("Link", zero_links_i)

#     if status != 'ok':
#         # save a backup
#         backup_fn = snakemake.output[0][:-3] + "_suboptimal.h5"
#         n.export_to_netcdf(backup_fn)
#         logger.error("Last solving step returned with status '{}': Aborting. A backup is at {}."
#                      .format(status, backup_fn))
#         raise AssertionError()

#     return n

# if __name__ == "__main__":
#     n = pypsa.Network(snakemake.input[0])
#     n = prepare_network(n)
#     n = apply_time_segmentation(n, 200, solver_name='cbc')
#     n = solve_network(n)

#     n.export_to_netcdf(snakemake.output[0])
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors, 2021 PyPSA-Africa Authors
#
# SPDX-License-Identifier: MIT
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
from _helpers import configure_logging
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
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)


def prepare_network(n, solve_opts):

    if "clip_p_max_pu" in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

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
            " load",
            bus=buses_i,
            carrier="load",
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


def add_CCL_constraints(n, config):
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


def add_EQ_constraints(n, o, scaling=1e-1):
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


def add_BAU_constraints(n, config):
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    lhs = (
        linexpr((1, get_var(n, "Generator", "p_nom")))
        .groupby(n.generators.carrier)
        .apply(join_exprs)
    )
    define_constraints(n, lhs, ">=", mincaps[lhs.index], "Carrier", "bau_mincaps")


def add_SAFE_constraints(n, config):
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


def add_operational_reserve_margin_constraint(n, config):

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


def update_capacity_constraint(n):
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


def add_battery_constraints(n):
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


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to ``pypsa.linopf.network_lopf``.
    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)


def solve_network(n, config, opts="", **kwargs):
    solver_options = config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get("skip_iterations", False):
        network_lopf(
            n,
            solver_name=solver_name,
            solver_options=solver_options,
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
            extra_functionality=extra_functionality,
            **kwargs
        )
    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "solve_network",
            network="elec",
            simpl="",
            clusters="54",
            ll="copt",
            opts="Co2L-1H",
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
        )
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
