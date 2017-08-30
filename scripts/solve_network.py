import numpy as np
import pandas as pd

import logging
logging.basicConfig(filename=snakemake.log.python, level=logging.INFO)

import pypsa

from _helpers import madd

if 'tmpdir' in snakemake.config['solving']:
    # PYOMO should write its lp files into tmp here
    tmpdir = snakemake.config['solving']['tmpdir']
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir

def prepare_network(n):
    solve_opts = snakemake.config['solving']['options']
    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        madd(n, "Generator", "Load",
             bus=n.buses.index,
             carrier='load',
             marginal_cost=1.0e4 * snakemake.config['costs']['EUR_to_ZAR'],
             # intersect between macroeconomic and surveybased
             # willingness to pay
             # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
             p_nom=1e6)

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            if 'capital_cost' in t.df:
                t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

    return n

def solve_network(n):
    solve_opts = snakemake.config['solving']['options']

    solver_options = snakemake.config['solving']['solver'].copy()
    solver_options['logfile'] = snakemake.log.gurobi
    solver_name = solver_options.pop('name')

    def run_lopf(n):
        status, termination_condition = \
        n.lopf(snapshots=n.snapshots, # extra_functionality=extra_functionality,
               solver_name=solver_name,
               solver_options=solver_options,
               formulation=solve_opts['formulation'])

        assert status == "ok", ("network_lopf did abort with status={} "
                                "and termination_condition={}"
                                .format(status, termination_condition))

        return status, termination_condition

    if not n.lines.s_nom_extendable.any():
        # easy :)
        status, termination_condition = run_lopf(n)
    else:
        # puh: ok, we need to iterate, since there is a relation
        # between s/p_nom and r, x for branches.
        # msq_threshold = 0.1
        msq_threshold = 0.1

        lines_ext_b = n.lines['s_nom_extendable']
        lines = pd.DataFrame(n.lines.loc[lines_ext_b, ['s_nom', 'r', 'x', 'type', 'num_parallel']])

        if (lines_ext_b & (n.lines.type != '')).any():
            # compute original s_nom to have something to compare against
            l = n.lines.loc[lines_ext_b & (n.lines.type != ''), ['type', 'bus0', 'num_parallel']]
            lines.loc[lines.type != '', 's_nom'] = (
                np.sqrt(3) * l['type'].map(n.line_types.i_nom) * l['bus0'].map(n.buses.v_nom) * l['num_parallel']
            )


        def update_line_parameters(n, drop_lines_below=10):
            if drop_lines_below > 0:
                small_lines_i = n.lines.index[n.lines.s_nom_opt < drop_lines_below]
                lines.drop(small_lines_i, inplace=True)
                n.lines.drop(small_lines_i, inplace=True)

            lines_ext_b = n.lines['s_nom_extendable']
            lines['s_nom_opt'] = n.lines.loc[lines_ext_b, 's_nom_opt']
            lines_typed_b = lines.type != ""

            if (~lines_typed_b).any():
                l_b = lines_ext_b & (n.lines.type == '')
                for attr in ('r', 'x'):
                    n.lines.loc[l_b, attr] = (
                        lines[attr].multiply(lines['s_nom']/lines['s_nom_opt'])
                    ).loc[~lines_typed_b]

            if lines_typed_b.any():
                l_b = lines_ext_b & (n.lines.type != '')
                n.lines.loc[l_b, 'num_parallel'] = (
                    lines['num_parallel'].multiply(lines['s_nom_opt']/lines['s_nom'])
                ).loc[lines_typed_b]
                logger.debug("lines.num_parallel={}".format(n.lines.loc[l_b, 'num_parallel']))

        iteration = 1

        lines['s_nom_opt'] = lines['s_nom']
        status, termination_condition = run_lopf(n)

        def msq_diff(n):
            lines_err = np.sqrt(((n.lines.loc[lines_ext_b, 's_nom_opt']/lines['s_nom_opt'] - 1)**2).mean())
            print("Mean square difference after iteration {} is {}".format(iteration, lines_err))
            return lines_err

        # def mae_diff(n):
        #     lines_err = abs(n.lines.loc[lines_ext_b, 's_nom_opt'] - lines['s_nom_opt']).mean()/n.lines.loc[lines_ext_b, 's_nom_opt'].mean()
        #     logger.info("Mean absolute difference after iteration {} is {}".format(iteration, lines_err))
        #     return lines_err

        while msq_diff(n) > msq_threshold or iteration < solve_opts.get('min_iterations', 2):
            if iteration >= solve_opts.get('max_iterations', 999):
                logger.info("Iteration %d beyond max_iterations %d. Stopping ...", iteration, into_t.categories['max_iterations'])
                break

            update_line_parameters(n)
            iteration += 1

            status, termination_condition = run_lopf(n)

    return n

if __name__ == "__main__":
    n = pypsa.Network(csv_folder_name=snakemake.input[0])

    n = prepare_network(n)
    n = solve_network(n)

    n.export_to_csv_folder(snakemake.output[0])
