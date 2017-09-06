import os
import pypsa
import pandas as pd
import numpy as np
from itertools import product, chain
from six.moves import map, zip
from six import itervalues, iterkeys
from collections import OrderedDict as odict

from _helpers import load_network

if 'snakemake' not in globals():
    from vresutils import Dict
    import yaml
    snakemake = Dict()
    snakemake.input = ['../results/networks/CSIR-Expected-Apr2016_corridors_E',
                       '../results/networks/IRP2016-Apr2016_corridors_E']
    snakemake.output = ['../results/summaries']
    snakemake.params = Dict(scenario_tmpl="{cost}_{mask}_{sectors}",
                            scenarios=Dict(cost=['CSIR-Expected-Apr2016',
                                                 'IRP2016-Apr2016'],
                                           mask=['corridors'], sectors=['E']))
    with open('../config.yaml') as f:
        snakemake.config = yaml.load(f)

opts = snakemake.config['plotting']

def collect_networks():
    basenames = list(map(os.path.basename, snakemake.input))
    networks = []

    for p in (odict(zip(iterkeys(snakemake.params.scenarios), o))
              for o in product(*itervalues(snakemake.params.scenarios))):
        scenario = snakemake.params.scenario_tmpl.replace('[', '{').replace(']', '}').format(**p)
        if scenario in basenames:
            networks.append((scenario, p, snakemake.input[basenames.index(scenario)]))

    return networks



group_sum_dir = snakemake.output[0]
if not os.path.isdir(group_sum_dir):
    os.mkdir(group_sum_dir)

def clean_and_save(df, fn, dropna=True):
    if dropna:
        df = df.dropna(axis=0, how='all')
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=list(snakemake.params.scenarios.keys()))
    df = df.sort_index(axis=1)
    df.to_csv(os.path.join(group_sum_dir, fn))
    return df


## Look at aggregated totals

class p(object):
    def __init__(s):
        s.storage = pd.DataFrame(index=opts['storage_techs'])
        s.gen = pd.DataFrame(index=opts['vre_techs']+opts['conv_techs'])
        s.links = pd.DataFrame(index=opts['link_carriers'] + opts['heat_links'] + opts['heat_generators'])
        s.loads = pd.DataFrame(index=opts['load_carriers'])

    def add(s, sn, p, n):
        s.storage[tuple(p.values())] = n.storage_units.groupby("carrier").p_nom_opt.sum()
        s.gen[tuple(p.values())] = n.generators.groupby("carrier").p_nom_opt.sum()
        s.links[tuple(p.values())] = n.links.groupby("carrier").p_nom_opt.sum()
        s.loads[tuple(p.values())] = n.loads_t.p.groupby(n.loads.carrier,axis=1).sum().mean()

    def write(s):
        return clean_and_save(pd.concat((s.storage, s.gen, s.links, s.loads)), "p_nom_opt-summary.csv")

class e(object):
    def __init__(s):
        s.nom = pd.DataFrame(index=opts['storage_techs'] + opts['store_techs'])
        s.storage = pd.DataFrame(index=opts['storage_techs'])
        s.store = pd.DataFrame(index=opts['store_techs'])
        s.gen = pd.DataFrame(index=opts['vre_techs'] + opts['conv_techs'])
        s.load = pd.DataFrame(index=opts['load_carriers'])

    def add(s, sn, p, n):
        s.nom[tuple(p.values())] = pd.concat([
            (n.storage_units["p_nom_opt"]*n.storage_units["max_hours"]).groupby(n.storage_units["carrier"]).sum(),
             n.stores["e_nom_opt"].groupby(n.stores["carrier"]).sum()
        ])

        s.storage[tuple(p.values())] = n.storage_units_t.p.sum().groupby(n.storage_units["carrier"]).sum()
        s.store[tuple(p.values())] = n.stores_t.p.sum().groupby(n.stores["carrier"]).sum()
        s.gen[tuple(p.values())] = n.generators_t.p.sum().groupby(n.generators["carrier"]).sum()

        #s.ambient.loc["Ambient", tuple(p.values())] = -(n.links_t.p0.sum().sum() + n.links_t.p1.sum().sum())
        s.load[tuple(p.values())] = -n.loads_t.p.sum().groupby(n.loads.carrier).sum()

    def write(s):
        clean_and_save(s.nom, "e_nom_opt-summary.csv")
        return clean_and_save(pd.concat((s.gen, s.storage, s.store, s.load)), "e-summary.csv")



## Examine curtailment as % of available VRE energy

class e_curtailed(object):
    def __init__(s):
        s.curtailed = pd.DataFrame()

    def add(s, sn, p, n):
        s.curtailed[tuple(p.values())] = pd.concat([
            ((n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt) - n.generators_t.p.sum())
             .groupby(n.generators.carrier).sum()),
            ((n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum())
             .groupby(n.storage_units.carrier).sum())
        ])

    def write(s):
        return clean_and_save(s.curtailed, "e_curtailed-summary.csv")


class costs(object):
    def __init__(s):
        s.components = dict(Link=("p_nom_opt", "p0"),
                            Generator=("p_nom_opt", "p"),
                            StorageUnit=("p_nom_opt", "p"),
                            Store=("e_nom_opt", "p"),
                            Line=("s_nom_opt", None),
                            Transformer=("s_nom_opt", None))

        combinations=chain(*(product([n], c)
                             for n, c in (("generators", opts['vre_techs'] + opts['conv_techs']),
                                          ("links", opts["heat_links"] + opts["heat_generators"] + opts['link_carriers']),
                                          ("storage_units", opts["storage_techs"]),
                                          ("stores", opts["store_techs"]),
                                          ("lines", opts["AC_carriers"]))))
        index = pd.MultiIndex.from_tuples([(comp, capmarg, carrier)
                                           for (comp, carrier), capmarg in product(combinations, ['capital', 'marginal'])])
        s.costs2 = pd.DataFrame(index=index)
        s.costs = pd.DataFrame(index=(opts['vre_techs'] + opts['conv_techs'] +
                                      [t + ' marginal' for t in opts['conv_techs']] +
                                      opts['heat_links'] + opts['heat_generators'] +
                                      opts['link_carriers'] + opts['storage_techs'] +
                                      opts['store_techs'] + opts['AC_carriers']))

    def add(s, sn, pa, n):
        costs = {}
        for c, (p_nom, p_attr) in zip(n.iterate_components(iterkeys(s.components), skip_empty=False), itervalues(s.components)):
            costs[(c.list_name, 'capital')] = (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
            if p_attr is not None:
                p = c.pnl[p_attr].sum()
                if c.name == 'StorageUnit':
                    p = p.loc[p > 0]
                costs[(c.list_name, 'marginal')] = (p*c.df.marginal_cost).groupby(c.df.carrier).sum()
        costs = pd.concat(costs)

        s.costs2[tuple(pa.values())] = costs

        costs = costs.reset_index(level=0, drop=True)
        s.costs[tuple(pa.values())] = costs['capital'].add((costs['marginal']
                                                            .rename(columns={t: t + ' marginal' for t in opts['conv_techs']})),
                                                           fill_value=0.)

    def write(s):
        clean_and_save(s.costs, "costs-summary.csv")
        return clean_and_save(s.costs2, "costs2-summary.csv")


if __name__ == '__main__':
    summers = [p(), e(), e_curtailed(), costs()]
    networks = collect_networks()
    for scenario, params, fn in networks:
        n = load_network(fn)
        for s in summers: s.add(scenario, params, n)

    for s in summers: s.write()
