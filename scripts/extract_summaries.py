import os
import pypsa
import pandas as pd
import numpy as np
from itertools import product, chain
from six.moves import map, zip
from six import itervalues, iterkeys

if 'snakemake' not in globals():
    from vresutils import Dict
    import yaml
    snakemake = Dict()
    snakemake.input = ['../results/corridors_E', '../results/redz_E']
    snakemake.output = ['../results/summaries']
    snakemake.params = Dict(scenario_tmpl="{mask}_{sectors}",
                            scenarios=Dict(mask=['corridors', 'redz'], sectors=['E']))
    with open('../config.yaml') as f:
        snakemake.config = yaml.load(f)

opts = snakemake.config['plotting']

def collect_networks():
    basenames = list(map(os.path.basename, snakemake.input))
    networks = []

    for p in (dict(zip(iterkeys(snakemake.params.scenarios), o))
              for o in product(*itervalues(snakemake.params.scenarios))):
        scenario = snakemake.params.scenario_tmpl.replace('[', '{').replace(']', '}').format(**p)
        if scenario in basenames:
            networks.append((scenario, p, snakemake.input[basenames.index(scenario)]))

    return networks



def load_network(fn):
    n = pypsa.Network(fn)

    n.loads["carrier"] = n.loads.bus.map(n.buses.carrier) + " load"
    n.links["carrier"] = (n.links.bus0.map(n.buses.carrier) + "-" + n.links.bus1.map(n.buses.carrier))
    n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)

    n.lines["carrier"] = "AC"
    n.transformers["carrier"] = "AC"

    # #if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    for name in opts['heat_links']:
        n.links.loc[n.links.index.to_series().str.endswith(name), "carrier"] = name

    return n


def _to_mind(df):
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=list(snakemake.params.scenarios.keys()))
    return df.sort_index(axis=1)


## Look at aggregated totals

class p(object):
    def __init__(s):
        s.storage = pd.DataFrame(index=opts['storage_techs'])
        s.gen = pd.DataFrame(index=opts['vre_techs']+opts['conv_techs'])
        s.links = pd.DataFrame(index=opts['heat_links'])
        s.loads = pd.DataFrame(index=opts['load_carriers'])

    def add(s, sn, p, n):
        s.storage[tuple(p.values())] = n.storage_units.groupby("carrier").sum()["p_nom_opt"]
        s.gen[tuple(p.values())] = n.generators.groupby("carrier").sum()["p_nom_opt"]
        s.links[tuple(p.values())] = n.links.groupby("carrier").sum()["p_nom_opt"]
        s.loads[tuple(p.values())] = n.loads_t.p.groupby(n.loads.carrier,axis=1).sum().mean()

    def write(s, group_sum_dir):
        p = _to_mind(pd.concat((s.storage, s.gen, s.links, s.loads)).dropna(axis=0, how='all'))
        p.to_csv(os.path.join(group_sum_dir, "p_nom_opt-summary.csv"))
        return p

class e(object):
    def __init__(s):
        s.nom = pd.DataFrame(index=opts['storage_techs'] + opts['store_techs'])
        s.storage = pd.DataFrame(index=opts['storage_techs'])
        s.store = pd.DataFrame(index=opts['store_techs'])
        s.gen = pd.DataFrame(index=opts['vre_techs'] + opts['conv_techs'])

        # s.ambient = pd.DataFrame(index=['Ambient'])
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

    def write(s, group_sum_dir):
        _to_mind(s.nom).to_csv(os.path.join(group_sum_dir,"e_nom_opt-summary.csv"))
        e = _to_mind(pd.concat((s.gen, s.storage, s.store, s.load)).dropna(axis=0, how='all'))
        e.to_csv(os.path.join(group_sum_dir,"e-summary.csv"))
        return e



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

    def write(s, group_sum_dir):
        df = _to_mind(s.curtailed.dropna(axis=0, how='all'))
        df.to_csv(os.path.join(group_sum_dir, "e_curtailed-summary.csv"))
        return df


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
                                          ("links", opts["heat_links"]),
                                          ("storage_units", opts["storage_techs"]),
                                          ("stores", opts["store_techs"]))))
        index = pd.MultiIndex.from_tuples([(comp, capmarg, carrier)
                                           for (comp, carrier), capmarg in product(combinations, ['capital', 'marginal'])])
        s.costs2 = pd.DataFrame(index=index)
        s.costs = pd.DataFrame(index=(opts['vre_techs'] + opts['conv_techs'] +
                                      [t + ' marginal' for t in opts['conv_techs']] +
                                      opts['heat_links'] + opts['storage_techs'] +
                                      opts['store_techs'] + ['lines']))

    def add(s, sn, pa, n):
        costs = {}
        for c, (p_nom, p_attr)  in zip(n.iterate_components(iterkeys(s.components), skip_empty=False), itervalues(s.components)):
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

    def write(s, group_sum_dir):
        _to_mind(s.costs.dropna(axis=0, how='all')).to_csv(os.path.join(group_sum_dir, "costs-summary.csv"))
        _to_mind(s.costs2.dropna(axis=0, how='all')).to_csv(os.path.join(group_sum_dir, "costs2-summary.csv"))


print("Where are we?")
if __name__ == '__main__':
    print("Creating dir")
    group_sum_dir = snakemake.output[0]
    if not os.path.isdir(group_sum_dir):
        os.mkdir(group_sum_dir)

    summers = [p(), e(), e_curtailed(), costs()]
    networks = collect_networks()
    for scenario, params, fn in networks:
        n = load_network(fn)
        for s in summers: s.add(scenario, params, n)

    for s in summers: s.write(group_sum_dir)
