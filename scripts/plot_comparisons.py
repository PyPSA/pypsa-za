import os
import pandas as pd
from itertools import product

opts = snakemake.config['plotting']
scenarios = snakemake.config['scenario'].copy()

plot_dir = snakemake.params.plot_dir
param = snakemake.wildcards.param
param_values = scenarios.pop(snakemake.wildcards.param)

cost_df = (pd.read_csv(snakemake.input.costs2,
                       index_col=[0,1,2],
                       header=list(range(len(scenarios))))
           .reset_index(level=0, drop=True))
#cost_df.index.rename(['components', 'capmarg', 'tech'], inplace=True)
cost_df = (cost_df.loc['capital']
           .add(cost_df.loc['marginal']
                .rename({s: s+' marginal' for s in opts['conv_techs']}),
                fill_value=0.))

tmpl = snakemake.params.tmpl.replace('[', '{').replace(']', '}')
for vals in product(*scenarios.values()):
    sc = dict(zip(scenarios.keys(), vals))
    sc[param] = "-"
    fn = tmpl.format(**sc)
    c = cost_df.xs(key=vals, level=scenarios.keys())
    fig, ax = plt.subplots()
    c.rename(nice_names).plot.bar(stacked=True, color=itemgetter(c.index)(opts['tech_colors']), ax=ax)
    for ext in snakemake.params.exts:
        fig.savefig(os.path.join(plot_dir, fn + '.' + ext))
