import os
from itertools import product
from snakemake.utils import report
from snakemake.report import data_uri
from textwrap import dedent

param = snakemake.wildcards.param
plot_dir = snakemake.params.plot_dir

text = dedent('''
    ===========================
    Comparison of param {param}
    ===========================

    .. contents:: Table of Contents

    ''').format(param=param)

scenarios = snakemake.config['scenario'].copy()
param_values = scenarios.pop(snakemake.wildcards.param)
scenarios['attr'] = ['p_nom']
tmpl = snakemake.params.tmpl.replace('[', '{').replace(']', '}')

links = {}
for vals in product(*scenarios.values()):
    sc = dict(zip(scenarios.keys(), vals))
    headline = ", ".join("{}={}".format(k,v) for k, v in sc.items())
    text += headline + "\n" + '-' * len(headline) + "\n\n"
    for p in param_values:
        sc[param] = p
        fn = tmpl.format(**sc)
        #links[fn] = [os.path.join(plot_dir, fn + '.pdf')]
        text += dedent('''
            .. figure:: {data}
               :scale: 50 %

               {param} = {value}

        ''').format(param=param, value=p, link=fn,
                    data=data_uri(os.path.join(plot_dir, fn + '.png')))

# text += dedent('''

#     Attachments
#     -----------
#     ''')

# text += '{} = {}: {}_\n\n.. image:: {}\n\n'.format(wildcards.param, p, fn, data_uri('results/plots/' + fn + '.png'))

report(text=text, path=snakemake.output.html, stylesheet='report.css', **links)
