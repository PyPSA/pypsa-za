# coding: utf-8

import networkx as nx
import pandas as pd
import numpy as np
from operator import attrgetter

from vresutils.costdata import annuity

import pypsa

def base_network():
    n = pypsa.Network()
    n.name = 'PyPSA-ZA'

    buses = pd.read_csv(snakemake.input.buses, index_col=0)
    lines = pd.read_csv(snakemake.input.lines, index_col=0)

    buses['population'] = pd.read_csv(snakemake.input.population, index_col=0)['population']

    line_config = snakemake.config['lines']
    v_nom = line_config['v_nom']
    line_type = line_config["types"][380.]#line_config['types']
    
    lines['capacity'] = np.sqrt(3)*v_nom*n.line_types.loc[line_type, 'i_nom']*lines.num_parallel

    # Buses from regions
    n.set_snapshots(pd.date_range(snakemake.config['historical_year'], periods=8760, freq='h'))
    n.import_components_from_dataframe(buses, 'Bus')

    if 'T' in snakemake.wildcards.opts.split('-'):
        n.import_components_from_dataframe(
            (lines
             .drop('num_parallel', axis=1)
             .rename(columns={'capacity': 'p_nom_min'})
             .assign(p_nom_extendable=True, p_min_pu=-1)),
            "Link"
        )
    elif 'FL' in snakemake.wildcards.opts.split('-'):
        n.import_components_from_dataframe(
            (lines
             .loc[lines.num_parallel > 0.1]
             .drop('capacity', axis=1)
             .assign(s_nom_extendable=False, type=line_type)),
            "Line"
        )
    else:
        n.import_components_from_dataframe(
            (lines
             .rename(columns={'capacity': 's_nom_min'})
             .assign(s_nom_extendable=True, type=line_type,
                     num_parallel=lambda df: df.num_parallel.clip(lower=0.5))),
            "Line"
        )

    return n

if __name__ == "__main__":
    n = base_network()
    n.export_to_netcdf(snakemake.output[0])

