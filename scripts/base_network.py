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
    if len(snakemake.config['years'])==1:
        n.set_snapshots(pd.date_range(snakemake.config['historical_year'], periods=8760, freq='h'))
        n.investment_periods=snakemake.config['years']
    else:
        snapshots = pd.DatetimeIndex([])
        for y in snakemake.config['years']:
            if (round(y/4,0)-y/4)==0:
                year_len=8784
            else:
                year_len=8760
            period = pd.date_range(start ='{}-01-01 00:00'.format(y), 
                                freq ='{}min'.format('60'),
                                periods=year_len/(float('60')/60))
            period = period[~((period.month == 2) & (period.day == 29))] # exclude Feb 29 for leap years
            snapshots = snapshots.append(period) 
        n.set_snapshots(pd.MultiIndex.from_arrays([snapshots.year, snapshots]))
        n.investment_periods=snakemake.config['years']

        n.investment_period_weightings["years"] = list(np.diff(snakemake.config['years'])) + [5]

        T = 0
        for period, nyears in n.investment_period_weightings.years.items():
            discounts = [(1 / (1 + snakemake.config['costs']['discountrate']) ** t) for t in range(T, T + nyears)]
            n.investment_period_weightings.at[period, "objective"] = sum(discounts)
            T += nyears
        n.investment_period_weightings

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
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('base_network', **{'costs':'original',
                            'regions':'27-supply',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC',
                            'attr':'p_nom'})

    n = base_network()
    n.export_to_netcdf(snakemake.output[0])

