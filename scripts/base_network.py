# SPDX-FileCopyrightText:  PyPSA-ZA2, PyPSA-ZA, PyPSA-Earth and PyPSA-Eur Authors
# # SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-

"""
Creates the network topology for South Africa from either South Africa's shape file, GCCA map extract for 10 supply regions or 27-supply regions shape file as a PyPSA
network.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    supply_regions:

    electricity:
        voltages:

    lines:
        types:
        s_max_pu:
        under_construction:

    links:
        p_max_pu:
        under_construction:
        include_tyndp:

    transformers:
        x:
        s_nom:
        type:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------

- ``data/bundle/supply_regions/{regions}.shp``:  Shape file for different supply regions.
- ``data/bundle/South_Africa_100m_Population/ZAF15adjv4.tif``: Raster file of South African population from https://hub.worldpop.org/doi/10.5258/SOTON/WP00246
- ``data/num_lines.xlsx``: confer :ref:`lines`


Outputs
-------

- ``networks/base_{model_file}_{regions}.nc``

    .. image:: ../img/base.png
        :scale: 33 %
"""

import networkx as nx
import pandas as pd
import numpy as np
from operator import attrgetter

from vresutils.costdata import annuity

import pypsa
import geopandas as gpd
import pandas as pd
import numpy as np
import pypsa
from _helpers import save_to_geojson

def create_network():
    n = pypsa.Network()
    n.name = 'PyPSA-ZA'
    return n

def load_buses_and_lines(n):
    buses = gpd.read_file(snakemake.input.buses)
    buses.set_index('name',drop=True,inplace=True)
    if snakemake.wildcards.regions!='1-supply':
        lines = gpd.read_file(snakemake.input.lines,index_col=[1])
        lines = lines.drop('index',axis=1)
    else:
        lines = pd.DataFrame(index=[],columns=['name','bus0','bus1','length','num_parallel'])
    return buses, lines

def set_snapshots(n,years):
    snapshots = pd.DatetimeIndex([])
    for y in years:
        year_len = 8784 if (round(y/4,0)-y/4) == 0 else 8760
        period = pd.date_range(start=f"{y}-01-01 00:00", freq='h', periods=year_len)
        period = period[~((period.month == 2) & (period.day == 29))]  # exclude Feb 29 for leap years
        snapshots = snapshots.append(period)
    n.set_snapshots(pd.MultiIndex.from_arrays([snapshots.year, snapshots]))

def set_investment_periods(n,years):
    n.investment_periods = years
    if len(years) > 1:
        n.investment_period_weightings["years"] = list(np.diff(years)) + [5]
        T = 0
        for period, nyears in n.investment_period_weightings.years.items():
            discounts = [(1 / (1 + snakemake.config['costs']['discountrate']) ** t) for t in range(T, T + nyears)]
            n.investment_period_weightings.at[period, "objective"] = sum(discounts)
            T += nyears
        n.investment_period_weightings
    else:
        n.investment_period_weightings["years"] = [1]
        n.investment_period_weightings["objective"] = [1]

def set_line_capacity(lines, parallel_lines, line_config):

    # Calculate the cumlative transfer capacity for each of the parallel lines base on thermal limits and dreating by s_nom
    for row in lines.index:
        row_p = parallel_lines[(parallel_lines['bus0']==lines.loc[row,'bus0']) & (parallel_lines['bus1']==lines.loc[row,'bus1'])]
        transf_cap = 0
        for v in [vn for vn in [220, 275, 400, 765] if vn in row_p.columns]:
            transf_cap += np.sqrt(3) * v * n.line_types.loc[line_config['type'][v], 'i_nom'] * row_p[v]*line_config['s_derating'][v]
        lines.loc[row,'capacity'] = transf_cap.values

    lines['DESIGN_VOL'] = 400
    # drop duplicate rows where bus0, bus1 are same rows
    lines = lines.drop_duplicates(subset=['bus0', 'bus1'])
    return lines

def add_components_to_network(n, buses, lines, line_config):
    line_type = line_config['type'][line_config['v_nom']]
    lines = lines.rename(columns={"capacity": "s_nom_min"})
    lines = lines.assign(s_nom_extendable=True, type=line_type)
    n.import_components_from_dataframe(buses, "Bus")
    n.import_components_from_dataframe(lines, "Line")

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
                        'base_network', 
                        **{
                            'model_file':'grid-2040',
                            'regions':'11-supply',
                        }
                    )

    # Create network and load buses and lines data
    n = create_network()
    buses, lines = load_buses_and_lines(n)
    parallel_lines = pd.read_csv(snakemake.input.parallel_lines,index_col=0)
    parallel_lines.columns = ['bus0','bus1']+list(parallel_lines.columns.drop(['bus0','bus1']).astype(float).astype(int))

    # Set snapshots and investment periods
    years = (
        pd.read_excel(
            snakemake.input.model_file,
            sheet_name="model_setup",
            index_col=0
        )
        .loc[snakemake.wildcards.model_file,"simulation_years"]
    )
    if isinstance(years, int):
        # convert years into a list 
        years = [years]
    else:
        years = list(map(int, years.strip('[]').split(',')))
    
    set_snapshots(n,years)
    set_investment_periods(n,years)
    
    # Set line capacity and add components to network
    line_config = snakemake.config['lines']
    save_to_geojson(buses.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.input.buses)
    buses.drop('geometry',axis=1,inplace=True)
    if not lines.empty:
        lines = set_line_capacity(lines, parallel_lines, line_config)
        save_to_geojson(lines.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.input.lines)
        lines.drop('geometry',axis=1,inplace=True)
    add_components_to_network(n, buses, lines, line_config)
    
    n.export_to_netcdf(snakemake.output[0])

