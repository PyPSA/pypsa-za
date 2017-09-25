# coding: utf-8

import networkx as nx
import pandas as pd
import numpy as np
from operator import attrgetter

import rasterstats
import geopandas as gpd

from vresutils.costdata import annuity
from vresutils.shapes import haversine

import pypsa

def base_network():
    ## Read in regions and calculate population per region

    regions = gpd.read_file(snakemake.input.supply_regions)[['name', 'geometry']]

    # Slighly Moved centroids of NAMAQUALAND and PRETORIA manually so that they are within the shapes
    centroids = gpd.read_file(snakemake.input.centroids).set_index('name')['geometry']

    regions['population'] = pd.DataFrame(rasterstats.zonal_stats(regions['geometry'], snakemake.input.population, stats='sum'))['sum']


    # touching regions are connected by lines, we use nx to take care of all the double countings efficiently

    def edges_between_touching_regions(regions):
        G = nx.Graph()
        G.add_nodes_from(regions.index)
        for r in regions.itertuples():
            neighs = regions.index[regions.touches(r.geometry)]
            G.add_edges_from((r.Index, r2_name) for r2_name in neighs.values)
        return G.edges()


    regions = regions.set_index('name')


    ## Build pypsa network
    line_config = snakemake.config['lines']
    v_nom = line_config['v_nom']
    line_type = line_config['type']

    n = pypsa.Network()
    n.name = 'PyPSA-ZA'
    n.crs = regions.crs

    # Buses from regions
    n.set_snapshots(pd.date_range(snakemake.config['historical_year'], periods=8760, freq='h'))
    n.import_components_from_dataframe(
        regions
        .assign(
            x=centroids.map(attrgetter('x')),
            y=centroids.map(attrgetter('y')),
            v_nom=v_nom
        )
        .drop('geometry', axis=1),
        'Bus'
    )

    # Lines from touching regions
    lines = pd.DataFrame(edges_between_touching_regions(regions), columns=['bus0', 'bus1'])

    discountrate = snakemake.config['costs']['discountrate']
    def asarray(x): return np.asarray(list(map(np.asarray, x)))
    lines['length'] = haversine(asarray(lines.bus0.map(centroids)),
                                asarray(lines.bus1.map(centroids))) * line_config['length_factor']
    lines['capital_cost'] = ((annuity(line_config['lifetime'], discountrate) +
                              line_config.get('fom', 0)) * line_config['overnight_cost'] *
                             lines['length'] / line_config['s_nom_factor'])

    num_lines = pd.read_csv(snakemake.input.num_lines, index_col=0).set_index(['bus0', 'bus1'])
    num_parallel = sum(num_lines['num_parallel_{}'.format(int(v))] * (v/v_nom)**2
                       for v in (275, 400, 765))

    lines = (lines
             .join(num_parallel.rename('num_parallel'), on=['bus0', 'bus1'])
             .join(num_parallel.rename("num_parallel_i"), on=['bus1', 'bus0']))
    lines['num_parallel'] = line_config['s_nom_factor'] * lines['num_parallel'].fillna(lines.pop('num_parallel_i'))
    lines['capacity'] = np.sqrt(3)*v_nom*n.line_types.loc[line_type, 'i_nom']*lines.num_parallel

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
             .loc[lines.num_parallel > 0.5]
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
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        snakemake.input = Dict(supply_regions='../data/external/supply_regions/supply_regions.shp',
                            centroids='../data/external/supply_regions/centroids.shp',
                            population='../data/external/afripop/ZAF15adjv4.tif')
        with open('../config.yaml') as f:
            snakemake.config = yaml.load(f)
        snakemake.output = ['../networks/base/']

    n = base_network()
    n.export_to_hdf5(snakemake.output[0])

