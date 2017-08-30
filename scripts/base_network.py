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
            v_nom=380
        )
        .drop('geometry', axis=1),
        'Bus'
    )

    # Lines from touching regions
    line_costs = snakemake.config['costs']['line']
    def asarray(x): return np.asarray(list(map(np.asarray, x)))
    n.import_components_from_dataframe(
        pd.DataFrame(edges_between_touching_regions(regions), columns=['bus0', 'bus1'])
        .assign(
            length=lambda df: haversine(asarray(df.bus0.map(centroids)),
                                        asarray(df.bus1.map(centroids))) * line_costs['length_factor'],
            s_nom_extendable=True,
            type='Al/St 240/40 4-bundle 380.0'
        ),
        'Line'
    )

    discountrate = snakemake.config['costs']['discountrate']
    n.lines['capital_cost'] = (
        annuity(line_costs['lifetime'], discountrate) * line_costs['overnight'] *
        n.lines['length'] / line_costs['s_nom_factor']
    )

    return n

if __name__ == "__main__":
    n = base_network()
    n.export_to_csv_folder(snakemake.output[0])

