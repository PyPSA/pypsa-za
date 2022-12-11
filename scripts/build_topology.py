# coding: utf-8

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from operator import attrgetter
import pypsa
from vresutils.costdata import annuity
from vresutils.shapes import haversine
import os

def save_to_geojson(df, fn):
    if os.path.exists(fn):
        os.unlink(fn)  # remove file if it exists
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(dict(geometry=df))

    # save file if the GeoDataFrame is non-empty
    if df.shape[0] > 0:
        df = df.reset_index()
        schema = {**gpd.io.file.infer_schema(df), "geometry": "Unknown"}
        df.to_file(fn, driver="GeoJSON", schema=schema)
    else:
        # create empty file to avoid issues with snakemake
        with open(fn, "w") as fp:
            pass

def build_topology():
    ## Read in regions and calculate population per region

    regions = gpd.read_file(snakemake.input.supply_regions)[['name', 'geometry']]

    # Slighly Moved centroids of NAMAQUALAND and PRETORIA manually so that they are within the shapes
    centroids = gpd.read_file(snakemake.input.centroids).set_index('name')['geometry']

    # touching regions are connected by lines, we use nx to take care of all the double countings efficiently

    def edges_between_touching_regions(regions):
        G = nx.Graph()
        G.add_nodes_from(regions.index)
        for r in regions.itertuples():
            neighs = regions.index[regions.touches(r.geometry)]
            G.add_edges_from((r.Index, r2_name) for r2_name in neighs.values)
        return G.edges()

    regions = regions.set_index('name')

    line_config = snakemake.config['lines']
    v_nom = line_config['v_nom']

    buses = (regions
             .assign(
                 x=centroids.map(attrgetter('x')),
                 y=centroids.map(attrgetter('y')),
                 v_nom=v_nom
             )
             .drop('geometry', axis=1))

    onshore_buses = (regions
            .assign(
                x=centroids.map(attrgetter('x')),
                y=centroids.map(attrgetter('y')),
            ))

    if snakemake.wildcards.regions=='RSA':
        lines=pd.DataFrame(index=[],columns=['name','bus0','bus1','length','num_parallel'])
    else:
        # Lines from touching regions
        def asarray(x): return np.asarray(list(map(np.asarray, x)))
        lines = pd.DataFrame(list(edges_between_touching_regions(regions)), columns=['bus0', 'bus1'])
        lines['length'] = haversine(asarray(lines.bus0.map(centroids)),
                                    asarray(lines.bus1.map(centroids))) * line_config['length_factor']

        num_lines = pd.read_excel(snakemake.input.num_lines, sheet_name = snakemake.wildcards.regions, index_col=0).set_index(['bus0', 'bus1'])
        num_parallel = sum(num_lines['num_parallel_{}'.format(int(v))] * (v/v_nom)**2
                        for v in (275, 400, 765))

        lines = (lines
                .join(num_parallel.rename('num_parallel'), on=['bus0', 'bus1'])
                .join(num_parallel.rename("num_parallel_i"), on=['bus1', 'bus0']))

        lines['num_parallel'] = line_config['s_nom_factor'] * lines['num_parallel'].fillna(lines.pop('num_parallel_i'))

    return buses, lines, onshore_buses

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_topology', **{'costs':'ambitions',
                            'regions':'RSA',
                            'resarea':'redz',
                            'll':'copt',
                            'opts':'LC-24H',
                            'attr':'p_nom'})


    buses, lines, onshore_buses = build_topology()
    buses.to_csv(snakemake.output.buses)
    lines.to_csv(snakemake.output.lines)
    save_to_geojson(onshore_buses,snakemake.output.regions)
