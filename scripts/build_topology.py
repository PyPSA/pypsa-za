# coding: utf-8

"""
Creates the network topology (buses and lines).


Relevant Settings
-----------------

.. code:: yaml

    snapshots:

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

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------

- ``data/bundle/supply_regions/{regions}.shp``:  Shape file for different supply regions.
- ``data/num_lines.xlsx``: confer :ref:`links`


Outputs
-------
- ``resources/buses_{regions}.geojson``
- ``resources/lines_{regions}.geojson``

"""

import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
# import rasterstats
from operator import attrgetter
# from vresutils.costdata import annuity
# from vresutils.shapes import haversine
import os
import pypsa
from _helpers import save_to_geojson

def convert_lines_to_gdf(lines,centroids):
    gdf = gpd.GeoDataFrame(lines)
    linestrings = []
    for i, row in lines.iterrows():
        point1 = centroids[row['bus0']]
        point2 = centroids[row['bus1']]
        linestring = LineString([point1, point2])
        linestrings.append(linestring)
    gdf.set_geometry(linestrings,inplace=True)
    gdf.set_crs(snakemake.config["crs"]["geo_crs"],inplace=True)
    return gdf

def check_centroid_in_region(regions,centroids):
    idx = regions.index[~centroids.intersects(regions['geometry'])]
    buffered_regions = regions.buffer(-200)
    boundary = buffered_regions.boundary
    for i in idx:
        # Initialize a variable to store the minimum distance
        min_distance = np.inf

        # Iterate over a range of distances along the boundary
        for d in np.arange(0, boundary[i].length, 200):
            # Interpolate a point at the current distance
            p = boundary[i].interpolate(d)
            # Calculate the distance between the centroid and the interpolated point
            distance = centroids[i].distance(p)
            # If the distance is less than the minimum distance, update the minimum distance and the closest point
            if distance < min_distance:
                min_distance = distance
                closest_point = p
        centroids[i] = closest_point
    return centroids


def build_topology():
    # Load num_parallel data and calculate num_parallel column for lines dataframe
    gcca_lines = pd.read_excel(
        snakemake.input.gcca_lines,
        sheet_name = snakemake.wildcards.regions, 
        index_col=0
    )#.set_index(['bus0', 'bus1'])

    # Load supply regions and calculate population per region
    regions = gpd.read_file(snakemake.input.supply_regions,
        layer=snakemake.wildcards.regions,
    ).to_crs(snakemake.config["crs"]["distance_crs"])

    index_column = 'name' if 'name' in regions.columns else 'Name' # some layers use Name or name
    regions = regions.set_index(index_column)

    centroids = regions['geometry'].centroid
    centroids = check_centroid_in_region(regions,centroids)
    centroids = centroids.to_crs(snakemake.config["crs"]["geo_crs"])

    # Find edges between touching regions using spatial join
    lines = gcca_lines[['bus0','bus1']]
    #lines = gpd.sjoin(regions, regions, op='touches')['index_right']
    # lines = lines.reset_index()
    # lines.columns = ['bus0', 'bus1']
    # drop rows from lines where bus0 not in regions.index or bus1 not in regions.index

    lines = lines[lines['bus0'].isin(regions.index) & lines['bus1'].isin(regions.index)]
    # Calculate length of lines
    def haversine_length(row):
        lon1, lat1, lon2, lat2 = map(np.radians, [centroids[row['bus0']].x, centroids[row['bus0']].y, centroids[row['bus1']].x, centroids[row['bus1']].y])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371

    # If lines is empty, return empty dataframes
    if lines.empty:
        lines = pd.DataFrame(index=[],columns=['name','bus0','bus1','length','total_capacity',f"equiv. {int(line_config['v_nom'])}kV num_parallel"])
    else:
        lines['length'] = lines.apply(haversine_length, axis=1) * snakemake.config['lines']['length_factor']
               
    # Initialize buses dataframe
    line_config = snakemake.config['lines']
    v_nom = line_config['v_nom']
    buses = (
        regions.assign(
            x=centroids.map(attrgetter('x')),
            y=centroids.map(attrgetter('y')),
            v_nom=v_nom
        )
    )
    # if column name is not 'name' or 'Name', rename it to 'name'
    if 'name' not in buses.columns:
        buses = buses.rename(columns={'Name': 'name'})


    
    gcca_lines[f"equiv. {int(line_config['v_nom'])}kV num_parallel"] = sum(
        gcca_lines[f"num_parallel_{int(v)}kV_ac"] 
        * (v/line_config['v_nom'])**2
                    for v in (132, 220, 275, 400, 765)
    )

    if not lines.empty:
        lines['num_parallel'] = line_config['s_nom_factor'] * gcca_lines[f"equiv. {int(line_config['v_nom'])}kV num_parallel"]
        # lines = (
        #     lines
        #         .join(gcca_lines[f"equiv. {int(line_config['v_nom'])}kV num_parallel"].rename('num_parallel'), on=['bus0', 'bus1'])
        #         .join(gcca_lines[f"equiv. {int(line_config['v_nom'])}kV num_parallel"].rename("num_parallel_i"), on=['bus1', 'bus0'])
        # )
        # lines['num_parallel'] = line_config['s_nom_factor'] * (lines['num_parallel'].fillna(lines.pop('num_parallel_i'))).fillna(0)     
        # check for lines in gcca_lines that are not in lines, i.e. not joining touching regions, excluding MZ,ZM,BW,SW
        #lines.reset_index(drop=True,inplace=True)
        lines = convert_lines_to_gdf(lines,centroids)
    buses.index.name='name' # ensure consistency for other scripts
    return buses, lines

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'build_topology', 
            **{
                'regions':'30-supply',
            }
        )

    buses, lines = build_topology()
    save_to_geojson(buses.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.buses)
    
    if not lines.empty:
        save_to_geojson(lines.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.lines)
    else:
        save_to_geojson(buses.to_crs(snakemake.config["crs"]["geo_crs"]),snakemake.output.lines) # Dummy file will not get used if single node model  
