# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Rasters the vector data of the `Natura 2000.
<https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas onto all
cutout regions.
Relevant Settings
-----------------
.. code:: yaml
    renewable:
        {technology}:
            cutout:
.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`renewable_cf`
Inputs
------
- ``data/bundle/natura/Natura2000_end2015.shp``: `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas.
    .. image:: ../img/natura.png
        :scale: 33 %
Outputs
-------
- ``resources/natura.tiff``: Rasterized version of `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas to reduce computation times.
    .. image:: ../img/natura.png
        :scale: 33 %
Description
-----------
"""

import logging
import os

import atlite
import geopandas as gpd
import rasterio as rio
from _helpers import configure_logging
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)

CUTOUT_CRS = "EPSG:4326"

def get_fileshapes(list_paths, accepted_formats=(".shp",)):
    "Function to parse the list of paths to include shapes included in folders, if any"

    list_fileshapes = []
    for lf in list_paths:
        if os.path.isdir(lf):  # if it is a folder, then list all shapes files contained

            # loop over all dirs and subdirs
            for path, subdirs, files in os.walk(lf):
                # loop over all files
                for subfile in files:
                    # add the subfile if it is a shape file
                    if subfile.endswith(accepted_formats):
                        list_fileshapes.append(os.path.join(path, subfile))

        elif lf.endswith(accepted_formats):
            list_fileshapes.append(lf)

    return list_fileshapes

def determine_cutout_xXyY(cutout_name):
    cutout = atlite.Cutout(cutout_name)
    assert cutout.crs == CUTOUT_CRS
    x, X, y, Y = cutout.extent
    dx, dy = cutout.dx, cutout.dy
    return [x - dx / 2.0, X + dx / 2.0, y - dy / 2.0, Y + dy / 2.0]


def get_transform_and_shape(bounds, res):
    left, bottom = [(b // res) * res for b in bounds[:2]]
    right, top = [(b // res + 1) * res for b in bounds[2:]]
    shape = int((top - bottom) // res), int((right - left) / res)
    transform = rio.Affine(res, 0, left, 0, -res, top)
    return transform, shape

def unify_protected_shape_areas(inputs, geo_crs):
    """
    Iterates thorugh all snakemake rule inputs and unifies shapefiles (.shp) only.
    The input is given in the Snakefile and shapefiles are given by .shp
    Returns
    -------
    unified_shape : GeoDataFrame with a unified "multishape"
    """
    import pandas as pd
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    # Read only .shp snakemake inputs
    shp_files = [string for string in inputs if ".shp" in string]
    assert len(shp_files) != 0, "no input shapefiles given"
    # Create one geodataframe with all geometries, of all .shp files
    total_files = 0
    read_files = 0
    list_shapes = []
    for i in shp_files:
        shp = gpd.read_file(i).to_crs(geo_crs)
        total_files +=1
        try:
            shp.geometry = shp.apply(
                lambda row: make_valid(row.geometry)
                if not row.geometry.is_valid
                else row.geometry,
                axis = 1,
            )
            list_shapes.append(shp)
            read_files += 1
        except:
            print(f'Error reading file{i}')
    shape = gpd.GeoDataFrame(pd.concat(list_shapes)).to_crs(geo_crs)

    # Removes shapely geometry with null values. Returns geoseries.
    shape = shape["geometry"][shape["geometry"].is_valid]

    # Create Geodataframe with crs
    shape = gpd.GeoDataFrame(shape, crs=geo_crs)
    shape = shape.rename(columns={0: "geometry"}).set_geometry("geometry")

    # Unary_union makes out of i.e. 1000 shapes -> 1 unified shape
    unified_shape_file = unary_union(shape["geometry"])

    unified_shape = gpd.GeoDataFrame(geometry=[unified_shape_file], crs=geo_crs)

    return unified_shape


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_natura_raster", cutouts=["cutouts/RSA-2013-era5.nc"])
    configure_logging(snakemake)

    #get_crs
    geo_crs = snakemake.config["crs"]["area_crs"]

    cutouts = snakemake.input.cutouts
    shapefiles = get_fileshapes(snakemake.input)

    xs, Xs, ys, Ys = zip(*(determine_cutout_xXyY(cutout) for cutout in cutouts))
    bounds = transform_bounds(CUTOUT_CRS, geo_crs, min(xs), min(ys), max(Xs), max(Ys))
    transform, out_shape = get_transform_and_shape(bounds, res=100)
    
    # adjusted boundaries
    shapes = unify_protected_shape_areas(shapefiles, geo_crs)
    raster = ~geometry_mask(shapes.geometry, out_shape, transform)
    raster = raster.astype(rio.uint8)

    with rio.open(
        snakemake.output[0],
        "w",
        driver="GTiff",
        dtype=rio.uint8,
        count=1,
        transform=transform,
        crs=geo_crs,
        compress="lzw",
        width=raster.shape[1],
        height=raster.shape[0],
    ) as dst:
        dst.write(raster, indexes=1)