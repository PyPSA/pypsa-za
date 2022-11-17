#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: : 2017-2022 Meridian Economics
#
# SPDX-License-Identifier: MIT

"""
The ERA5 grid spacing of ~30km results in predictions of annual average capacity factors in wind locations
in South Africa that are underestimated. This function takes the predicted hourly wind speed at 100m from 
the cutout and scales it so that the average annual matches the high resolution data from the Wind Atlas https://globalwindatlas.info/en.

Within each cell the 75% quantile of the Wind Atlas data is used to scale the ERA5 wind speed. 
"""

import logging
import atlite
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
import shapely
from _helpers import configure_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("apply_wind_correction", 
                                    cutout="RSA-2012-era5")

    cutout = atlite.Cutout(snakemake.input.cutout) 
    cells = cutout.grid

    era5_wnd100m=cutout.data.wnd100m.to_dataframe()
    era5_wnd100m_corrected=era5_wnd100m.copy()

    logging.info(f"Scaling annual wind speed at 100m to match {snakemake.input.wasa_map}.")
    src=rasterio.open(snakemake.input.wasa_map)

    # Correct wind speed in each grid cell of the cutout. This can take up to 10min, but only needs to 
    # be run once for each cutout and then can be disabled. 
    for c in cells.index:
        mm=cells.geometry.bounds.loc[c,:]
        window = rasterio.windows.from_bounds(mm['minx'], mm['miny'], mm['maxx'], mm['maxy'], src.transform)

        box = shapely.geometry.box(mm['minx'], mm['miny'], mm['maxx'], mm['maxy'])
        transform = rasterio.windows.transform(window, src.transform)
        src_data = src.read(1, window=window)
        xpos = cells.loc[c,'x']
        ypos = cells.loc[c,'y']
        era5_wnd100m_corrected.loc[(slice(None),ypos,xpos),'wnd100m']=(np.quantile(src_data,0.75)/era5_wnd100m.loc[(slice(None),ypos,xpos),'wnd100m'].mean()
                                                            *era5_wnd100m.loc[(slice(None),ypos,xpos),'wnd100m'])    

    # Use original wind speed values in cases where the above operation introduces artificaial NaN
    use_era5 = (era5_wnd100m_corrected.wnd100m.notnull()==False) & (era5_wnd100m.wnd100m.notnull()==True)
    era5_wnd100m_corrected.wnd100m[use_era5]=era5_wnd100m.wnd100m[use_era5]
    cutout.data.wnd100m.data=era5_wnd100m_corrected.to_xarray().chunk(cutout.chunks).wnd100m.data

    cutout.to_file(snakemake.output[0])