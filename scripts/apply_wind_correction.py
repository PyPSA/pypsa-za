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
import rioxarray
import shapely
#from _helpers import configure_logging
import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning) 
logger = logging.getLogger(__name__)
from progressbar import progressbar

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("apply_wind_correction", 
                                    cutout="RSA-2012-era5")

    cutout = atlite.Cutout(snakemake.input.cutout) 
    cells = cutout.grid

    era5_wnd100m=cutout.data.wnd100m.to_dataframe()
    era5_wnd100m_corrected=era5_wnd100m.copy()

    logging.info(f"Scaling annual wind speed at 100m to match {snakemake.input.wasa_map} at each cell in the cutout.")
    
    gwa_data = rioxarray.open_rasterio(snakemake.input.wasa_map)
    gwa_data_clipped = gwa_data.rio.clip_box(16.46, -34.82, 32.94, -22.13)
    
    # wasa=xr.open_dataarray(snakemake.input.wasa_map)
    # clipped_wasa = wasa.rio.clip_box(16.46, -34.82, 32.94, -22.13)

    # Remove GWA data associated with unusable terrain
    terrain_roughness = rioxarray.open_rasterio(snakemake.input.terrain_ruggedness_index)
    terrain_roughness = terrain_roughness.interp(x=gwa_data_clipped.coords['x'], y=gwa_data_clipped.coords['y'], method="linear")
    terrain_roughness.values[(terrain_roughness.values>0)&(terrain_roughness.values<=200)]=1
    terrain_roughness.values[terrain_roughness.values>200]=np.nan
    terrain_roughness.values[terrain_roughness.values<0]=np.nan
    gwa_data_clipped.values = np.multiply(terrain_roughness.values,gwa_data_clipped.values)

    ds=gwa_data_clipped.sel(band=1, x=slice(*cutout.extent[[0,1]]), y=slice(*cutout.extent[[3,2]]))
    ds=ds.where(ds!=-999)
    ds=atlite.gis.regrid(ds,cutout.data.x, cutout.data.y,resampling=rasterio.warp.Resampling.average)
    
    bias_correction=ds/cutout.data.wnd100m.mean("time")
    
    # Correct wind speed in each grid cell of the cutout. This can take up to 10min, but only needs to 
    # be run once for each cutout and then can be disabled. 
    for c in progressbar(cells.index):
        bounds=cells.geometry.bounds.loc[c,:]
        mask_lon = (clipped_wasa.x >= bounds['minx']) & (clipped_wasa.x <= bounds['maxx'])
        mask_lat = (clipped_wasa.y >= bounds['miny']) & (clipped_wasa.y <= bounds['maxy'])
        cell_wasa = clipped_wasa.where(mask_lon & mask_lat, drop=True)
        cell_wasa_mean = np.nanquantile((cell_wasa.values),0.75)
        
        xpos = cells.loc[c,'x']
        ypos = cells.loc[c,'y']

        era5_mean = era5_wnd100m.loc[(slice(None),ypos,xpos),'wnd100m'].mean()
        if (np.isnan(cell_wasa_mean)) | (era5_mean==0):
            era5_wnd100m_corrected.loc[(slice(None),ypos,xpos),'wnd100m']=era5_wnd100m.loc[(slice(None),ypos,xpos),'wnd100m']
        else:
            era5_wnd100m_corrected.loc[(slice(None),ypos,xpos),'wnd100m']=((cell_wasa_mean/era5_mean)
                                                                        *era5_wnd100m.loc[(slice(None),ypos,xpos),'wnd100m'])    

    cutout.data.wnd100m.data=era5_wnd100m_corrected.to_xarray().chunk(cutout.chunks).wnd100m.data
    cutout.to_file(snakemake.output[0])