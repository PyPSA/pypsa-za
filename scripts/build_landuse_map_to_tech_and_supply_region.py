import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.features, rasterio.mask
import rasterstats

from zipfile import ZipFile
import tempfile
import os

src = rasterio.open(snakemake.input.landuse)
src_data = src.read(1)
meta = src.meta.copy()

# Write only percent value in allowed landuse gridcells
landusetype_percent = snakemake.config['respotentials']['landusetype_percent'][snakemake.wildcards.tech]
data = np.zeros_like(src_data)
for grid_codes, value in landusetype_percent:
    data.ravel()[np.in1d(src_data.ravel(), grid_codes)] = value

del src_data

with tempfile.TemporaryDirectory() as tempdir:
    resarea_dir = os.path.join(tempdir, 'resarea')
    with ZipFile(snakemake.input.resarea[0]) as zipf:
        zipf.extractall(resarea_dir)

    resareas = gpd.read_file(resarea_dir).to_crs(meta['crs'])
    mask = rasterio.mask.geometry_mask(resareas['geometry'], data.shape, meta['affine'])
    data = np.ma.array(data, mask=mask, fill_value=0).filled()

meta.update(compress='lzw', transform=meta['affine'])
with rasterio.open(snakemake.output.raster, 'w', **meta) as dst:
    dst.write_band(1, data)

regions = gpd.read_file(snakemake.input.supply_regions).to_crs(meta['crs'])
stats = (
    pd.DataFrame(
        rasterstats.zonal_stats(regions.geometry, data, affine=meta['affine'],
                                nodata=-999, stats='mean'))
    .rename(columns={'mean': 'area_ratio'})
    / 100.
)
stats['area'] = regions.to_crs(dict(proj='aea')).area/1e6 # albert equal area has area in m^2
stats['available_area'] = stats['area_ratio'] * stats['area']

stats.set_index(regions.name).to_csv(snakemake.output.area)
