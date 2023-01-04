import rasterio, rasterio.features
import geopandas as gpd
import os

src = rasterio.open(snakemake.input.landuse)
data = src.read(1)

for n in ('protected_areas', 'conservation_areas'):
    area_dir = snakemake.input[n]

    sh = gpd.read_file(area_dir).to_crs(src.crs)
    rasterio.features.rasterize(sh['geometry'], out=data, transform=src.transform, default_value=0)

with rasterio.open(snakemake.output[0], 'w', **src.meta) as dst:
    dst.write_band(1, data)
