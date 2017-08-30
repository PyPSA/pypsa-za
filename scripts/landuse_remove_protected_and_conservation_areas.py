import rasterio, rasterio.features
import geopandas as gpd

src = rasterio.open(snakemake.input.landuse)
data = src.read(1)
meta = src.meta.copy()

for n in ('protected_areas', 'conservation_areas'):
    sh = gpd.read_file(snakemake.input[n]).to_crs(meta['crs'])
    rasterio.features.rasterize(sh['geometry'], out=data, transform=meta['affine'], default_value=0)

meta.update(compress='lzw', transform=meta['affine'])
with rasterio.open(snakemake.output[0], 'w', **meta) as dst:
    dst.write_band(1, data)
