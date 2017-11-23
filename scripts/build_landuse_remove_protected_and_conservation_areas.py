import rasterio, rasterio.features
import geopandas as gpd

from zipfile import ZipFile
import tempfile
import os

with tempfile.TemporaryDirectory() as tempdir:
    landuse_fn = os.path.join(tempdir, 'sa_lcov_2013-14_gti_utm35n_vs22b.tif')
    with ZipFile(snakemake.input.landuse[0]) as zipf:
        zipf.extract(os.path.basename(landuse_fn),
                     os.path.dirname(landuse_fn))

    src = rasterio.open(landuse_fn)
    data = src.read(1)
    meta = src.meta.copy()

    for n in ('protected_areas', 'conservation_areas'):
        area_dir = os.path.join(tempdir, n)
        with ZipFile(snakemake.input[n][0]) as zipf:
            zipf.extractall(area_dir)

        sh = gpd.read_file(area_dir).to_crs(meta['crs'])
        rasterio.features.rasterize(sh['geometry'], out=data, transform=meta['affine'], default_value=0)

meta.update(compress='lzw', transform=meta['affine'])
with rasterio.open(snakemake.output[0], 'w', **meta) as dst:
    dst.write_band(1, data)
