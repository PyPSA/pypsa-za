import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.features, rasterio.mask
import rasterstats
import shapely.geometry

area_crs = snakemake.config["crs"]["area_crs"]

# Translate the landuse file into a raster of percentages of available area
landusetype_percent = snakemake.config['respotentials']['landusetype_percent'][snakemake.wildcards.tech]

with rasterio.open(snakemake.input.landuse) as src, rasterio.open(snakemake.output.raster, 'w', **src.meta) as dst:

    resareas = gpd.read_file(snakemake.input.resarea).to_crs(src.crs)
    regions = gpd.read_file(snakemake.input.supply_regions).to_crs(src.crs)

    stats = []

    for region in regions.itertuples():
        resareas_b = resareas.intersects(region.geometry)
        if not resareas_b.any():
            dst.write_band(1, dst_data, window=window)
            stats.append({'mean': 0.})
            continue

        minx, miny, maxx, maxy = region.geometry.bounds
        minx -= (maxx - minx)*0.05
        maxx += (maxx - minx)*0.05
        miny -= (maxy - miny)*0.05
        maxy += (maxy - miny)*0.05

        window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        box = shapely.geometry.box(minx, miny, maxx, maxy)
        transform = rasterio.windows.transform(window, src.transform)

        src_data = src.read(1, window=window)
        dst_data = np.zeros_like(src_data)

        for grid_codes, value in landusetype_percent:
            dst_data.ravel()[np.in1d(src_data.ravel(), grid_codes)] = value

        mask = rasterio.mask.geometry_mask(resareas.loc[resareas_b, 'geometry'], dst_data.shape, transform)
        dst_data = np.ma.array(dst_data, mask=mask, fill_value=0).filled()

        dst.write_band(1, dst_data, window=window)

        stats.extend(rasterstats.zonal_stats(region.geometry, dst_data, affine=transform,
                                             nodata=-999, stats='mean'))

    stats = pd.DataFrame(stats)

    stats['area_ratio'] = stats.pop('mean') / 100
    stats['area'] = regions.to_crs(area_crs).area/1e6 # albert equal area has area in m^2
    stats['available_area'] = stats['area_ratio'] * stats['area']

    stats.set_index(regions.name).to_csv(snakemake.output.area)
