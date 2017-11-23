# coding: utf-8

import pandas as pd
import rasterstats
import geopandas as gpd

import os
from zipfile import ZipFile
import tempfile

def build_population():
    ## Read in regions and calculate population per region
    regions = gpd.read_file(snakemake.input.supply_regions)[['name', 'geometry']]

    with tempfile.TemporaryDirectory() as tempdir:
        with ZipFile(snakemake.input.afripop[0]) as zipf:
            zipf.extract("ZAF15adjv4.tif", path=tempdir)
        fn = os.path.join(tempdir, "ZAF15adjv4.tif")

        population = pd.DataFrame(rasterstats.zonal_stats(regions['geometry'], fn, stats='sum'))['sum']
        population.index = regions['name']
        return population

if __name__ == "__main__":
    pop = build_population()
    pop.to_csv(snakemake.output[0], header=['population'])

