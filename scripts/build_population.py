# coding: utf-8

import pandas as pd
import rasterstats
import geopandas as gpd

def build_population():
    ## Read in regions and calculate population per region
    regions = gpd.read_file(snakemake.input.supply_regions)[['name', 'geometry']]

    population = pd.DataFrame(rasterstats.zonal_stats(regions['geometry'], snakemake.input.population, stats='sum'))['sum']
    population.index = regions['name']
    return population

if __name__ == "__main__":
    pop = build_population()
    pop.to_csv(snakemake.output[0], header=['population'])

