import pandas as pd

import atlite

from vresutils import shapes as vshapes
from vresutils import hydro as vhydro

if 'snakemake' not in globals():
    from vresutils import Dict
    snakemake = Dict()
    snakemake.input = Dict(EIA_hydro_gen="../data/external/EIA_hydro_generation_2011_2014.csv")
    snakemake.config = dict(hydro_inflow=Dict(countries=['ZA'],
                                              cutout='south_africa-2011-2016'))
    snakemake.output = ['../data/internal/hydro_inflow.csv']

countries = snakemake.config['hydro_inflow']['countries']

cutout = atlite.Cutout(snakemake.config['hydro_inflow']['cutout'])
shapes = pd.Series(vshapes.countries(countries))
shapes.index.rename('countries', inplace=True)

annual_hydro = vhydro.get_eia_annual_hydro_generation(snakemake.input.EIA_hydro_gen).reindex(columns=countries)

inflow = cutout.runoff(shapes=shapes,
                       smooth=True,
                       lower_threshold_quantile=True,
                       normalize_using_yearly=annual_hydro)

inflow.transpose('time', 'countries').to_pandas().to_csv(snakemake.output[0])
