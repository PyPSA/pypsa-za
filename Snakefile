configfile: "config.yaml"

localrules: all, base_network, add_electricity, add_sectors, extract_summaries

wildcard_constraints:
    mask="[a-zA-Z0-9]+",
    cost="[-a-zA-Z0-9]+",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule all:
    input: "results/summaries"

rule landuse_remove_protected_and_conservation_areas:
    input:
        landuse = "data/Original_UTM35north/sa_lcov_2013-14_gti_utm35n_vs22b.tif",
        protected_areas = "data/SAPAD_OR_2017_Q2/",
        conservation_areas = "data/SACAD_OR_2017_Q2/"
    output: "resources/landuse_without_protected_conservation.tiff"
    benchmark: "benchmarks/landuse_remove_protected_and_conservation_areas"
    threads: 1
    resources: mem_mb=10000
    script: "scripts/landuse_remove_protected_and_conservation_areas.py"

rule landuse_map_to_tech_and_supply_region:
    input:
        landuse = "resources/landuse_without_protected_conservation.tiff",
        supply_regions = "data/supply_regions/supply_regions.shp",
        maskshape = "data/masks/{mask}"
    output:
        raster = "resources/raster_{tech}_percent_{mask}.tiff",
        area = "resources/area_{tech}_{mask}.csv"
    benchmark: "benchmarks/landuse_map_to_tech_and_supply_region/{tech}_{mask}"
    threads: 1
    resources: mem_mb=17000
    script: "scripts/landuse_map_to_tech_and_supply_region.py"

rule inflow_per_country:
    input: EIA_hydro_gen="data/EIA_hydro_generation_2011_2014.csv"
    output: "resources/hydro_inflow.csv"
    benchmark: "benchmarks/inflow_per_country"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/inflow_per_country.py"

rule base_network:
    input:
        supply_regions='data/supply_regions/supply_regions.shp',
        population='data/afripop/ZAF15adjv4.tif',
        centroids='data/supply_regions/centroids.shp'
    output: "networks/base_{opts}"
    benchmark: "benchmarks/base_network_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity:
    input:
        base_network='networks/base_{opts}',
        supply_regions='data/supply_regions/supply_regions.shp',
        load='data/SystemEnergy2009_13.csv',
        wind_pv_profiles='data/Wind_PV_Normalised_Profiles.xlsx',
        wind_area='resources/area_wind_{mask}.csv',
        solar_area='resources/area_solar_{mask}.csv',
        existing_generators="data/Existing Power Stations SA.xlsx",
        hydro_inflow="resources/hydro_inflow.csv",
        tech_costs="data/IRP2016_Inputs_Technology-Costs (PUBLISHED).xlsx"
    output: "networks/elec_{cost}_{mask}_{opts}"
    params: costs_sheetname=lambda w: w.cost
    benchmark: "benchmarks/add_electricity/elec_{mask}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{mask}_{opts}",
        emobility="data/emobility"
    output: "networks/sector_{cost}_{mask}_{sectors}_{opts}"
    benchmark: "benchmarks/add_sectors/sector_{mask}_{sectors}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_sectors.py"

rule solve_network:
    input: network="networks/sector_{network}"
    output: "results/networks/{network}"
    log: gurobi="logs/{network}_gurobi.log", python="logs/{network}_python.log"
    benchmark: "benchmarks/solve_network/{network}"
    threads: 4
    resources: mem_mb=20000 # for electricity only
    script: "scripts/solve_network.py"

rule extract_summaries:
    input:
        expand("results/networks/{cost}_{mask}_{sectors}_{opts}",
               **config['scenario'])
    output: "results/summaries"
    params:
        scenario_tmpl="[cost]_[mask]_[sectors]_[opts]",
        scenarios=config['scenario']
    script: "scripts/extract_summaries.py"


# Local Variables:
# mode: python
# End:
