configfile: "config.yaml"

# localrules: all, add_flexibilities

# rule all:
#     input: expand("results/no_grid_expansion/{flex}", flex = config['flexibilities'])
wildcard_constraints:
    mask="[a-zA-Z]+"

rule all:
    input: "results/summaries"

rule landuse_remove_protected_and_conservation_areas:
    input:
        landuse = "data/external/Original_UTM35north/sa_lcov_2013-14_gti_utm35n_vs22b.tif",
        protected_areas = "data/external/SAPAD_OR_2017_Q2/",
        conservation_areas = "data/external/SACAD_OR_2017_Q2/"
    output: "data/internal/landuse_without_protected_conservation.tiff"
    benchmark: "benchmarks/landuse_remove_protected_and_conservation_areas"
    threads: 1
    resources: mem_mb=10000
    script: "scripts/landuse_remove_protected_and_conservation_areas.py"

rule landuse_map_to_tech_and_supply_region:
    input:
        landuse = "data/internal/landuse_without_protected_conservation.tiff",
        supply_regions = "data/external/supply_regions/supply_regions.shp",
        maskshape = "data/external/masks/{mask}"
    output:
        raster = "data/internal/raster_{tech}_percent_{mask}.tiff",
        area = "data/internal/area_{tech}_{mask}.csv"
    benchmark: "benchmarks/landuse_map_to_tech_and_supply_region/{tech}_{mask}"
    threads: 1
    resources: mem_mb=17000
    script: "scripts/landuse_map_to_tech_and_supply_region.py"

rule inflow_per_country:
    input: EIA_hydro_gen="data/external/EIA_hydro_generation_2011_2014.csv"
    output: "data/internal/hydro_inflow.csv"
    benchmark: "benchmarks/inflow_per_country"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/inflow_per_country.py"

rule base_network:
    input:
        supply_regions='data/external/supply_regions/supply_regions.shp',
        population='data/external/afripop/ZAF15adjv4.tif',
        centroids='data/external/supply_regions/centroids.shp'
    output: "networks/base_{opts}"
    benchmark: "benchmarks/base_network_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity:
    input:
        base_network='networks/base_{opts}',
        supply_regions='data/external/supply_regions/supply_regions.shp',
        load='data/external/SystemEnergy2009_13.csv',
        wind_pv_profiles='data/external/Wind_PV_Normalised_Profiles.xlsx',
        wind_area='data/internal/area_wind_{mask}.csv',
        solar_area='data/internal/area_solar_{mask}.csv',
        existing_generators="data/external/Existing Power Stations SA.xlsx",
        hydro_inflow="data/internal/hydro_inflow.csv",
        tech_costs="data/external/IRP2016_Inputs_Technology-Costs (PUBLISHED).xlsx"
    output: "networks/elec_{cost}_{mask}_{opts}"
    params: costs_sheetname=lambda w: w.cost
    benchmark: "benchmarks/add_electricity/elec_{mask}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{mask}_{opts}",
        emobility="data/external/emobility"
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
    resources: mem_mb=15000 # for electricity only
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

# rule add_flexibilities:
#     input:
#         entsoe_gridkit_de = "networks/intermediates/03-cluster-de-update-17aug/pre1-45",
#         cop = "data/internal/DE_cop_2011.csv",
#         heat_demand_timeseries = "data/internal/heat_demand_timeseries.csv",
#         transport_timeseries = "data/internal/transport_timeseries.csv",
#         pop_weight = "data/internal/nodal_pop_share.csv"
#     output: prepared_network = "networks/prepared/{flex}_prepared"
#     script: "scripts/add_flexibilities.py"

# rule run_lopf:
#     input: prepared_network = "networks/prepared/{flex}_prepared"
#     output: solved_network = "results/no_grid_expansion/{flex}"
# #   resources: mem=31000 #31GB
#     script: "scripts/run_lopf.py"

# rule generate_sector_load_series:
#     input:
#         pop_map = "data/internal/DE_pop_layout.csv",
#         emobility = "data/external/emobility",
#         energy_totals_eurostat = "data/external/europe_energy_totals-eurostat.pkl",
#         heating_residential = "data/external/residential_heating_shares.csv",
#         heating_tertiary = "data/external/tertiary_heating_shares.csv",
#         heat_demand = "data/internal/DE_heat_demand_2011.csv",
#         heat_profile = "data/external/heat_load_profile_DK_AdamJensen.csv",
#     output:
#         transport_timeseries = "data/internal/transport_timeseries.csv",
#         heat_demand_timeseries = "data/internal/heat_demand_timeseries.csv",
#         pop_weight = "data/internal/nodal_pop_share.csv"
#     script: "scripts/generate_sector_load_series.py"






#rule analysis_cost_distribution:
#    input: expand("results/no_grid_expansion/{flex}", flex = config['flexibilities'])
#    output: "results/figures/cost_distribution_bar.pdf"
#    script: "scripts/cost_distribution.py"
#
#rule analysis_fact_comparison:
#    input: expand("results/no_grid_expansion/{flex}", flex = config['flexibilities'])
#    output: "results/figures/fact_comparison_bar.pdf"
#    script: "scripts/fact_comparison.py"

# Local Variables:
# mode: python
# End:
