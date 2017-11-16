configfile: "config.yaml"

localrules: all, base_network, add_electricity, add_sectors, plot_network, scenario_comparions # , extract_summaries

wildcard_constraints:
    resarea="[a-zA-Z0-9]+",
    cost="[-a-zA-Z0-9]+",
    sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule all:
    input:
        expand("results/version-" + str(config['version']) + "/plots/scenario_{param}.html",
               param=list(config['scenario']))

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
        resarea = lambda w: config['data']['resarea'][w.resarea]
    output:
        raster = "resources/raster_{tech}_percent_{resarea}.tiff",
        area = "resources/area_{tech}_{resarea}.csv"
    benchmark: "benchmarks/landuse_map_to_tech_and_supply_region/{tech}_{resarea}"
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
        centroids='data/supply_regions/centroids.shp',
        num_lines='data/num_lines.csv'
    output: "networks/base_{opts}.h5"
    benchmark: "benchmarks/base_network_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity:
    input:
        base_network='networks/base_{opts}.h5',
        supply_regions='data/supply_regions/supply_regions.shp',
        load='data/SystemEnergy2009_13.csv',
        wind_pv_profiles='data/Wind_PV_Normalised_Profiles.xlsx',
        wind_area='resources/area_wind_{resarea}.csv',
        solar_area='resources/area_solar_{resarea}.csv',
        existing_generators="data/Existing Power Stations SA.xlsx",
        hydro_inflow="resources/hydro_inflow.csv",
        tech_costs="data/technology_costs.xlsx"
    output: "networks/elec_{cost}_{resarea}_{opts}.h5"
    benchmark: "benchmarks/add_electricity/elec_{resarea}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{resarea}_{opts}.h5",
        emobility="data/emobility"
    output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.h5"
    benchmark: "benchmarks/add_sectors/sector_{resarea}_{sectors}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_sectors.py"

rule solve_network:
    input: network="networks/sector_{cost}_{resarea}_{sectors}_{opts}.h5"
    output: "results/version-" + str(config['version']) + "/networks/{cost}_{resarea}_{sectors}_{opts}.h5"
    shadow: "shallow"
    log:
        gurobi="logs/{cost}_{resarea}_{sectors}_{opts}_gurobi.log",
        python="logs/{cost}_{resarea}_{sectors}_{opts}_python.log"
    benchmark: "benchmarks/solve_network/{cost}_{resarea}_{sectors}_{opts}"
    threads: 4
    resources: mem_mb=19000 # for electricity only
    script: "scripts/solve_network.py"

rule plot_network:
    input:
        network='results/version-' + str(config['version']) + '/networks/{cost}_{resarea}_{sectors}_{opts}.h5',
        supply_regions='data/supply_regions/supply_regions.shp',
        resarea=lambda w: config['data']['resarea'][w.resarea]
    output:
        only_map=touch('results/version-' + str(config['version']) + '/plots/network_{cost}_{resarea}_{sectors}_{opts}_{attr}'),
        ext=touch('results/version-' + str(config['version']) + '/plots/network_{cost}_{resarea}_{sectors}_{opts}_{attr}_ext')
    params: ext=['png', 'pdf']
    script: "scripts/plot_network.py"

rule scenario_comparison:
    input:
        expand('results/version-{version}/plots/network_{cost}_{resarea}_{sectors}_{opts}_{attr}_ext',
               version=config['version'],
               attr=['p_nom'],
               **config['scenario'])
    output:
       html='results/version-' + str(config['version']) + '/plots/scenario_{param}.html'
    params:
       tmpl="network_[cost]_[resarea]_[sectors]_[opts]_[attr]_ext",
       plot_dir='results/version-' + str(config['version']) + '/plots'
    script: "scripts/scenario_comparison.py"

# extract_summaries and plot_costs needs to be updated before it can be used again
#
# rule extract_summaries:
#     input:
#         expand("results/version-{version}/networks/{cost}_{resarea}_{sectors}_{opts}.h5",
#                version=config['version'],
#                **config['scenario'])
#     output:
#         **{n: "results/version-{version}/summaries/{}-summary.csv".format(n, version=config['version'])
#            for n in ['costs', 'costs2', 'e_curtailed', 'e_nom_opt', 'e', 'p_nom_opt']}
#     params:
#         scenario_tmpl="[cost]_[resarea]_[sectors]_[opts]",
#         scenarios=config['scenario']
#     script: "scripts/extract_summaries.py"

# rule plot_costs:
#     input: 'results/summaries/costs2-summary.csv'
#     output:
#         expand('results/plots/costs_{cost}_{resarea}_{sectors}_{opt}',
#                **dict(chain(config['scenario'].items(), (('{param}')))
#         touch('results/plots/scenario_plots')
#     params:
#         tmpl="results/plots/costs_[cost]_[resarea]_[sectors]_[opt]"
#         exts=["pdf", "png"]
#     scripts: "scripts/plot_costs.py"


# Local Variables:
# mode: python
# End:
