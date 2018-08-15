from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()
dataurl = "https://vfs.fias.science/d/f204668ef2/files/?p="

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

rule build_landuse_remove_protected_and_conservation_areas:
    input:
        landuse = HTTP.remote(dataurl + "/SALandCover_OriginalUTM35North_2013_GTI_72Classes.zip", keep_local=True),
        protected_areas = HTTP.remote(dataurl + "/SAPAD_OR_2017_Q3.zip", keep_local=True),
        conservation_areas = HTTP.remote(dataurl + "/SACAD_OR_2017_Q3.zip", keep_local=True)
    output: "resources/landuse_without_protected_conservation.tiff"
    benchmark: "benchmarks/landuse_remove_protected_and_conservation_areas"
    threads: 1
    resources: mem_mb=10000
    script: "scripts/build_landuse_remove_protected_and_conservation_areas.py"

rule build_landuse_map_to_tech_and_supply_region:
    input:
        landuse = "resources/landuse_without_protected_conservation.tiff",
        supply_regions = "data/supply_regions/supply_regions.shp",
        resarea = lambda w: HTTP.remote(dataurl + '/' + config['data']['resarea'][w.resarea], keep_local=True)
    output:
        raster = "resources/raster_{tech}_percent_{resarea}.tiff",
        area = "resources/area_{tech}_{resarea}.csv"
    benchmark: "benchmarks/build_landuse_map_to_tech_and_supply_region/{tech}_{resarea}"
    threads: 1
    resources: mem_mb=17000
    script: "scripts/build_landuse_map_to_tech_and_supply_region.py"

rule build_population:
    input:
        supply_regions='data/supply_regions/supply_regions.shp',
        afripop=HTTP.remote(dataurl + '/South Africa 100m Population.zip', keep_local=True)
    output: 'resources/population.csv'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/build_population.py"

rule build_inflow_per_country:
    input: EIA_hydro_gen="data/EIA_hydro_generation_2011_2014.csv"
    output: "resources/hydro_inflow.csv"
    benchmark: "benchmarks/inflow_per_country"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/build_inflow_per_country.py"

rule build_topology:
    input:
        supply_regions='data/supply_regions/supply_regions.shp',
        centroids='data/supply_regions/centroids.shp',
        num_lines='data/num_lines.csv'
    output:
        buses='resources/buses.csv',
        lines='resources/lines.csv'
    threads: 1
    script: "scripts/build_topology.py"

rule base_network:
    input:
        buses='resources/buses.csv',
        lines='resources/lines.csv',
        population='resources/population.csv'
    output: "networks/base_{opts}.h5"
    benchmark: "benchmarks/base_network_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

rule add_electricity:
    input:
        base_network='networks/base_{opts}.h5',
        supply_regions='data/supply_regions/supply_regions.shp',
        load=HTTP.remote(dataurl + '/SystemEnergy2009_13.csv', keep_local=True),
        wind_profiles=HTTP.remote('www.csir.co.za/sites/default/files/Documents/Supply%20area%20nomalised%20power%20feed-in%20for%20wind.xlsx', keep_local=True),
        pv_profiles=HTTP.remote('www.csir.co.za/sites/default/files/Documents/Supply%20area%20nomalised%20power%20feed-in%20for%20PV.xlsx', keep_local=True),
        wind_area='resources/area_wind_{resarea}.csv',
        solar_area='resources/area_solar_{resarea}.csv',
        existing_generators="data/Existing Power Stations SA.xlsx",
        hydro_inflow="resources/hydro_inflow.csv",
        tech_costs="data/technology_costs.xlsx"
    output: "networks/elec_{cost}_{resarea}_{opts}.h5"
    benchmark: "benchmarks/add_electricity/elec_{cost}_{resarea}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

rule add_sectors:
    input:
        network="networks/elec_{cost}_{resarea}_{opts}.h5"
        # emobility="data/emobility"
    output: "networks/sector_{cost}_{resarea}_{sectors}_{opts}.h5"
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
        resarea=lambda w: HTTP.remote(dataurl + '/' + config['data']['resarea'][w.resarea])
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
