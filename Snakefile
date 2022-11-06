configfile: "config.yaml"

from os.path import normpath, exists, isdir

localrules: all, base_network, add_electricity, plot_network, scenario_comparison # , extract_summaries, add_sectors

ATLITE_NPROCESSES = config["atlite"].get("nprocesses", 4)

wildcard_constraints:
    resarea="[a-zA-Z0-9]+",
    model_file="[-a-zA-Z0-9]+",
    regions="[-+a-zA-Z0-9]+",
    #sectors="[+a-zA-Z0-9]+",
    opts="[-+a-zA-Z0-9]+"

rule all:
    input:
        expand("results/version-" + str(config['version']) + "/plots/scenario_{param}.html",
               param=list(config['scenario']))

if config['enable']['build_land_use']: 
    rule build_landuse_remove_protected_and_conservation_areas:
        input:
            landuse = "data/bundle/SALandCover_OriginalUTM35North_2013_GTI_72Classes/sa_lcov_2013-14_gti_utm35n_vs22b.tif",
            protected_areas = "data/bundle/SAPAD_OR_2017_Q2",
            conservation_areas = "data/bundle/SACAD_OR_2017_Q2"
        output: "resources/landuse_without_protected_conservation.tiff"
        benchmark: "benchmarks/landuse_remove_protected_and_conservation_areas"
        threads: 1
        resources: mem_mb=20000
        script: "scripts/build_landuse_remove_protected_and_conservation_areas.py"

    rule build_landuse_map_to_tech_and_supply_region:
        input:
            landuse = "resources/landuse_without_protected_conservation.tiff",
            supply_regions = "data/supply_regions/supply_regions_{regions}.shp",
            resarea = lambda w: "data/bundle/" + config['data']['resarea'][w.resarea]
        output:
            raster = "resources/raster_{tech}_percent_{regions}_{resarea}.tiff",
            area = "resources/area_{tech}_{regions}_{resarea}.csv"
        benchmark: "benchmarks/build_landuse_map_to_tech_and_supply_region/{tech}_{regions}_{resarea}"
        threads: 1
        resources: mem_mb=20000
        script: "scripts/build_landuse_map_to_tech_and_supply_region.py"

if config['enable']['build_population']: 
    rule build_population:
        input:
            supply_regions='data/supply_regions/supply_regions_{regions}.shp',
            population='data/bundle/South_Africa_100m_Population/ZAF15adjv4.tif'
        output: 'resources/population_{regions}.csv'
        threads: 1
        resources: mem_mb=1000
        script: "scripts/build_population.py"

if config['enable']['build_cutout']:
    rule build_cutout:
        input:
            regions_onshore="resources/offshore_shapes_{regions}.geojson",
            #regions_offshore="resources/" + RDIR + "regions_offshore.geojson",
        output:
            "cutouts/{cutout}_{regions}.nc",
        log:
            "logs/build_cutout/{cutout}_{regions}.log",
        benchmark:
            "benchmarks/build_cutout_{cutout}_{regions}"
        threads: ATLITE_NPROCESSES
        resources:
            mem_mb=ATLITE_NPROCESSES * 1000,
        script:
            "scripts/build_cutout.py"

if not config['hydro_inflow']['disable']:
    rule build_inflow_per_country:
        input: EIA_hydro_gen="data/EIA_hydro_generation_2011_2014.csv"
        output: "resources/hydro_inflow.csv"
        benchmark: "benchmarks/inflow_per_country"
        threads: 1
        resources: mem_mb=1000
        script: "scripts/build_inflow_per_country.py"

if config['enable']['build_topology']: 
    rule build_topology:
        input:
            supply_regions='data/supply_regions/supply_regions_{regions}.shp',
            centroids='data/supply_regions/centroids_{regions}.shp',
            num_lines='data/num_lines.xlsx',
        output:
            buses='resources/buses_{regions}.csv',
            lines='resources/lines_{regions}.csv'
        threads: 1
        script: "scripts/build_topology.py"

rule base_network:
    input:
        buses='resources/buses_{regions}.csv',
        lines='resources/lines_{regions}.csv',
        population='resources/population_{regions}.csv'
    output: "networks/base_{regions}_{opts}.nc"
    benchmark: "benchmarks/base_network_{regions}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/base_network.py"

#rule build_renewable_profiles:
#    input:
#        base_network="networks/base_{regions}_{opts}.nc",
#        regions = 'resources/area_wind_{regions}_{resarea}.csv'#onwind_area='resources/area_wind_{regions}_{resarea}.csv',
#        cutout=lambda w: "cutouts/"+ config["renewable"][w.technology]["cutout"] + ".nc",
#    output: 
#        profile = "resources/profile_{technology}_{regions}_{resarea}_{opts}.nc",
#    log: "logs/build_renewable_profile_{technology}_{regions}_{resarea}_{opts}.log",
#    benchmark: "benchmarks/build_renewable_profiles_{regions}_{resarea}_{opts}",
#    threads: ATLITE_NPROCESSES
#    resources: 
#        mem_mb=ATLITE_NPROCESSES*5000,
#    wildcard_constraints:
#        technology="(?!hydro).*",  # Any technology other than hydro
#    script: "scripts/build_renewable_profiles.py"

rule build_renewable_profiles:
    input:
        base_network="networks/base_{regions}_{opts}.nc",
        regions = 'resources/offshore_shapes_{regions}.geojson',
        natura=lambda w: (
            "resources/landuse_without_protected_conservation.tiff"
            if config["renewable"][w.technology]["natura"]
            else []
        ),
        #country_shapes="resources/" + RDIR + "country_shapes.geojson",
        #offshore_shapes="resources/" + RDIR + "offshore_shapes.geojson",
        cutout=lambda w: "cutouts/"+ config["renewable"][w.technology]["cutout"] + ".nc",
    output:
        profile="resources/profile_{technology}_{regions}.nc",
    log:
        "logs/build_renewable_profile_{technology}_{regions}.log",
    benchmark:
        "benchmarks/build_renewable_profiles_{technology}_{regions}"
    threads: ATLITE_NPROCESSES
    resources:
        mem_mb=ATLITE_NPROCESSES * 5000,
    wildcard_constraints:
        technology="(?!hydro).*",  # Any technology other than hydro
    script:
        "scripts/build_renewable_profiles.py"

rule add_electricity:
    input:
        base_network='networks/base_{regions}_{opts}.nc',
        supply_regions='data/supply_regions/supply_regions_{regions}.shp',
        load='data/bundle/SystemEnergy2009_13.csv',
        onwind_area='resources/area_wind_{regions}_{resarea}.csv',
        solar_area='resources/area_solar_{regions}_{resarea}.csv',
        wind_solar_profiles="resources/wind_solar_profiles_{regions}_{resarea}_{opts}.nc",
        other_re_profiles="resources/other_re_profiles_{regions}_{resarea}_{opts}.nc",
        model_file="data/model_file.xlsx",
        existing_generators_eaf="data/Eskom EAF data.xlsx",
    output: "networks/elec_{model_file}_{regions}_{resarea}_{opts}.nc",
    benchmark: "benchmarks/add_electricity/elec_{model_file}_{regions}_{resarea}_{opts}"
    threads: 1
    resources: mem_mb=1000
    script: "scripts/add_electricity.py"

# rule add_sectors:
#     input:
#         network="networks/elec_{model_file}_{resarea}_{regions}_{opts}.nc"
#         # emobility="data/emobility"
#     output: "networks/sector_{model_file}_{resarea}_{sectors}_{regions}_{opts}.nc"
#     threads: 1
#     resources: mem_mb=1000
#     script: "scripts/add_sectors.py"

rule prepare_network:
    input:
        network="networks/elec_{model_file}_{regions}_{resarea}_{opts}.nc",
        model_file="data/model_file.xlsx",
        onwind_area='resources/area_wind_{regions}_{resarea}.csv',
        solar_area='resources/area_solar_{regions}_{resarea}.csv',
    output:"networks/pre_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
    log:"logs/prepare_network/pre_{model_file}_{regions}_{resarea}_l{ll}_{opts}.log",
    benchmark:"benchmarks/prepare_network/pre_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc",
    threads: 1
    resources:
        mem=4000,
    script:
        "scripts/prepare_network.py"

rule solve_network:
    input: network="networks/pre_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc"
    output: "results/version-" + str(config['version']) + "/networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc"
    shadow: "shallow"
    log:
        solver=normpath(
            "logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_solver.log"
        ),
        python="logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_python.log",
        memory="logs/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}_memory.log",
    benchmark: "benchmarks/solve_network/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}"
    threads: 15
    resources: mem_mb=40000 # for electricity only
    script: "scripts/solve_network.py"


rule plot_network:
    input:
        network='results/version-0.6/networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc',
        model_file="data/model_file.xlsx",
        supply_regions='data/supply_regions/supply_regions_{regions}.shp',
        resarea = lambda w: "data/bundle/" + config['data']['resarea'][w.resarea]
    output:
        only_map='results/version-0.6/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}.{ext}',
        ext='results/version-0.6/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}_ext.{ext}',
    log: 'logs/plot_network/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}.{ext}.log'
    script: "scripts/plot_network.py"

# rule plot_network:

#     input:
#         network='results/version-' + str(config['version']) + '/networks/{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc',
#         supply_regions = "data/supply_regions/supply_regions_{regions}.shp",
#         resarea = lambda w: "data/bundle/" + config['data']['resarea'][w.resarea]
#     output:
#         only_map=touch('results/version-' + str(config['version']) + '/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}'),
#         ext=touch('results/version-' + str(config['version']) + '/plots/{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}_ext')
#     params: ext=['png', 'pdf']
#     script: "scripts/plot_network.py"

rule scenario_comparison:
    input:
        expand('results/version-{version}/plots/network_{model_file}_{regions}_{resarea}_l{ll}_{opts}_{attr}_ext',
               version=config['version'],
               attr=['p_nom'],
               **config['scenario'])
    output:
       html='results/version-' + str(config['version']) + '/plots/scenario_{param}.html'
    params:
       tmpl="network_[cost]_[resarea]_[sectors]_[opts]_[attr]_ext",
       plot_dir='results/version-' + str(config['version']) + '/plots'
    script: "scripts/scenario_comparison.py"

def input_make_summary(w):
    # It's mildly hacky to include the separate costs input as first entry
    return (expand("results/version-" + str(config['version']) + "/networks/{model_file}_{regions}_{resarea}_{sectors}_l{ll}_{opts}.nc",
                   **{k: config["scenario"][k] if getattr(w, k) == "all" else getattr(w, k)
                      for k in ["cost", "resarea", "sectors", "opts"]}))

rule make_summary:
    input: input_make_summary
    output: directory("results/version-" + str(config['version']) + "/summaries/{model_file}_{regions}_{resarea}_{sectors}_l{ll}_{opts}")
    script: "scripts/make_summary.py"

# extract_summaries and plot_costs needs to be updated before it can be used again
#
# rule extract_summaries:
#     input:
#         expand("results/version-{version}/networks/{model_file}_{resarea}_{sectors}_{opts}.nc",
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
#         expand('results/plots/costs_{model_file}_{resarea}_{sectors}_{opt}',
#                **dict(chain(config['scenario'].items(), (('{param}')))
#         touch('results/plots/scenario_plots')
#     params:
#         tmpl="results/plots/costs_[cost]_[resarea]_[sectors]_[opt]"
#         exts=["pdf", "png"]
#     scripts: "scripts/plot_costs.py"


# Local Variables:
# mode: python
# End:
