##########################################
Installation
##########################################

To build and solve the model, a computer with about 20GB of memory with a strong interior-point solver supported by the modelling library '_PYOMO <https://github.com/Pyomo/pyomo>'_ like Gurobi or CPLEX are required.

We recommend as preparatory steps (the path before the ''%'' sign denotes the directory in which the commands following the ''%'' should be entered):

cloning the repository using ''git''' (to a directory without any spaces in the path)

..code:: bash
/some/other/path % cd /some/path/without/spaces
/some/path/without/spaces % git clone https://github.com/FRESNA/pypsa-za.git

installing the necessary python dependencies using conda (from within the pypsa-za directory)

..code:: bash 
.../pypsa-za % conda env create -f environment.yaml
.../pypsa-za % source activate pypsa-za  # or conda activate pypsa-za on windows
getting the separate data bundle (see also [Data dependencies] below) and unpacking it in data

..code:: bash
.../data % wget "https://vfs.fias.science/d/f204668ef2/files/?dl=1&p=/pypsa-za-bundle.7z"
.../data % 7z x pypsa-za-bundle.7z

All results and scenario comparisons are reproduced using the workflow management system snakemake

..code:: bash
.../pypsa-za % snakemake
[... will take about a week on a recent computer with all scenarios ...]
snakemake will first compute several intermediate data files in the directory resources, then prepare unsolved networks in networks, solve them and save the resulting networks in results/version-0.x/networks and finally render the main plots into results/version-0.5/plots.

Instead of computing all scenarios (defined by the product of all wildcards in the scenario config section), snakemake also allows to compute only a specific scenario like csir-aggressive_redz_E_LC:

..code:: bash
.../pypsa-za % snakemake results/version-0.5/plots/network_csir-aggressive_redz_E_LC_p_nom