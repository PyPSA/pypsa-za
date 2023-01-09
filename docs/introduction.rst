..
  SPDX-FileCopyrightText: 2021 The PyPSA-ZA authors

  SPDX-License-Identifier: CC-BY-4.0

.. _introduction:

##########################################
Introduction
##########################################


Workflow
========

The generation of the model is controlled by the workflow management system `Snakemake <https://snakemake.bitbucket.io/>`_. In a nutshell,
the ``Snakefile`` declares for each python script in the ``scripts`` directory a rule which describes which files the scripts consume and
produce (their corresponding input and output files). The ``snakemake`` tool then runs the scripts in the correct order according to the
rules' input/output dependencies. Moreover, it is able to track, what parts of the workflow have to be regenerated, when a data file or a
script is modified/updated. For example, by executing the following snakemake routine

.. code:: bash

    .../pypsa-za % snakemake -j results/networks/solved_CSIR-ambitions_9-supply_redz_lcopt_LC-1H.nc

the following workflow is automatically executed.

.. image:: img/workflow.png
    :align: center

The **blocks** represent the individual rules which are required to create the file ``results/networks/solved_CSIR-ambitions_9-supply_redz_lcopt_LC-1H.nc``.
Each rule requires scripts (e.g. Python) to convert inputs to outputs.
The **arrows** indicate the outputs from preceding rules which a particular rule takes as input data.

.. note::
    For reproducibility purposes, the image can be obtained through
    ``snakemake --dag results/networks/solved_CSIR-ambitions_9-supply_redz_lcopt_LC-1H.nc | dot -Tpng -o workflow.png``
    using `Graphviz <https://graphviz.org/>`_


Folder structure
================

The content in this package is organized in folders as described below; for more details, please see the documentation.

- ``data``: Includes input data that is not produced by any ``snakemake`` rule.
- ``scripts``: Includes all the Python scripts executed by the ``snakemake`` rules.
- ``resources``: Stores intermediate results of the workflow which can be picked up again by subsequent rules.
- ``networks``: Stores intermediate, unsolved stages of the PyPSA network that describes the energy system model.
- ``results``: Stores the solved PyPSA network data, summary files and plots.
- ``benchmarks``: Stores ``snakemake`` benchmarks.
- ``logs``: Stores log files about solving, including the solver output, console output and the output of a memory logger.
- ``envs``: Stores the conda environment files to successfully run the workflow.


License
=======

PyPSA-ZA work is released under multiple licenses:

* All original source code is licensed as free software under `GPL-3.0 License <https://github.com/pypsa-meets-earth/pypsa-earth/blob/main/LICENSE>`_.
* The documentation is licensed under `CC-BY-4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
* Configuration files are mostly licensed under `CC0-1.0 <https://creativecommons.org/publicdomain/zero/1.0/>`_.
* Data files are licensed under different licenses as noted below.

Licenses and urls of the data used in PyPSA-ZA:

.. csv-table::
   :header-rows: 1
   :file: configtables/licenses.csv


* *BY: Attribute Source*
* *NC: Non-Commercial Use Only*
* *SA: Share Alike*
