..
  SPDX-FileCopyrightText: 2021 The PyPSA-ZA authors

  SPDX-License-Identifier: CC-BY-4.0

.. _introduction:

##########################################
Introduction
##########################################

History
========

Historically, commercial tools have been extensively utilised by both governments and utilities to develop capacity expansion plans. 
One of the challenges of utilising such commercial tools to support transparent energy policy is the closed nature of the models, 
which are essentially a ‘black-box’. For this reason, initiatives such as openmod1 have developed with the goal of creating open 
energy modelling platforms that are based on open-source software2 as well as open data. 

An initial PyPSA model of the South African power system called PyPSA-ZA was developed by Hörsch and Calitz1 who studied a future 
renewable energy based South African electricity network for a single reference year (in other words, it was not an intertemporally 
optimised power system development model – showing an optimised transition over time – it only provided a snapshot of an optimised 
high renewables system at a particular point in time – 2040, albeit with reasonable spatial (network) resolution.
PyPSA has also been utilised to build an extensive interconnected power system model of the European power system (see PyPSA-Eur2). 
This model has been further extended to include full sector coupling of power, transportation and heat demand (see PyPSA-Eur-Sec4). 
Like the PyPSA-ZA model, the PyPSA-Eur model framework has been designed to handle high spatial resolution (detailed network 
specification, i.e. grid constraints), but it does not perform multi-year optimisation and instead analyses a single year snapshot. 
In the African context the project PyPSA meets Africa is currently ongoing and aims to develop an interconnected, high-level, 
African power system model on the same basis as PyPSA-Eur3
Multi-horizon planning functionality, based on the assumption of “perfect foresight”, was only recently added to PyPSA in 2021, 
and currently PyPSA-Eur, PyPSA-ZA, and PyPSA meets Africa do not yet incorporate this capability. The PyPSA-Eur-Sec model does include myopic expansion planning capability that is based on separating the planning horizon into several shorter periods that are solved as separate optimisations.


The PyPSA-ZA model was first developed by ...
Explain model - single year 2040 higher spatial resolution 27-supply areas, image etc
Link to paper 
Branch of PyPSA-ZA original model

In 2022 Meridian Economics `<https://meridianeconomics.co.za/>`_ 


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
