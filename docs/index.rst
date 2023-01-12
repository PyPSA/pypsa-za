.. pypsa-za documentation master file, created by
   sphinx-quickstart on Fri Jan  6 10:47:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyPSA-ZA: An open Optimisation Model of the South African Power System
=======================================================================
PyPSA-ZA is an open model dataset of the South African power system at four levels

.. subfigure:: AB|CD
   :layout-sm: A|B|C|D
   :gap: 8px
   :subcaptions: above
   :name: myfigure
   :class-grid: outline

   .. image:: img/validation-4_RSA_redz_lcopt_LC_p_nom_ext.png
      :alt: 27-supply

   .. image:: img/validation-4_RSA_redz_lcopt_LC_p_nom_ext.png
      :alt: 10-supply

   .. image:: img/validation-4_RSA_redz_lcopt_LC_p_nom_ext.png
      :alt: 9-supply

   .. image:: img/validation-4_RSA_redz_lcopt_LC_p_nom_ext.png
      :alt: RSA

    Figure Caption

The restriction to freely available and open data encourages the open exchange of model data developments and eases the comparison of model results. 
It provides a full, automated software pipeline to assemble the load-flow-ready model from the original datasets, which enables easy replacement and 
improvement of the individual parts.

PyPSA-ZA is designed to be imported into the open toolbox `PyPSA <https://pypsa.org/>`_ for which `documentation <https://pypsa.readthedocs.io/en/latest/index.html>`_ is available as well.

This project is currently maintained by `Meridian Economics <https://meridianeconomics.co.za/>`_. Previous versions were developed within the Energy Centre 
at the Council for Scientific and Industrial Research as part of the `CoNDyNet project <https://fias.institute/en/projects/condynet/>`_, which is supported by the 
`German Federal Ministry of Education and Research <https://www.bmbf.de/bmbf/en/home/home_node.html>`_ under grant no. 03SF0472C.


=============
Documentation
=============

**Getting Started**

* :doc:`introduction`
* :doc:`installation`
* :doc:`tutorials`
* :doc:`data_workflow`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   introduction
   installation  
   tutorials
   data_workflow    

**Configuration**

* :doc:`wildcards`
* :doc:`configuration`
* :doc:`costs`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Configuration

   wildcards
   configuration
   costs

**Work flow and API**

* :doc:`structure`
* :doc:`rules_overview`
* :doc:`api_reference`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Work flow and API

   structure
   rules_overview
   api_reference

**Help and References**

* :doc:`release_notes`
* :doc:`how_to_contribute`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Project Info

   release_notes
   how_to_contribute
