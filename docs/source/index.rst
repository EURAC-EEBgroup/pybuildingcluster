Welcome to pyBuildingCluster's documentation!
==============================================

.. image:: https://img.shields.io/pypi/v/pybuildingcluster.svg
   :target: https://pypi.python.org/pypi/pybuildingcluster
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/EURAC-EEBgroup/pybuildingcluster.svg
   :target: https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/pybuildingcluster/badge/?version=latest
   :target: https://pybuildingcluster.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/stars/EURAC-EEBgroup/pybuildingcluster.svg?style=social
   :target: https://github.com/EURAC-EEBgroup/pybuildingcluster
   :alt: GitHub stars

**pyBuildingCluster** is a comprehensive Python library for analyzing building energy performance data through clustering, regression modeling, and sensitivity analysis. Designed for building energy researchers, engineers, and practitioners working with large-scale building datasets.

.. image:: _static/images/pybuildingcluster_workflow.png
   :alt: pyBuildingCluster Workflow
   :align: center
   :width: 80%

🏢 **What makes pyBuildingCluster special?**

* **Domain-specific**: Built specifically for building energy analysis
* **Complete workflow**: From data loading to professional reports
* **Research-grade**: Developed by EURAC Research energy experts
* **Easy to use**: High-level interface with sensible defaults
* **Extensible**: Modular architecture for customization

Key Features
------------

🔬 **Advanced Clustering**
   Cluster buildings by energy performance with automatic optimal cluster determination using silhouette, elbow, and Calinski-Harabasz methods.

🤖 **Machine Learning Models**
   Train Random Forest, XGBoost, and LightGBM models with automatic hyperparameter tuning and cross-validation.

📊 **Sensitivity Analysis**
   Conduct comprehensive one-at-a-time and scenario-based sensitivity analysis with interactive visualizations.

🎯 **Parameter Optimization**
   Use Optuna-based optimization to find optimal parameter combinations for energy efficiency.

📄 **Professional Reporting**
   Generate comprehensive HTML reports with insights, visualizations, and actionable recommendations.

Quick Example
-------------

.. code-block:: python

   from pybuildingcluster import GeoClusteringAnalyzer
   
   # Initialize analyzer
   analyzer = GeoClusteringAnalyzer(
       data_path="building_energy_data.csv",
       feature_columns_clustering=['QHnd', 'degree_days'],
       target_column='QHnd'
   )
   
   # Run complete analysis
   results = analyzer.run_complete_analysis(
       clustering_method="silhouette",
       models_to_train=['random_forest', 'xgboost'],
       sensitivity_vars=[
           'average_opaque_surface_transmittance',
           'average_glazed_surface_transmittance'
       ]
   )
   
   # Get summary
   summary = analyzer.get_summary()
   print(summary)

Installation
------------

Install pyBuildingCluster with pip:

.. code-block:: bash

   pip install pybuildingcluster

Or for the latest development version:

.. code-block:: bash

   git clone https://github.com/EURAC-EEBgroup/pybuildingcluster.git
   cd pybuildingcluster
   pip install -e .

Why pyBuildingCluster?
----------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎯 Purpose-Built
      :text-align: center

      Specifically designed for building energy analysis with domain expertise built-in.

   .. grid-item-card:: 🏗️ Complete Workflow
      :text-align: center

      End-to-end analysis from data loading to professional reporting.

   .. grid-item-card:: 🔬 Research-Grade
      :text-align: center

      Developed by EURAC Research with validation on real building datasets.

   .. grid-item-card:: 🚀 Easy to Use
      :text-align: center

      High-level interface with sensible defaults and comprehensive documentation.

Use Cases
----------

**Building Stock Analysis**
   Analyze national or regional building stocks to identify energy performance patterns and renovation opportunities.

**Energy Efficiency Research**
   Study the impact of different building parameters on energy consumption for academic research.

**Policy Analysis**
   Evaluate the effectiveness of building energy policies and regulations.

**ESCO Portfolio Management**
   Identify and prioritize buildings for energy service company interventions.

**Utility Planning**
   Forecast energy demand and plan infrastructure investments.

Research Background
-------------------

pyBuildingCluster was developed by the **Energy Efficient Buildings group** at **EURAC Research** as part of the **MODERATE project** (Horizon Europe grant agreement No 101069834).

The library implements state-of-the-art methods for:

* Building energy performance clustering
* Machine learning for energy prediction
* Sensitivity analysis for parameter optimization
* Scenario analysis for policy evaluation

.. seealso::

   **Academic Publications**
   
   If you use pyBuildingCluster in your research, please cite our work. See :doc:`research/publications` for citation information.

Navigation
----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   examples/index
   cli/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Research & Applications

   research/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index

Community & Support
-------------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 💬 Community
      :text-align: center
      :link: https://github.com/EURAC-EEBgroup/pybuildingcluster/discussions

      Join our GitHub Discussions for questions and community support.

   .. grid-item-card:: 🐛 Bug Reports
      :text-align: center
      :link: https://github.com/EURAC-EEBgroup/pybuildingcluster/issues

      Report bugs and request features on GitHub Issues.

   .. grid-item-card:: 📧 Professional Support
      :text-align: center
      :link: mailto:contact@eurac.edu

      Contact EURAC Research for professional support and consulting.

Acknowledgments
---------------

This work was carried out within the European project **MODERATE** - Horizon Europe research and innovation programme under grant agreement No 101069834, with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment.

License
-------

pyBuildingCluster is released under the BSD 3-Clause License. See the `LICENSE <https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`