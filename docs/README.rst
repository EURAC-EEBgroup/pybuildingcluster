=====================
pyBuildingCluster
=====================

.. image:: https://img.shields.io/pypi/v/pybuildingcluster.svg
    :target: https://pypi.python.org/pypi/pybuildingcluster
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pybuildingcluster.svg
    :target: https://pypi.python.org/pypi/pybuildingcluster
    :alt: Python versions

.. image:: https://readthedocs.org/projects/pybuildingcluster/badge/?version=latest
    :target: https://pybuildingcluster.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


.. image:: https://img.shields.io/github/license/EURAC-EEBgroup/pybuildingcluster.svg
    :target: https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/LICENSE
    :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg
..    :target: https://doi.org/10.5281/zenodo.XXXXXXX
..    :alt: DOI

A comprehensive Python library for analyzing building energy performance data through clustering, regression modeling, and sensitivity analysis. Designed for building energy researchers, engineers, and practitioners working with large-scale building datasets.

🏢 **Building Energy Intelligence Made Simple**

pyBuildingCluster transforms complex building energy datasets into actionable insights through advanced machine learning and sensitivity analysis techniques.

.. contents:: Table of Contents
   :depth: 2

============
Key Features
============

🔬 **Advanced Clustering**
    * Automatic optimal cluster determination using silhouette, elbow, and Calinski-Harabasz methods
    * Support for K-means, DBSCAN, and hierarchical clustering algorithms
    * Statistical validation and cluster quality metrics
    * Domain-specific clustering for building energy patterns

🤖 **Machine Learning Models**
    * Random Forest, XGBoost, and LightGBM regression models
    * Automatic model selection and hyperparameter tuning with Optuna
    * Cross-validation and comprehensive performance evaluation
    * Cluster-specific model training for improved accuracy

🔍 **Sensitivity Analysis**
    * One-at-a-time and scenario-based sensitivity analysis
    * Parameter optimization for energy efficiency
    * Interactive 3D visualizations and comprehensive plotting
    * Professional HTML reports with actionable insights

📊 **Professional Reporting**
    * Comprehensive HTML reports with visualizations
    * Export capabilities (CSV, PDF, Excel)
    * Interactive dashboards and plots
    * Research-grade documentation and citations

🛠️ **Developer-Friendly**
    * High-level API with sensible defaults
    * Command-line interface for automation
    * Extensible modular architecture
    * Comprehensive documentation and examples

============
Quick Start
============

Installation
------------

Install pyBuildingCluster with pip:

.. code-block:: bash

    pip install pybuildingcluster

For additional features:

.. code-block:: bash

    # Include machine learning models
    pip install pybuildingcluster[ml]
    
    # Include interactive visualizations  
    pip install pybuildingcluster[viz]
    
    # Install everything
    pip install pybuildingcluster[all]

Basic Usage
-----------

.. code-block:: python

    from pybuildingcluster import GeoClusteringAnalyzer
    
    # Initialize analyzer
    analyzer = GeoClusteringAnalyzer(
        data_path="building_energy_data.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd',
        output_dir='./results'
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

This will:

1. **Cluster buildings** by energy performance characteristics
2. **Train ML models** for each cluster to predict energy demand
3. **Analyze sensitivity** of energy demand to building parameters
4. **Generate reports** with visualizations and recommendations

============
Use Cases
============

**🏗️ Building Stock Analysis**
    Analyze national or regional building stocks to identify energy performance patterns, renovation priorities, and policy effectiveness.

**🔬 Energy Efficiency Research**
    Conduct academic research on building energy performance, parameter sensitivity, and retrofit potential with publication-ready results.

**💼 Energy Service Companies (ESCOs)**
    Identify buildings with highest energy savings potential, optimize retrofit strategies, and quantify energy performance improvements.

**🏛️ Policy Analysis & Development**
    Evaluate the effectiveness of building energy policies, simulate regulation impacts, and support evidence-based policy making.

**⚡ Utility Planning & Forecasting**
    Forecast energy demand, plan infrastructure investments, and understand customer energy consumption patterns.

**🏘️ Real Estate & Property Assessment**
    Assess property energy performance, estimate energy labels, and quantify the value impact of energy efficiency improvements.

============
Documentation
============

**📖 Comprehensive Documentation**: https://pybuildingcluster.readthedocs.io/

* **Installation Guide** - Detailed setup instructions for all platforms
* **Quick Start Tutorial** - Get running in 5 minutes
* **API Reference** - Complete function and class documentation  
* **User Guide** - In-depth explanations and best practices
* **Examples** - Real-world applications with actual datasets
* **Research Applications** - Academic use cases and citation information

============
Examples
============

Building Energy Certificate Analysis
------------------------------------

.. code-block:: python

    # Analyze European building energy certificates
    analyzer = GeoClusteringAnalyzer(
        data_path="european_epc_data.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd'
    )
    
    # Define renovation scenarios
    scenarios = [
        {
            'name': 'Current State',
            'parameters': {
                'average_opaque_surface_transmittance': 0.75,
                'average_glazed_surface_transmittance': 3.0
            }
        },
        {
            'name': 'Deep Renovation', 
            'parameters': {
                'average_opaque_surface_transmittance': 0.15,
                'average_glazed_surface_transmittance': 1.0
            }
        }
    ]
    
    results = analyzer.run_complete_analysis(scenarios=scenarios)

ESCO Portfolio Optimization
---------------------------

.. code-block:: python

    # Analyze building portfolio for energy service company
    analyzer = GeoClusteringAnalyzer("esco_portfolio.csv")
    
    # Focus on buildings with highest savings potential
    results = analyzer.run_complete_analysis(
        clustering_method="silhouette",
        models_to_train=['random_forest', 'lightgbm'],
        sensitivity_vars=[
            'average_opaque_surface_transmittance',
            'average_glazed_surface_transmittance',
            'system_efficiency'
        ]
    )
    
    # Results include ROI analysis and retrofit recommendations

Command Line Interface
----------------------

.. code-block:: bash

    # Quick analysis from command line
    pybuildingcluster analyze \
        --data building_data.csv \
        --clustering-features QHnd degree_days \
        --target QHnd \
        --models random_forest xgboost \
        --output-dir ./results
    
    # Advanced analysis with custom scenarios
    pybuildingcluster scenarios \
        --data energy_certificates.csv \
        --scenarios retrofit_scenarios.json \
        --output comprehensive_report.html

============
Data Format
============

pyBuildingCluster works with building energy datasets containing:

**Required Columns:**

* ``QHnd`` - Heating energy demand (kWh/m²/year)
* ``degree_days`` - Heating degree days (°C·day)

**Common Building Features:**

* ``net_area`` - Floor area (m²)
* ``construction_year`` - Year of construction
* ``average_opaque_surface_transmittance`` - Wall U-value (W/m²K) 
* ``average_glazed_surface_transmittance`` - Window U-value (W/m²K)
* ``floors`` - Number of floors
* ``system_type`` - Heating system type

**Example Dataset:**

.. code-block:: csv

    QHnd,degree_days,net_area,construction_year,average_opaque_surface_transmittance
    85.3,2856,120.5,1985,0.65
    42.1,2856,95.2,2010,0.25
    120.7,3124,200.8,1975,0.85

See the `Data Preparation Guide <https://pybuildingcluster.readthedocs.io/en/latest/user_guide/data_preparation.html>`_ for detailed requirements.

============
Research Background
============

pyBuildingCluster was developed by the **Energy Efficient Buildings group** at `EURAC Research <https://www.eurac.edu/en/institutes-centers/institute-for-renewable-energy>`_ as part of the **MODERATE project** (Horizon Europe grant agreement No 101069834).

The library implements state-of-the-art methods for:

* **Building energy performance clustering** based on physics-informed features
* **Machine learning for energy prediction** with domain-specific validation
* **Sensitivity analysis** for building parameter optimization
* **Scenario analysis** for policy and retrofit evaluation

**Academic Applications:**

* Building stock characterization and segmentation
* Energy efficiency potential assessment  
* Policy impact analysis and evaluation
* Climate change adaptation studies
* Retrofit optimization and prioritization

============
Installation
============

System Requirements
-------------------

* **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
* **Operating System**: Linux, macOS, or Windows
* **RAM**: Minimum 4 GB, recommended 8+ GB for large datasets
* **Storage**: 1 GB free space

Basic Installation
------------------

.. code-block:: bash

    pip install pybuildingcluster

Development Installation
------------------------

.. code-block:: bash

    git clone https://github.com/EURAC-EEBgroup/pybuildingcluster.git
    cd pybuildingcluster
    pip install -e ".[dev,docs]"

Optional Dependencies
--------------------

.. code-block:: bash

    # Machine learning models (XGBoost, LightGBM, Optuna)
    pip install pybuildingcluster[ml]
    
    # Interactive visualizations (Plotly, Bokeh) 
    pip install pybuildingcluster[viz]
    
    # Development tools (testing, linting, formatting)
    pip install pybuildingcluster[dev]
    
    # Documentation building (Sphinx, themes)
    pip install pybuildingcluster[docs]
    
    # Jupyter integration (notebooks, widgets)
    pip install pybuildingcluster[interactive]
    
    # Everything included
    pip install pybuildingcluster[all]

============
Contributing
============

We welcome contributions from the building energy community! 

**Ways to Contribute:**

* 🐛 **Report bugs** and request features via GitHub Issues
* 📖 **Improve documentation** with examples and tutorials  
* 🔧 **Submit code** for new features or bug fixes
* 🧪 **Add test cases** and improve code coverage
* 💡 **Share use cases** and real-world applications
* 🎓 **Academic collaborations** and research partnerships

**Getting Started:**

1. Fork the repository on GitHub
2. Clone your fork: ``git clone https://github.com/yourusername/pybuildingcluster.git``
3. Install development dependencies: ``pip install -e ".[dev]"``
4. Create a feature branch: ``git checkout -b feature-name``
5. Make changes and add tests
6. Run tests: ``pytest tests/``
7. Submit a pull request

See `CONTRIBUTING.rst <https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/CONTRIBUTING.rst>`_ for detailed guidelines.

============
Citation
============

If you use pyBuildingCluster in your research, please cite:

.. code-block:: bibtex

    @software{pybuildingcluster2024,
      title={pyBuildingCluster: A Python Library for Building Energy Clustering and Sensitivity Analysis},
      author={EURAC Research - Energy Efficient Buildings Group},
      year={2024},
      url={https://github.com/EURAC-EEBgroup/pybuildingcluster},
      doi={10.5281/zenodo.XXXXXXX},
      note={Developed under the MODERATE project (Horizon Europe grant agreement No 101069834)}
    }

**Related Publications:**

* [Add your publications here when available]
* [Submit your work using pyBuildingCluster to be featured]

============
Support
============

**📚 Documentation & Tutorials**
    * Complete documentation: https://pybuildingcluster.readthedocs.io/
    * API reference with examples
    * Step-by-step tutorials for all skill levels

**💬 Community Support**
    * GitHub Discussions: https://github.com/EURAC-EEBgroup/pybuildingcluster/discussions
    * GitHub Issues: https://github.com/EURAC-EEBgroup/pybuildingcluster/issues
    * Stack Overflow: Tag questions with ``pybuildingcluster``

**🔬 Professional Support**
    * Research collaborations: contact@eurac.edu
    * Commercial support and consulting available
    * Training workshops and seminars
    * Custom development for specific applications

**📧 Contact**
    * Email: contact@eurac.edu
    * Website: https://www.eurac.edu/en/institutes-centers/institute-for-renewable-energy
    * MODERATE Project: https://moderate-project.eu/

============
License
============

pyBuildingCluster is released under the **BSD 3-Clause License**.

.. code-block:: text

    Copyright (c) 2024, EURAC Research - Energy Efficient Buildings Group
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the conditions in the 
    LICENSE file are met.

See the `LICENSE <https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/LICENSE>`_ file for the complete license text.

============
Acknowledgments
============

This work was carried out within the European project:

**MODERATE** - *Horizon Europe research and innovation programme under grant agreement No 101069834*, with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment.

**Research Team:**

* **EURAC Research** - Institute for Renewable Energy
* **Energy Efficient Buildings Group** - Research and development
* **Contributors** - See `AUTHORS.rst <https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/AUTHORS.rst>`_