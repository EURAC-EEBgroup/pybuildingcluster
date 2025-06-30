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


**pyBuildingCluster** is a comprehensive Python library for analyzing building energy performance data through clustering, regression modeling, sensitivity analysis ans scenario-based analysis. 
Designed for building energy researchers, engineers, and practitioners working with large-scale building datasets.

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
    def feature_columns_regression(building_data):
        """Define feature columns for clustering and modeling."""
        feature_remove_regression = ["QHnd","EPl", "EPt", "EPc", "EPv", "EPw", "EPh", "QHimp", "theoric_nominal_power", "energy_vectors_used"]
        feature_columns_df = building_data.columns
        feature_columns_regression = [item for item in feature_columns_df if item not in feature_remove_regression]
        return feature_columns_regression
    
    feature_columns_regression = feature_columns_regression(building_data)
    
    analyzer = GeoClusteringAnalyzer(
        data_path=building_data,                   
        feature_columns_clustering=['QHnd', 'degree_days'],
        feature_columns_regression=feature_columns_regression,
        output_dir=os.path.join(temp_dir, 'results'),
        target_column='QHnd',
        random_state=42,
        user_features=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
    )

    # Clean data
    loaded_data = analyzer.load_and_clean_data(columns_to_remove=["EPl", "EPt", "EPc", "EPv", "EPw", "EPh", "QHimp", "theoric_nominal_power"])

    # Build models
    regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
    models = regression_builder.build_models(
        data=building_data,
        clusters=clusters,
        target_column='QHnd',
        feature_columns=feature_columns_regression,
        models_to_train=['random_forest'],
        hyperparameter_tuning="none",
        models_dir=os.path.join(temp_dir, 'models'),
        save_models=True,
        user_features = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
    )
    
    sensitivity_analyzer = SensitivityAnalyzer(random_state=42)
    sensitivity_results = sensitivity_analyzer.analyze(
        model=models[1]['best_model'],
        data=building_data,
        scenarios=sensitivity_parameters,
        feature_columns=feature_columns_regression,
        target_column='QHnd',
        sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
        n_points=20,
        normalize_=True,
        plot_3d=False,
        cluster_id=None,
        save_results=True,
        results_dir=os.path.join(temp_dir, 'sensitivity'),
        create_html_report=True
    )

    optimizer = ParameterOptimizer(random_state=42)
        
    # Pick one cluster for optimization
    cluster_id = list(models.keys())[0]
    cluster_data = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == cluster_id]
    
    optimization_parameter_space = {
        'average_opaque_surface_transmittance': {'type': 'float', 'low': 0.1, 'high': 1.0},
        'average_glazed_surface_transmittance': {'type': 'float', 'low': 0.7, 'high': 3.0}
    }
    
    optimization_results = optimizer.optimize_cluster_parameters(
        cluster_data=cluster_data,
        models=models,
        parameter_space=optimization_parameter_space,
        target_column='QHnd',
        n_trials=5,
        optimization_direction="minimize"
    )
    
    # Create scenarios
    list_dict_scenarios = analyzer.create_scenarios_from_cluster(cluster_id=0, sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'], n_scenarios=10)
    # or   
    list_dict_scenarios = [
        {'name': 'Scenario 1', 'parameters': {'average_opaque_surface_transmittance': 0.5, 
                                            'average_glazed_surface_transmittance': 1}},
        {'name': 'Scenario 2', 'parameters': {'average_opaque_surface_transmittance': 0.2, 
                                            'average_glazed_surface_transmittance': 0.7}}
    ]

    scenario_results = sensitivity_analyzer.compare_scenarios(
        cluster_df=data_with_clusters,
        scenarios=list_dict_scenarios,
        target='QHnd',
        feature_columns=models[1]['feature_columns'],
        modello=models[1]['best_model']
    )


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

* under construction

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
    * Research collaborations: daniele.antonucci@eurac.edu
    * Commercial support and consulting available
    * Training workshops and seminars
    * Custom development for specific applications

**📧 Contact**
    * Email: daniele.antonucci@eurac.edu
    * Website: https://www.eurac.edu/en/institutes-centers/institute-for-renewable-energy
    * MODERATE Project: https://moderate-project.eu/

============
License
============

pyBuildingCluster is released under the **MIT License**.

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