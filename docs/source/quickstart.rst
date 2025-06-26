==========
Quickstart
==========

This guide gets you up and running with pyBuildingCluster in 5 minutes.

Installation
============

First, install pyBuildingCluster:

.. code-block:: bash

    pip install pybuildingcluster

Your First Analysis
===================

Let's analyze a sample building energy dataset. We'll create some synthetic data to demonstrate the workflow:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from pybuildingcluster import GeoClusteringAnalyzer
    
    # Create sample building energy data
    np.random.seed(42)
    n_buildings = 1000
    
    # Simulate realistic building energy data
    data = pd.DataFrame({
        # Energy performance
        'QHnd': np.random.lognormal(4.2, 0.5, n_buildings),  # Heating demand (kWh/m²)
        'degree_days': np.random.normal(2500, 500, n_buildings),  # Climate (°C·day)
        
        # Building characteristics  
        'net_area': np.random.lognormal(4.8, 0.4, n_buildings),  # Floor area (m²)
        'construction_year': np.random.randint(1960, 2021, n_buildings),
        'floors': np.random.randint(1, 8, n_buildings),
        
        # Building envelope
        'average_opaque_surface_transmittance': np.random.lognormal(-0.8, 0.5, n_buildings),
        'average_glazed_surface_transmittance': np.random.lognormal(0.8, 0.4, n_buildings),
        
        # System characteristics
        'system_type': np.random.choice([1, 2, 3, 4], n_buildings),
        'ventilation_type': np.random.choice([1, 2, 3], n_buildings)
    })
    
    # Ensure realistic ranges
    data['QHnd'] = np.clip(data['QHnd'], 20, 300)
    data['degree_days'] = np.clip(data['degree_days'], 1000, 4000)
    data['average_opaque_surface_transmittance'] = np.clip(data['average_opaque_surface_transmittance'], 0.1, 2.5)
    data['average_glazed_surface_transmittance'] = np.clip(data['average_glazed_surface_transmittance'], 0.8, 5.0)
    
    # Save sample data
    data.to_csv('sample_building_data.csv', index=False)
    print(f"Created sample dataset with {len(data)} buildings")
    print(data.head())

Now let's run a complete analysis:

.. code-block:: python

    # Initialize the analyzer
    analyzer = GeoClusteringAnalyzer(
        data_path="sample_building_data.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd',
        output_dir='./results'
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        clustering_method="silhouette",
        models_to_train=['random_forest'],
        sensitivity_vars=[
            'average_opaque_surface_transmittance',
            'average_glazed_surface_transmittance'
        ]
    )
    
    # Print summary
    summary = analyzer.get_summary()
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")

What Just Happened?
===================

The analysis performed these steps:

1. **Data Loading**: Loaded and cleaned the building energy dataset
2. **Clustering**: Found optimal number of building clusters using silhouette analysis
3. **Modeling**: Trained Random Forest models for each cluster
4. **Sensitivity Analysis**: Analyzed how U-values affect energy consumption
5. **Reporting**: Generated visualizations and HTML report

Expected Output
===============

You should see output like this:

.. code-block:: text

    📊 Caricamento dati da: sample_building_data.csv
    ✅ Dataset caricato: (1000, 9) righe, colonne
    🧹 Dataset pulito: (1000, 9) righe, colonne
    📊 Feature per clustering: ['QHnd', 'degree_days']
    📈 Feature per regressione: 7 colonne
    
    🔍 Determinazione numero ottimale di cluster (silhouette)...
    🎯 Numero ottimale cluster: 4
    ⚙️ Esecuzione clustering...
    ✅ Clustering completato: 4 cluster
    
    🤖 Addestramento modelli: ['random_forest']
    ✅ Modelli addestrati per 4 cluster
    
    🔬 Analisi sensibilità in corso...
    📈 Analisi sensibilità parametrica...
    🎭 Analisi scenari...
    📊 Generazione grafici...
    📄 Creazione report HTML...
    ✅ Analisi sensibilità completata!

Understanding the Results
=========================

Your analysis produces several outputs:

Results Directory Structure
---------------------------

.. code-block:: text

    results/
    ├── scenario_analysis_report_QHnd.html  # Main HTML report
    ├── cluster_data/                       # Individual cluster datasets
    │   ├── cluster_0.csv
    │   ├── cluster_1.csv
    │   └── ...
    └── visualizations/                     # Generated plots
        ├── clustering_visualization.png
        ├── sensitivity_plots.png
        └── scenario_comparison.png

Key Results Interpretation
--------------------------

**Clusters Found**: The algorithm identified distinct groups of buildings with similar energy performance patterns.

**Model Performance**: Each cluster has a trained model predicting energy demand based on building characteristics.

**Sensitivity Analysis**: Shows how changing U-values (insulation quality) affects energy consumption.

**Scenarios**: Compares different building efficiency scenarios (e.g., "High Efficiency" vs "Standard").

Real-World Data
===============

For real building analysis, your CSV should have columns like:

.. code-block:: python

    # Required columns for building energy analysis
    required_columns = [
        'QHnd',  # Heating energy demand (kWh/m²/year) - TARGET
        'degree_days',  # Heating degree days (°C·day) - CLIMATE
        
        # Building characteristics (features)
        'net_area',  # Floor area (m²)
        'construction_year',  # Year of construction
        'average_opaque_surface_transmittance',  # Wall U-value (W/m²K)
        'average_glazed_surface_transmittance',  # Window U-value (W/m²K)
        'floors',  # Number of floors
        'system_type',  # Heating system type (categorical)
    ]

Energy Performance Certificate Data
-----------------------------------

pyBuildingCluster works excellently with Energy Performance Certificate (EPC) data:

.. code-block:: python

    # Example with real EPC data
    analyzer = GeoClusteringAnalyzer(
        data_path="energy_certificates.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd',
        output_dir='./epc_analysis'
    )
    
    # Define building renovation scenarios
    retrofit_scenarios = [
        {
            'name': 'Current State',
            'parameters': {
                'average_opaque_surface_transmittance': 0.75,
                'average_glazed_surface_transmittance': 3.0
            }
        },
        {
            'name': 'Standard Renovation',
            'parameters': {
                'average_opaque_surface_transmittance': 0.35,
                'average_glazed_surface_transmittance': 1.8
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
    
    # Run analysis with custom scenarios
    results = analyzer.run_complete_analysis(
        scenarios=retrofit_scenarios,
        clustering_method="silhouette",
        models_to_train=['random_forest', 'xgboost']
    )

Step-by-Step Workflow
=====================

For more control, you can run each step individually:

.. code-block:: python

    # 1. Initialize analyzer
    analyzer = GeoClusteringAnalyzer(
        data_path="building_data.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd'
    )
    
    # 2. Load and clean data
    data = analyzer.load_and_clean_data(
        columns_to_remove=['building_id', 'address']
    )
    
    # 3. Perform clustering
    clusters = analyzer.perform_clustering(
        method="silhouette",
        k_range=(2, 8)
    )
    
    # 4. Build predictive models
    models = analyzer.build_models(
        models_to_train=['random_forest', 'xgboost'],
        hyperparameter_tuning="optuna"
    )
    
    # 5. Create energy efficiency scenarios
    scenarios = analyzer.create_scenarios_from_cluster(
        cluster_id=1,
        sensitivity_vars=[
            'average_opaque_surface_transmittance',
            'average_glazed_surface_transmittance'
        ],
        n_scenarios=8
    )
    
    # 6. Run sensitivity analysis
    sensitivity_results = analyzer.run_sensitivity_analysis(
        cluster_id=1,
        scenarios=scenarios,
        n_points=25
    )

Common Use Cases
================

Building Stock Analysis
-----------------------

Analyze a national or regional building stock:

.. code-block:: python

    # Large-scale building stock analysis
    stock_analyzer = GeoClusteringAnalyzer(
        data_path="national_building_stock.csv",
        feature_columns_clustering=['QHnd', 'degree_days', 'construction_year'],
        target_column='QHnd'
    )
    
    # Focus on older buildings with high energy consumption
    results = stock_analyzer.run_complete_analysis(
        clustering_method="silhouette",
        models_to_train=['random_forest', 'lightgbm'],
        columns_to_remove=['owner_name', 'address']
    )

Energy Service Company (ESCO) Analysis
---------------------------------------

Identify buildings with highest retrofit potential:

.. code-block:: python

    # ESCO portfolio analysis
    esco_analyzer = GeoClusteringAnalyzer(
        data_path="esco_portfolio.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd'
    )
    
    # Define retrofit investment scenarios
    investment_scenarios = [
        {'name': 'Low Investment', 'parameters': {'average_opaque_surface_transmittance': 0.5}},
        {'name': 'Medium Investment', 'parameters': {'average_opaque_surface_transmittance': 0.3}},
        {'name': 'High Investment', 'parameters': {'average_opaque_surface_transmittance': 0.15}}
    ]
    
    results = esco_analyzer.run_complete_analysis(scenarios=investment_scenarios)

Policy Impact Analysis
----------------------

Evaluate building energy policy effectiveness:

.. code-block:: python

    # Policy analysis
    policy_analyzer = GeoClusteringAnalyzer(
        data_path="buildings_policy_study.csv",
        feature_columns_clustering=['QHnd', 'degree_days'],
        target_column='QHnd'
    )
    
    # Compare pre- and post-policy scenarios
    policy_scenarios = [
        {
            'name': 'Pre-Policy (2015)',
            'parameters': {
                'average_opaque_surface_transmittance': 0.8,
                'construction_year': 2015
            }
        },
        {
            'name': 'Post-Policy (2020)',
            'parameters': {
                'average_opaque_surface_transmittance': 0.4,
                'construction_year': 2020
            }
        }
    ]
    
    results = policy_analyzer.run_complete_analysis(scenarios=policy_scenarios)

Command Line Usage
==================

pyBuildingCluster also provides a command-line interface:

.. code-block:: bash

    # Basic analysis
    pybuildingcluster analyze \
        --data building_data.csv \
        --clustering-features QHnd degree_days \
        --target QHnd \
        --output-dir ./results
    
    # Advanced analysis with custom parameters
    pybuildingcluster analyze \
        --data energy_certificates.csv \
        --clustering-features QHnd degree_days construction_year \
        --regression-features net_area floors avg_opaque_transmittance \
        --target QHnd \
        --clustering-method silhouette \
        --models random_forest xgboost \
        --sensitivity-vars avg_opaque_transmittance avg_glazed_transmittance \
        --output-dir ./comprehensive_analysis

Visualization Examples
======================

The analysis generates several types of visualizations:

**Clustering Visualization**
   Scatter plot showing how buildings cluster in the energy-climate space.

**Model Performance**
   Charts showing prediction accuracy for each cluster.

**Sensitivity Analysis**
   Plots showing how parameters affect energy consumption.

**Scenario Comparison**
   Bar charts comparing different efficiency scenarios.

**Parameter Heatmaps**
   Heatmaps showing parameter values across scenarios.

Interpreting Results
====================

HTML Report
-----------

The main output is a comprehensive HTML report containing:

* **Executive Summary**: Key findings and recommendations
* **Cluster Analysis**: Description of building clusters found
* **Model Performance**: Accuracy metrics for predictive models
* **Sensitivity Results**: Parameter impact analysis
* **Scenario Comparison**: Energy savings potential
* **Recommendations**: Actionable insights for energy efficiency

Key Metrics
-----------

**Silhouette Score**: Measures cluster quality (higher is better, >0.5 is good)

**R² Score**: Model prediction accuracy (higher is better, >0.7 is good)

**Energy Savings**: Percentage reduction in energy demand between scenarios

**Parameter Sensitivity**: How much energy changes when parameters change

Next Steps
==========

Now that you've completed your first analysis:

1. **Explore the HTML report** generated in your results directory
2. **Try different clustering methods** (elbow, calinski_harabasz)
3. **Experiment with more models** (xgboost, lightgbm)
4. **Add more sensitivity variables** relevant to your analysis
5. **Create custom scenarios** specific to your use case

Continue Learning
=================

* :doc:`tutorials/index` - Detailed tutorials for specific workflows
* :doc:`examples/index` - Real-world examples with actual datasets  
* :doc:`user_guide/index` - In-depth user guide for advanced features
* :doc:`api/index` - Complete API reference

Common Issues
=============

**"No module named 'pybuildingcluster'"**
   Make sure you've installed the package: ``pip install pybuildingcluster``

**"Not enough samples in cluster"**
   Try reducing the number of clusters or increasing your dataset size.

**"Model performance is low"**
   Consider adding more relevant features or checking data quality.

**Memory errors with large datasets**
   Use data sampling or run analysis in chunks for datasets >50k buildings.

Need Help?
==========

* **Documentation**: This documentation covers most use cases
* **GitHub Issues**: Report bugs or request features
* **GitHub Discussions**: Ask questions and get community help
* **Professional Support**: Contact EURAC Research for consulting

.. tip::

   **Start Small**: Begin with a subset of your data (~1000 buildings) to test the workflow, then scale up to your full dataset.