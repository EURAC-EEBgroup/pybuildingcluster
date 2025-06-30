# Geoclustering Sensitivity Analysis

A Python library for sensitivity analysis of building clusters, evaluating refurbishment scenarios.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This library provides comprehensive tools for analyzing building energy performance data through clustering, regression modeling, and sensitivity analysis. It was developed by the Energy Efficient Buildings group at EURAC Research as part of the MODERATE project (Horizon Europe grant agreement No 101069834).

### Key Features

- **Clustering Analysis**: K-means clustering with automatic cluster number determination using elbow method or silhouette analysis, also DBSCAN and hierarchical clustering are supported.
- **Regression Modeling**: Support for Random Forest, XGBoost, and LightGBM models with automatic model selection
- **Sensitivity Analysis**: Scenario-based analysis to understand parameter impacts on clusters
- **Parameter Optimization**: Optuna-based optimization for finding optimal parameter combinations
- **CLI Interface**: Command-line tools for easy integration into workflows
- **Extensible Design**: Modular architecture for easy customization and extension

## Installation

### From PyPI (recommended)

```bash
pip install pybuildingcluster
```

### From Source

```bash
git clone https://github.com/EURAC-EEBgroup/pybuildingcluster.git
cd pybuildingcluster
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/EURAC-EEBgroup/pybuildingcluster.git
cd pybuildingcluster
pip install -e ".[dev]"
```

## Quick Start

### Example

Here an `Example <https://github.com/EURAC-EEBgroup/pyBuildingCluster/tree/master/examples>` of pybuildingcluster application, using Energy Performance Certifcate dataset

The example use Synthetic dataset of Energy Performance Certificates of public buildings in Piedmont Region, Italy.

The process of synthesis was carried out using the library provided by MOSTLY AI <https://github.com/mostly-ai/mostlyai>
All within the synthetic folder it is possible to view the report relative to the generation of the synthetic dataset.

### How to use the synthetic dataset and the library.
The synthesized dataset, in addition to preserving the same statistical characteristics as the original data, represents a very useful resource for evaluating potential energy efficiency improvements across a building stock where only some buildings' performance data is known. In fact, by generating synthetic data, more robust assessments can be made, since the analysis can be based on a larger number of buildings that closely resemble those present in the actual territory.

Here `Report <https://github.com/EURAC-EEBgroup/pybuildingcluster/tree/master/synthetization/EPC_tabular.html>` of the synthetic dataset of EPC.

Once the data is generated, it can be divided into different clusters based on specific properties.
In the example provided, the clustering is done using the QHnd property (heating energy demand of the building) and  heating degree days.

<p align="center">
  <img src="https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/src/pybuildingcluster/assets/elbow.png" alt=" Elbow Method" style="width: 100%;">
</p>

Each cluster is then analyzed through a sensitivity analysis of selected parameters.

In this case, the average thermal transmittance of opaque components and the average thermal transmittance of transparent components are used.

<p align="center">
  <img src="https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/src/pybuildingcluster/assets/sensitivity.png" alt=" Sensitivity Analysis" style="width: 100%;">
</p>

The analysis shows how varying these parameters can lead to significant reductions in energy consumption for the selected cluster.
For instance, the map illustrates that the dark blue areas correspond to the greatest reductions in consumption, as they represent combinations of low values for both selected parameters. However, this may not always represent the best performance-to-cost ratio. In fact, considerable savings can also be achieved by slightly improving these parameters, which requires a lower investment.

Moreover, specific retrofit scenarios can be identified. 
<p align="center">
  <img src="https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/main/src/pybuildingcluster/assets/scenarios.png" alt=" Scenario Analysis" style="width: 100%;">
</p>

In the example, 10 scenarios were analyzed. Not all of them necessarily lead to benefits—only a few may contribute positively to energy consumption reduction.

To support better decision-making, an `HTML report <https://github.com/EURAC-EEBgroup/pybuildingcluster/blob/master/src/pybuildingcluster/examples/report/scenario_analysis_report.html>` is generated that allows users to identify the most effective solution applied


### Python API

```python

import pandas as pd
import pybuildingcluster as pbui
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Example .env
'''
Configurazione Paths
CLUSTERING_CSV_PATH=.../data/clustering.csv
DATA_DIR=.../pybuildingcluster/data
RESULTS_DIR=.../pybuildingcluster/results
MODELS_DIR=.../pybuildingcluster/models
'''

# Esplora il dataset
building_data = explore_dataset(building_data)
# Feature columns for clustering
feature_columns = ['QHnd', 'degree_days']
# Feature columns for regression
feature_columns_regression_ = feature_columns_regression(building_data)

#%%
df_cluster = building_data[feature_columns]

# Get optimal number of clusters
optimal_k_elbow  = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="elbow", k_range=(2, 15), plot=True)
optimal_k_silhouette = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="silhouette", k_range=(2, 15), plot=True)


clustering_analyzer = pbui.ClusteringAnalyzer(random_state=42)
clusters = clustering_analyzer.fit_predict(
    data=building_data,
    feature_columns=feature_columns,
    method="silhouette",
    n_clusters=optimal_k_silhouette,
    algorithm="kmeans",
    save_clusters=True,
    output_dir="../examples/example_results"
)

results = pbui.ClusteringAnalyzer(random_state=42).fit_predict(
            data=df,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
            output_dir="../examples/example_results"
        )

#  Evaluate metrics
stats = pbui.ClusteringAnalyzer(random_state=42).get_cluster_statistics(
    results['data_with_clusters'], 
    feature_columns
)
print(stats)

# ======= Regression Models =======

models = pbui.RegressionModelBuilder(random_state=42, problem_type="regression").build_models(
    data=building_data,
    clusters=clusters,
    target_column='QHnd',
    feature_columns=feature_columns_regression_,
    models_to_train=['random_forest','xgboost','lightgbm'],
    hyperparameter_tuning="none",
    save_models=False,
    user_features=['average_opaque_surface_transmittance','average_glazed_surface_transmittance']
)

# ======= Sensitivity Analysis =======
# From cluster 1 get limits of average opaque surface transmittance and average glazed surface transmittance
cluster_1 = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == 1]
cluster_1_limits = {
    'average_opaque_surface_transmittance': {'min': float(cluster_1['average_opaque_surface_transmittance'].min()), 'max': float(cluster_1['average_opaque_surface_transmittance'].max())},
    'average_glazed_surface_transmittance': {'min': float(cluster_1['average_glazed_surface_transmittance'].min()), 'max': float(cluster_1['average_glazed_surface_transmittance'].max())}
}

sensitivity_analyzer = pbui.SensitivityAnalyzer(random_state=42)
data_with_clusters = clusters['data_with_clusters']

oat_results = sensitivity_analyzer.sensitivity_analysis(
    cluster_df=data_with_clusters,
    sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
    target='QHnd',
    modello=models[1]['best_model'],
    n_points=20,
    normalize_=True,
    plot_3d=False,
    cluster_id=None,
    feature_columns=models[1]['feature_columns']
)


# # ======= Scenario Analysis =======
# Get scenarios generated by generate_scenario.py

import numpy as np
import pandas as pd

# Extract limits for better readability
opaque_min = cluster_1_limits['average_opaque_surface_transmittance']['min']
opaque_max = cluster_1_limits['average_opaque_surface_transmittance']['max']
glazed_min = cluster_1_limits['average_glazed_surface_transmittance']['min']
glazed_max = cluster_1_limits['average_glazed_surface_transmittance']['max']

print(f"Range opaque transmittance: {opaque_min:.3f} - {opaque_max:.3f}")
print(f"Range glazed transmittance: {glazed_min:.3f} - {glazed_max:.3f}")

# Generate 10 strategic scenarios for energy efficiency
list_dict_scenarios = [
    # 1. OPTIMAL SCENARIO - Maximum efficiency (minimum values)
    {
        'name': 'High Efficiency Optimal', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min,
            'average_glazed_surface_transmittance': glazed_min
        }
    },
    
    # 2. WORST SCENARIO - Minimum efficiency (maximum values)
    {
        'name': 'Low Efficiency Worst', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_max,
            'average_glazed_surface_transmittance': glazed_max
        }
    },
    
    # 3. BALANCED SCENARIO - Intermediate values
    {
        'name': 'Balanced Performance', 
        'parameters': {
            'average_opaque_surface_transmittance': (opaque_min + opaque_max) / 2,
            'average_glazed_surface_transmittance': (glazed_min + glazed_max) / 2
        }
    },
    
    # 4. ENVELOPE OPTIMIZED - Efficient envelope, standard glazed
    {
        'name': 'Optimized Envelope', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.2,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.4
        }
    },
    
    # 5. GLAZED ADVANCED - Efficient glazed, standard envelope  
    {
        'name': 'Advanced Glazing', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.4,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.2
        }
    },
    
    # 6. CONSERVATIVE SCENARIO - Moderate improvement
    {
        'name': 'Conservative Upgrade', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.3,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.3
        }
    },
    
    # 7. AGGRESSIVE SCENARIO - High efficiency
    {
        'name': 'Aggressive Efficiency', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.1,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.15
        }
    },
    
    # 8. CURRENT MARKET STANDARD - Typical market performance
    {
        'name': 'Current Market Standard', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.6,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.5
        }
    },
    
    # 9. ECONOMIC RETROFIT SCENARIO - Cost-effective improvement
    {
        'name': 'Economic Retrofit', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.45,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.35
        }
    },
    
    # 10. HIGH PERFORMANCE SCENARIO - Quasi ottimale ma realistico
    {
        'name': 'High Performance Realistic', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.15,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.25
        }
    }
]

scenario_results = sensitivity_analyzer.compare_scenarios(
    cluster_df=data_with_clusters,
    scenarios=list_dict_scenarios,
    target='QHnd',
    feature_columns=models[0]['feature_columns'],
    modello=models[0]['best_model']
)
# EVALUATE SCENARIOS AND CREATE REPORT 
sensitivity_analyzer._plot_scenario_results(scenario_results, 'QHnd')
sensitivity_analyzer.create_scenario_report_html(
    scenario_results, 
    list_dict_scenarios, 
    'QHnd', 
    models[0]['feature_columns'],
    output_path = "../examples/example_results/scenario_analysis_report.html")
```

# Web Application 
An example of web app that uses pyBuildingCluster is available at: 
https://tools.eeb.eurac.edu/epc_clustering/piemonte/synthetic_epc

In this case it is also possible to filter the data according to 12 types of tax filters defined, which take into account minimum conditions so that an EPC can be considered good or not. 

N.B. Some parts of the app need to be updated and finalized. 


# Acknowledgment

This work was carried out within European projects: 
Moderate - Horizon Europe research and innovation programme under grant agreement No 101069834, 
with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment
