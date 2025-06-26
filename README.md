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

### Python API

```python
from pybuildingcluster import GeoClusteringAnalyzer

# Inizializzazione
analyzer = GeoClusteringAnalyzer(
    data_path="path/to/clustering.csv",
    feature_columns_clustering=['QHnd', 'degree_days'],
    target_column='QHnd',
    output_dir='./results'
)

# Analisi completa automatica
results = analyzer.run_complete_analysis(
    cluster_id=1,
    clustering_method="silhouette",
    models_to_train=['random_forest'],
    sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
    columns_to_remove=['energy_vectors_used']
)

# Risultati
summary = analyzer.get_summary()
print(summary)

# Acknowledgment

This work was carried out within European projects: 
Moderate - Horizon Europe research and innovation programme under grant agreement No 101069834, 
with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment
