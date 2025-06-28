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

The example use Synthetic dataset of Energy Performance Certificates of public buildings of Italian Region.

Il processo di sintetizzazione è stato realizzato tramite la libreria fornita da MOSTLY AI <https://github.com/mostly-ai/mostlyai>
All'interno della cartella synthetic è possibile visualizzare il report relativo alla generazione del dataset sintetizzato.

### How to use the synthetic dataset and the library.
The synthesized dataset, in addition to preserving the same statistical characteristics as the original data, represents a very useful resource for evaluating potential energy efficiency improvements across a building stock where only some buildings' performance data is known. In fact, by generating synthetic data, more robust assessments can be made, since the analysis can be based on a larger number of buildings that closely resemble those present in the actual territory.

Once the data is generated, it can be divided into different clusters based on specific properties.
In the example provided, the clustering is done using the QHnd property (heating energy demand of the building) and degree days (days with temperatures below 18°C).

Each cluster is then analyzed through a sensitivity analysis of selected parameters.
In this case, the average thermal transmittance of opaque components and the average thermal transmittance of transparent components are used.

The analysis shows how varying these parameters can lead to significant reductions in energy consumption for the selected cluster.
For instance, the map illustrates that the dark blue areas correspond to the greatest reductions in consumption, as they represent combinations of low values for both selected parameters. However, this may not always represent the best performance-to-cost ratio. In fact, considerable savings can also be achieved by slightly improving these parameters, which requires a lower investment.

Moreover, specific retrofit scenarios can be identified. In the example, 10 scenarios were analyzed. Not all of them necessarily lead to benefits—only a few may contribute positively to energy consumption reduction.

To support better decision-making, an HTML report is generated that allows users to identify the most effective solution applied


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
