=============
API Reference
=============

This section contains the complete API documentation for pyBuildingCluster.

Main Classes
============

.. currentmodule:: pybuildingcluster

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GeoClusteringAnalyzer
   ClusteringAnalyzer
   RegressionModelBuilder
   SensitivityAnalyzer

Core Modules
============

.. toctree::
   :maxdepth: 2

   clustering
   regression
   sensitivity
   utils

GeoClusteringAnalyzer
=====================

.. autoclass:: pybuildingcluster.GeoClusteringAnalyzer
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~GeoClusteringAnalyzer.load_and_clean_data
      ~GeoClusteringAnalyzer.perform_clustering
      ~GeoClusteringAnalyzer.build_models
      ~GeoClusteringAnalyzer.create_scenarios_from_cluster
      ~GeoClusteringAnalyzer.run_sensitivity_analysis
      ~GeoClusteringAnalyzer.run_complete_analysis
      ~GeoClusteringAnalyzer.get_summary

ClusteringAnalyzer
==================

.. autoclass:: pybuildingcluster.ClusteringAnalyzer
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~ClusteringAnalyzer.determine_optimal_clusters
      ~ClusteringAnalyzer.fit_predict
      ~ClusteringAnalyzer.get_cluster_statistics

RegressionModelBuilder
======================

.. autoclass:: pybuildingcluster.RegressionModelBuilder
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~RegressionModelBuilder.build_models
      ~RegressionModelBuilder.evaluate_models
      ~RegressionModelBuilder.predict

SensitivityAnalyzer
===================

.. autoclass:: pybuildingcluster.SensitivityAnalyzer
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~SensitivityAnalyzer.sensitivity_analysis
      ~SensitivityAnalyzer.compare_scenarios
      ~SensitivityAnalyzer.create_scenario_report_html



Constants and Enums
===================

.. currentmodule:: pybuildingcluster.constants

.. autodata:: DEFAULT_REGRESSION_FEATURES
   :annotation: = List of default regression features

.. autodata:: SUPPORTED_MODELS
   :annotation: = ['random_forest', 'xgboost', 'lightgbm', 'svr']

.. autodata:: CLUSTERING_METHODS
   :annotation: = ['elbow', 'silhouette']

Exceptions
==========

.. currentmodule:: pybuildingcluster.exceptions

.. autoexception:: PyBuildingClusterError
   :members:

.. autoexception:: DataValidationError
   :members:

.. autoexception:: ClusteringError
   :members:

.. autoexception:: ModelTrainingError
   :members:

.. autoexception:: SensitivityAnalysisError
   :members:

Type Definitions
================

.. currentmodule:: pybuildingcluster.types

.. autodata:: ClusterResults
   :annotation: = TypedDict for cluster analysis results

.. autodata:: ModelResults
   :annotation: = TypedDict for model training results

.. autodata:: SensitivityResults
   :annotation: = TypedDict for sensitivity analysis results

.. autodata:: ScenarioDefinition
   :annotation: = TypedDict for scenario definitions