=========
Tutorials
=========

Learn pyBuildingCluster through comprehensive, step-by-step tutorials.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🏗️ Basic Clustering
      :link: basic_clustering
      :link-type: doc

      Learn the fundamentals of building energy clustering with real examples.

   .. grid-item-card:: 🤖 Regression Modeling
      :link: regression_modeling
      :link-type: doc

      Build and evaluate machine learning models for energy prediction.

   .. grid-item-card:: 🔬 Sensitivity Analysis
      :link: sensitivity_analysis
      :link-type: doc

      Conduct comprehensive sensitivity analysis and scenario comparison.

   .. grid-item-card:: 🚀 Advanced Workflows
      :link: advanced_workflows
      :link-type: doc

      Master advanced features and custom analysis workflows.

Tutorial Progression
====================

These tutorials are designed to be followed in order, building upon concepts from previous tutorials:

1. **Basic Clustering** - Start here to understand the core concepts
2. **Regression Modeling** - Learn how to build predictive models
3. **Sensitivity Analysis** - Explore parameter impacts and scenarios
4. **Advanced Workflows** - Master complex analysis pipelines

Prerequisites
=============

Before starting the tutorials, make sure you have:

* pyBuildingCluster installed (see :doc:`../installation`)
* Basic knowledge of Python and pandas
* Understanding of building energy concepts (helpful but not required)
* Sample dataset (provided in tutorials)

What You'll Learn
=================

By completing these tutorials, you'll be able to:

* **Cluster buildings** by energy performance characteristics
* **Train machine learning models** for energy demand prediction
* **Conduct sensitivity analysis** to understand parameter impacts
* **Compare scenarios** for energy efficiency evaluation
* **Generate professional reports** with visualizations and insights
* **Apply domain knowledge** for building energy analysis
* **Handle real-world data** issues and preprocessing
* **Optimize analysis** for large datasets

Tutorial Format
===============

Each tutorial includes:

* **Learning objectives** - What you'll accomplish
* **Step-by-step instructions** - Detailed code examples
* **Explanations** - Why each step is important
* **Visualizations** - Charts and plots to understand results
* **Best practices** - Professional tips and recommendations
* **Exercises** - Practice problems to reinforce learning
* **Real-world applications** - How to apply concepts to your work

Sample Data
===========

All tutorials use realistic building energy datasets that represent:

* **Energy Performance Certificates** - European building data
* **Building characteristics** - Construction year, area, envelope properties
* **Climate data** - Heating degree days, regional variations
* **System information** - Heating and ventilation systems

Getting Help
============

If you get stuck while following tutorials:

* **Check the FAQ** - Common issues and solutions
* **Review API documentation** - Detailed function references
* **Search GitHub Issues** - See if others had similar problems
* **Ask on GitHub Discussions** - Get help from the community
* **Contact support** - Professional assistance available

.. toctree::
   :maxdepth: 2
   :hidden:

   basic_clustering
   regression_modeling
   sensitivity_analysis
   advanced_workflows

Tutorial Contents
=================

Basic Clustering Tutorial
--------------------------

**Duration**: 30 minutes  
**Level**: Beginner

Learn how to:

* Load and explore building energy data
* Determine optimal number of clusters
* Perform K-means clustering
* Validate cluster quality
* Interpret clustering results
* Visualize building clusters

.. code-block:: python

   # Tutorial preview
   from pybuildingcluster import ClusteringAnalyzer
   
   analyzer = ClusteringAnalyzer()
   results = analyzer.fit_predict(
       data=building_data,
       feature_columns=['QHnd', 'degree_days'],
       method='silhouette'
   )

Regression Modeling Tutorial
----------------------------

**Duration**: 45 minutes  
**Level**: Intermediate

Learn how to:

* Prepare data for machine learning
* Train multiple model types
* Evaluate model performance
* Handle overfitting and underfitting
* Interpret feature importance
* Make predictions on new data

.. code-block:: python

   # Tutorial preview
   from pybuildingcluster import RegressionModelBuilder
   
   model_builder = RegressionModelBuilder()
   models = model_builder.build_models(
       data=building_data,
       clusters=cluster_results,
       target_column='QHnd',
       models_to_train=['random_forest', 'xgboost']
   )

Sensitivity Analysis Tutorial
-----------------------------

**Duration**: 60 minutes  
**Level**: Intermediate

Learn how to:

* Design meaningful scenarios
* Conduct one-at-a-time analysis
* Compare multiple scenarios
* Interpret sensitivity results
* Generate professional reports
* Apply results to decision-making

.. code-block:: python

   # Tutorial preview
   from pybuildingcluster import SensitivityAnalyzer
   
   sensitivity = SensitivityAnalyzer()
   results = sensitivity.compare_scenarios(
       cluster_df=data_with_clusters,
       scenarios=retrofit_scenarios,
       target='QHnd'
   )

Advanced Workflows Tutorial
---------------------------

**Duration**: 90 minutes  
**Level**: Advanced

Learn how to:

* Create custom analysis pipelines
* Handle large datasets efficiently
* Implement parallel processing
* Build reusable analysis templates
* Integrate with other tools
* Deploy analysis in production

.. code-block:: python

   # Tutorial preview
   from pybuildingcluster import GeoClusteringAnalyzer
   
   analyzer = GeoClusteringAnalyzer(
       data_path="large_dataset.csv",
       feature_columns_clustering=['QHnd', 'degree_days'],
       target_column='QHnd'
   )
   
   results = analyzer.run_complete_analysis(
       clustering_method="silhouette",
       models_to_train=['random_forest', 'xgboost', 'lightgbm'],
       sensitivity_vars=['U_wall', 'U_window', 'vintage']
   )

Hands-On Exercises
==================

Each tutorial includes practical exercises:

**Exercise 1: Cluster Validation**
   Compare different clustering methods and validate results using multiple metrics.

**Exercise 2: Model Comparison**
   Train and compare different machine learning models for the same dataset.

**Exercise 3: Scenario Design**
   Create realistic building retrofit scenarios based on actual policies.

**Exercise 4: Custom Analysis**
   Design a complete analysis workflow for a specific research question.

Next Steps After Tutorials
===========================

Once you've completed the tutorials:

1. **Explore Examples** - See :doc:`../examples/index` for real-world applications
2. **Read User Guide** - Dive deeper into specific features
3. **Check API Reference** - Understand all available functions and parameters
4. **Join Community** - Contribute to discussions and development
5. **Apply to Your Work** - Use pyBuildingCluster for your own building energy projects

.. tip::

   **Learning Path Recommendation**
   
   1. Start with :doc:`basic_clustering` even if you're experienced - it covers important domain-specific concepts
   2. Practice with your own data after each tutorial
   3. Read the corresponding API documentation for functions you use
   4. Join our GitHub Discussions to share your experience and learn from others