============
Installation
============

pyBuildingCluster supports Python 3.8+ and is available on multiple platforms.

Quick Install
=============

The easiest way to install pyBuildingCluster is using pip:

.. code-block:: bash

    pip install pybuildingcluster

This will install the latest stable release with all required dependencies.

System Requirements
===================

**Operating Systems**
   * Linux (Ubuntu 18.04+, CentOS 7+)
   * macOS (10.14+)
   * Windows (10+)

**Python Versions**
   * Python 3.8+
   * Python 3.9 (recommended)
   * Python 3.10
   * Python 3.11

**Hardware Requirements**
   * Minimum: 4 GB RAM
   * Recommended: 8+ GB RAM for large datasets
   * Storage: 1 GB free space

Installation Options
====================

Stable Release (Recommended)
-----------------------------

Install the latest stable release from PyPI:

.. code-block:: bash

    pip install pybuildingcluster

This is the preferred method as it installs the most recent stable release.

Development Version
-------------------

To get the latest features and bug fixes:

.. code-block:: bash

    pip install git+https://github.com/EURAC-EEBgroup/pybuildingcluster.git

From Source
-----------

For development or customization:

.. code-block:: bash

    git clone https://github.com/EURAC-EEBgroup/pybuildingcluster.git
    cd pybuildingcluster
    pip install -e .

Conda Installation
------------------

.. note::
   Conda package coming soon! For now, use pip installation in conda environments.

.. code-block:: bash

    conda create -n pybuildingcluster python=3.9
    conda activate pybuildingcluster
    pip install pybuildingcluster

Virtual Environment Setup
==========================

It's recommended to use a virtual environment:

Using virtualenv
----------------

.. code-block:: bash

    # Create virtual environment
    python -m venv pybuildingcluster-env
    
    # Activate (Linux/macOS)
    source pybuildingcluster-env/bin/activate
    
    # Activate (Windows)
    pybuildingcluster-env\Scripts\activate
    
    # Install
    pip install pybuildingcluster

Using pipenv
------------

.. code-block:: bash

    # Create and activate environment
    pipenv shell --python 3.9
    
    # Install
    pipenv install pybuildingcluster

Using Poetry
------------

.. code-block:: bash

    # Initialize project
    poetry init
    
    # Add dependency
    poetry add pybuildingcluster
    
    # Activate environment
    poetry shell

Dependencies
============

Core Dependencies
-----------------

pyBuildingCluster automatically installs these required packages:

* **pandas** (>=2.0.0) - Data manipulation and analysis
* **numpy** (>=1.24.0) - Numerical computing
* **scikit-learn** (>=1.3.0) - Machine learning algorithms
* **matplotlib** (>=3.7.0) - Basic plotting
* **seaborn** (>=0.12.0) - Statistical visualization

Optional Dependencies
---------------------

Install with specific extras for additional functionality:

.. code-block:: bash

    # For advanced ML models
    pip install pybuildingcluster[ml]
    
    # For interactive visualizations
    pip install pybuildingcluster[viz]
    
    # For development
    pip install pybuildingcluster[dev]
    
    # For documentation
    pip install pybuildingcluster[docs]
    
    # All extras
    pip install pybuildingcluster[all]

**Machine Learning extras** (``[ml]``):
   * xgboost (>=2.0.0)
   * lightgbm (>=4.0.0)
   * optuna (>=3.0.0)

**Visualization extras** (``[viz]``):
   * plotly (>=5.15.0)
   * bokeh (>=3.0.0)

**Development extras** (``[dev]``):
   * pytest (>=7.0.0)
   * black (>=23.0.0)
   * flake8 (>=6.0.0)
   * mypy (>=1.0.0)

Installation Verification
==========================

Basic Verification
-------------------

Test your installation:

.. code-block:: python

    import pybuildingcluster as pbc
    print(f"pyBuildingCluster version: {pbc.__version__}")
    
    # Quick functionality test
    from pybuildingcluster import GeoClusteringAnalyzer
    print("✅ Installation successful!")

