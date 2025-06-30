#!/usr/bin/env python3
"""
Setup script for pybuildingcluster.
Versione pulita senza conflitti con pyproject.toml.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

def read_file(filename):
    """Read a file and return its contents."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def read_requirements():
    """Read requirements from requirements.txt."""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        # Fallback requirements
        requirements = [
            "numpy>=2.1.3",
            "pandas>=2.2.3",
            "scipy>=1.15.1",
            "scikit-learn>=1.5.2",
            "matplotlib>=3.10.3",
            "seaborn>=0.13.2",
            "plotly>=5.24.1",
            "joblib>=1.4.2",
            "tqdm>=4.60.0",
            "xgboost>=2.1.4",
            "lightgbm>=4.6.0",
            "optuna>=4.3.0",
            "pydantic>=2.7.2",
            "python-dotenv>=1.0.0",
        ]
    return requirements

# Determina se stiamo usando src layout
src_layout = os.path.exists('src/pybuildingcluster')
if src_layout:
    package_dir = {"": "src"}
    packages = find_packages(where="src")
else:
    package_dir = {}
    packages = find_packages()

print(f"Using {'src' if src_layout else 'flat'} layout")
print(f"Found packages: {packages}")

setup(
    name="pybuildingcluster",
    version="1.0.7",
    description="A Python library for clustering energy performance data and conducting sensitivity analysis on building clusters",
    long_description=read_file("README.md") or "Building Energy Performance Clustering Library",
    long_description_content_type="text/markdown",
    author="Daniele Antonucci",
    author_email="daniele.antonucci@eurac.edu",
    url="https://github.com/EURAC-EEBgroup/pybuildingcluster",
    
    # Package configuration
    package_dir=package_dir,
    packages=packages,
    
    # Requirements
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'isort>=5.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'interactive': [
            'jupyter>=1.0.0',
            'ipywidgets>=7.6.0',
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="clustering sensitivity analysis energy efficiency buildings machine learning",
    
    # Entry points
    entry_points={
        'console_scripts': [
            'pybuildingcluster-analysis=pybuildingcluster.cli:main',
        ],
    },
    
    # Package data
    include_package_data=True,
    zip_safe=False,
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/EURAC-EEBgroup/pybuildingcluster/issues',
        'Source': 'https://github.com/EURAC-EEBgroup/pybuildingcluster',
        'Documentation': 'https://github.com/EURAC-EEBgroup/pybuildingcluster',
    },
)