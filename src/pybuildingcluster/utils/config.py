"""
Configuration Management Module

This module provides configuration management functionality for the
geoclustering sensitivity analysis system.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import warnings


class ConfigManager:
    """
    Configuration manager for geoclustering sensitivity analysis.
    
    This class handles loading, saving, and managing configuration files
    for the analysis pipeline.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters
        ----------
        config_file : Optional[str], optional
            Path to configuration file, by default None
        """
        self.config_file = config_file
        self.config = {}
        self._default_config = self._get_default_config()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            self.config = self._default_config.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'encoding': 'utf-8',
                'missing_value_strategy': 'median',
                'outlier_detection': True,
                'outlier_threshold': 3.0
            },
            'clustering': {
                'algorithm': 'kmeans',
                'method': 'elbow',
                'k_range': [2, 15],
                'random_state': 42,
                'n_init': 10
            },
            'regression': {
                'models_to_train': ['random_forest'],
                'hyperparameter_tuning': 'randomized',
                'test_size': 0.2,
                'cv_folds': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'sensitivity': {
                'analysis_types': ['one_at_a_time', 'scenario', 'global'],
                'n_samples': 1000,
                'random_state': 42
            },
            'optimization': {
                'n_trials': 100,
                'optimization_direction': 'minimize',
                'timeout': None,
                'random_state': 42,
                'n_jobs': 1
            },
            'output': {
                'save_clusters': True,
                'save_models': True,
                'save_plots': True,
                'results_dir': 'results',
                'models_dir': 'models',
                'clusters_dir': 'clusters',
                'plots_dir': 'plots'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Parameters
        ----------
        config_file : str
            Path to configuration file (JSON or YAML)
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge with default config
            self.config = self._merge_configs(self._default_config, file_config)
            self.config_file = config_file
            
            print(f"✅ Configuration loaded from: {config_file}")
            
        except Exception as e:
            warnings.warn(f"Error loading config file {config_file}: {str(e)}. Using default configuration.")
            self.config = self._default_config.copy()
    
    def save_config(self, config_file: str, format: str = 'yaml') -> None:
        """
        Save current configuration to file.
        
        Parameters
        ----------
        config_file : str
            Path to save configuration file
        format : str, optional
            Format to save ('yaml' or 'json'), by default 'yaml'
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            print(f"✅ Configuration saved to: {config_file}")
            
        except Exception as e:
            raise ValueError(f"Error saving config file {config_file}: {str(e)}")
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge user configuration with default configuration.
        
        Parameters
        ----------
        default : Dict[str, Any]
            Default configuration
        user : Dict[str, Any]
            User configuration
            
        Returns
        -------
        Dict[str, Any]
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., 'clustering.algorithm')
        default : Any, optional
            Default value if key not found, by default None
            
        Returns
        -------
        Any
            Configuration value
        """
        keys = key.split('.')
        current = self.config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Parameters
        ----------
        key : str
            Configuration key (e.g., 'clustering.algorithm')
        value : Any
            Value to set
        """
        keys = key.split('.')
        current = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            Configuration dictionary to merge
        """
        self.config = self._merge_configs(self.config, config_dict)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Parameters
        ----------
        section : str
            Section name (e.g., 'clustering', 'regression')
            
        Returns
        -------
        Dict[str, Any]
            Configuration section
        """
        return self.config.get(section, {})
    
    def validate_config(self) -> Dict[str, list]:
        """
        Validate current configuration.
        
        Returns
        -------
        Dict[str, list]
            Validation results with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        # Validate clustering configuration
        clustering = self.get_section('clustering')
        if clustering.get('algorithm') not in ['kmeans', 'dbscan', 'hierarchical']:
            errors.append("Invalid clustering algorithm. Must be one of: kmeans, dbscan, hierarchical")
        
        if clustering.get('method') not in ['elbow', 'silhouette', 'calinski_harabasz']:
            errors.append("Invalid clustering method. Must be one of: elbow, silhouette, calinski_harabasz")
        
        # Validate regression configuration
        regression = self.get_section('regression')
        valid_models = ['random_forest', 'xgboost', 'lightgbm']
        models_to_train = regression.get('models_to_train', [])
        if not isinstance(models_to_train, list) or not models_to_train:
            errors.append("models_to_train must be a non-empty list")
        else:
            invalid_models = [m for m in models_to_train if m not in valid_models]
            if invalid_models:
                warnings.append(f"Unknown models (may not be available): {invalid_models}")
        
        # Validate optimization configuration
        optimization = self.get_section('optimization')
        if optimization.get('optimization_direction') not in ['minimize', 'maximize']:
            errors.append("optimization_direction must be 'minimize' or 'maximize'")
        
        # Validate sensitivity configuration
        sensitivity = self.get_section('sensitivity')
        valid_analysis_types = ['one_at_a_time', 'scenario', 'global']
        analysis_types = sensitivity.get('analysis_types', [])
        if not isinstance(analysis_types, list):
            errors.append("analysis_types must be a list")
        else:
            invalid_types = [t for t in analysis_types if t not in valid_analysis_types]
            if invalid_types:
                errors.append(f"Invalid analysis types: {invalid_types}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def create_directories(self) -> None:
        """Create output directories based on configuration."""
        output_config = self.get_section('output')
        
        directories = [
            output_config.get('results_dir', 'results'),
            output_config.get('models_dir', 'models'),
            output_config.get('clusters_dir', 'clusters'),
            output_config.get('plots_dir', 'plots')
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {directory}")
    
    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration for ClusteringAnalyzer."""
        return self.get_section('clustering')
    
    def get_regression_config(self) -> Dict[str, Any]:
        """Get regression configuration for RegressionModelBuilder."""
        return self.get_section('regression')
    
    def get_sensitivity_config(self) -> Dict[str, Any]:
        """Get sensitivity configuration for SensitivityAnalyzer."""
        return self.get_section('sensitivity')
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration for ParameterOptimizer."""
        return self.get_section('optimization')
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config_file={self.config_file})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return f"ConfigManager(config_file={self.config_file}, sections={list(self.config.keys())})"


def create_sample_config(output_file: str = "config.yaml") -> None:
    """
    Create a sample configuration file.
    
    Parameters
    ----------
    output_file : str, optional
        Output file path, by default "config.yaml"
    """
    config_manager = ConfigManager()
    config_manager.save_config(output_file)
    print(f"📝 Sample configuration created: {output_file}")


def load_config_from_file(config_file: str) -> ConfigManager:
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
        
    Returns
    -------
    ConfigManager
        Initialized configuration manager
    """
    return ConfigManager(config_file)