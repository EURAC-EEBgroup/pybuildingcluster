"""
Parameter Optimization Module

This module provides optimization functionality using Optuna for finding optimal
parameter combinations within building clusters to maximize or minimize target objectives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Union, Tuple
import joblib
import os
import warnings
import matplotlib.pyplot as plt

# Handle optional Optuna dependency
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    TPESampler = None
    MedianPruner = None


class ParameterOptimizer:
    """
    A comprehensive parameter optimizer for building cluster analysis.
    
    This class uses Optuna to optimize sensitive parameters within clusters
    with the objective of maximizing or minimizing predicted targets.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = 1):
        """
        Initialize the parameter optimizer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        n_jobs : int, optional
            Number of parallel jobs, by default 1
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        
        # Configure optuna logging if available
        if HAS_OPTUNA:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            warnings.warn("Optuna not available. Using simplified optimization approach.")
    
    def _check_optuna_availability(self):
        """Check if Optuna is available and raise appropriate error if not."""
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is required for parameter optimization but is not installed. "
                "Install it with: pip install optuna"
            )
    
    def define_parameter_space(
        self, 
        parameter_config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Define the parameter space for optimization.
        
        Parameters
        ----------
        parameter_config : Dict[str, Dict[str, Any]]
            Configuration for each parameter with type and bounds
            Example:
            {
                'temperature': {'type': 'float', 'low': 18.0, 'high': 26.0},
                'humidity': {'type': 'float', 'low': 30.0, 'high': 70.0},
                'occupancy': {'type': 'int', 'low': 1, 'high': 10},
                'hvac_mode': {'type': 'categorical', 'choices': ['heating', 'cooling', 'off']}
            }
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Validated parameter space configuration
        """
        validated_config = {}
        
        for param_name, config in parameter_config.items():
            param_type = config.get('type')
            
            if param_type == 'float':
                if 'low' not in config or 'high' not in config:
                    raise ValueError(f"Float parameter '{param_name}' must have 'low' and 'high' bounds")
                validated_config[param_name] = {
                    'type': 'float',
                    'low': float(config['low']),
                    'high': float(config['high']),
                    'step': config.get('step', None)
                }
                
            elif param_type == 'int':
                if 'low' not in config or 'high' not in config:
                    raise ValueError(f"Integer parameter '{param_name}' must have 'low' and 'high' bounds")
                validated_config[param_name] = {
                    'type': 'int',
                    'low': int(config['low']),
                    'high': int(config['high']),
                    'step': config.get('step', 1)
                }
                
            elif param_type == 'categorical':
                if 'choices' not in config:
                    raise ValueError(f"Categorical parameter '{param_name}' must have 'choices'")
                validated_config[param_name] = {
                    'type': 'categorical',
                    'choices': config['choices']
                }
                
            else:
                raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_name}'")
        
        return validated_config
    
    def _simplified_optimization(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, Dict[str, Any]],
        n_trials: int = 100,
        optimization_direction: str = "maximize"
    ) -> Dict[str, Any]:
        """
        Simplified optimization when Optuna is not available.
        Uses random search with basic tracking.
        
        Parameters
        ----------
        objective_function : Callable
            Function to optimize
        parameter_space : Dict[str, Dict[str, Any]]
            Parameter space configuration
        n_trials : int, optional
            Number of trials, by default 100
        optimization_direction : str, optional
            Direction of optimization, by default "maximize"
            
        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        np.random.seed(self.random_state)
        
        best_value = float('-inf') if optimization_direction == "maximize" else float('inf')
        best_params = {}
        trial_history = []
        
        print(f"Running simplified optimization with {n_trials} trials...")
        print("Note: Install Optuna for advanced optimization features.")
        
        for trial_idx in range(n_trials):
            # Sample parameters randomly
            params = {}
            for param_name, config in parameter_space.items():
                if config['type'] == 'float':
                    if config.get('step'):
                        # Generate stepped values
                        n_steps = int((config['high'] - config['low']) / config['step'])
                        step_idx = np.random.randint(0, n_steps + 1)
                        params[param_name] = config['low'] + step_idx * config['step']
                    else:
                        params[param_name] = np.random.uniform(config['low'], config['high'])
                elif config['type'] == 'int':
                    params[param_name] = np.random.randint(config['low'], config['high'] + 1)
                elif config['type'] == 'categorical':
                    params[param_name] = np.random.choice(config['choices'])
            
            # Evaluate objective function
            try:
                # Create a mock trial object for compatibility
                class MockTrial:
                    def __init__(self, number, params):
                        self.number = number
                        self.params = params
                        self.user_attrs = {}
                    
                    def suggest_float(self, name, low, high, step=None):
                        return self.params[name]
                    
                    def suggest_int(self, name, low, high, step=1):
                        return self.params[name]
                    
                    def suggest_categorical(self, name, choices):
                        return self.params[name]
                    
                    def set_user_attr(self, key, value):
                        self.user_attrs[key] = value
                
                mock_trial = MockTrial(trial_idx, params)
                value = objective_function(mock_trial)
                
                # Check if this is the best value so far
                is_better = (
                    (optimization_direction == "maximize" and value > best_value) or
                    (optimization_direction == "minimize" and value < best_value)
                )
                
                if is_better and not (np.isnan(value) or np.isinf(value)):
                    best_value = value
                    best_params = params.copy()
                
                trial_history.append({
                    'trial': trial_idx,
                    'value': value,
                    'params': params.copy()
                })
                
            except Exception as e:
                warnings.warn(f"Trial {trial_idx} failed: {str(e)}")
                continue
        
        # Calculate success rate
        valid_trials = [t for t in trial_history if not (np.isnan(t['value']) or np.isinf(t['value']))]
        success_rate = len(valid_trials) / n_trials if n_trials > 0 else 0
        
        return {
            'best_parameters': best_params,
            'best_value': best_value,
            'n_trials': n_trials,
            'n_valid_trials': len(valid_trials),
            'success_rate': success_rate,
            'optimization_direction': optimization_direction,
            'method': 'simplified_random_search',
            'trial_history': trial_history
        }
    
    def optimize_cluster_parameters(
        self,
        cluster_data: pd.DataFrame,
        models: Dict,
        parameter_space: Dict[str, Dict[str, Any]],
        target_column: str,
        n_trials: int = 100,
        optimization_direction: str = "maximize",
        constraint_function: Optional[Callable] = None,
        study_name: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a specific cluster.
        
        Parameters
        ----------
        cluster_data : pd.DataFrame
            Data for the specific cluster
        models : Dict
            Dictionary of trained models
        parameter_space : Dict[str, Dict[str, Any]]
            Parameter space configuration
        target_column : str
            Target column to optimize
        n_trials : int, optional
            Number of optimization trials, by default 100
        optimization_direction : str, optional
            Direction of optimization ('maximize' or 'minimize'), by default "maximize"
        constraint_function : Optional[Callable], optional
            Function to check constraints, by default None
        study_name : Optional[str], optional
            Name for the optimization study, by default None
        timeout : Optional[int], optional
            Timeout in seconds, by default None
            
        Returns
        -------
        Dict[str, Any]
            Optimization results including best parameters and value
        """
        # Validate inputs
        if cluster_data.empty:
            raise ValueError("cluster_data cannot be empty")
        
        if not models:
            raise ValueError("models dictionary cannot be empty")
        
        if not parameter_space:
            raise ValueError("parameter_space cannot be empty")
        
        # Get cluster ID from data
        cluster_id = None
        if 'cluster' in cluster_data.columns:
            unique_clusters = cluster_data['cluster'].unique()
            if len(unique_clusters) == 1:
                cluster_id = unique_clusters[0]
            else:
                raise ValueError("cluster_data must contain data from only one cluster")
        else:
            # Assume cluster 0 if no cluster column
            cluster_id = 0
        
        # Check if model exists for this cluster
        if cluster_id not in models:
            available_clusters = list(models.keys())
            raise ValueError(f"No model found for cluster {cluster_id}. Available clusters: {available_clusters}")
        
        # Get model and feature information
        cluster_model_info = models[cluster_id]
        if 'best_model' not in cluster_model_info:
            raise ValueError(f"No best_model found for cluster {cluster_id}")
        
        model = cluster_model_info['best_model']
        feature_columns = cluster_model_info.get('feature_columns', [])
        
        if not feature_columns:
            raise ValueError(f"No feature_columns found for cluster {cluster_id}")
        
        # Validate parameter space
        validated_space = self.define_parameter_space(parameter_space)
        
        # Check that all parameters in parameter_space are in feature_columns
        missing_features = [param for param in validated_space.keys() if param not in feature_columns]
        if missing_features:
            warnings.warn(f"Parameters not in feature_columns will be ignored: {missing_features}")
            validated_space = {k: v for k, v in validated_space.items() if k in feature_columns}
        
        if not validated_space:
            raise ValueError("No valid parameters found in parameter_space that match feature_columns")
        
        # Create enhanced objective function
        def objective(trial):
            try:
                # Sample parameters from the defined space
                params = {}
                for param_name, config in validated_space.items():
                    if config['type'] == 'float':
                        if config.get('step'):
                            params[param_name] = trial.suggest_float(
                                param_name, config['low'], config['high'], step=config['step']
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, config['low'], config['high']
                            )
                    elif config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, config['low'], config['high'], step=config.get('step', 1)
                        )
                    elif config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, config['choices']
                        )
                
                # Apply constraints if provided
                if constraint_function and not constraint_function(params):
                    penalty_value = float('-inf') if optimization_direction == "maximize" else float('inf')
                    trial.set_user_attr('constraint_violated', True)
                    return penalty_value
                
                # Create sample data with new parameters
                sample_data = cluster_data.copy()
                
                # Update only the parameters being optimized
                for param_name, value in params.items():
                    sample_data[param_name] = value
                
                # Prepare features for prediction
                try:
                    X_sample = sample_data[feature_columns].fillna(sample_data[feature_columns].median())
                    
                    # Ensure we have the right number of features
                    if X_sample.shape[1] != len(feature_columns):
                        raise ValueError(f"Feature dimension mismatch: expected {len(feature_columns)}, got {X_sample.shape[1]}")
                    
                    # Make predictions
                    predictions = model.predict(X_sample.values)
                    
                    # Calculate objective value (mean prediction)
                    objective_value = np.mean(predictions)
                    
                    # Store additional information in trial
                    trial.set_user_attr('parameters', params)
                    trial.set_user_attr('prediction', objective_value)
                    trial.set_user_attr('n_samples', len(sample_data))
                    trial.set_user_attr('constraint_violated', False)
                    
                    # Check for invalid predictions
                    if np.isnan(objective_value) or np.isinf(objective_value):
                        penalty_value = float('-inf') if optimization_direction == "maximize" else float('inf')
                        trial.set_user_attr('invalid_prediction', True)
                        return penalty_value
                    
                    return objective_value
                    
                except Exception as model_error:
                    # Handle model prediction errors
                    penalty_value = float('-inf') if optimization_direction == "maximize" else float('inf')
                    trial.set_user_attr('model_error', str(model_error))
                    warnings.warn(f"Model prediction error in trial {trial.number}: {str(model_error)}")
                    return penalty_value
                    
            except Exception as trial_error:
                # Handle any other errors in the trial
                penalty_value = float('-inf') if optimization_direction == "maximize" else float('inf')
                trial.set_user_attr('trial_error', str(trial_error))
                warnings.warn(f"Trial error in trial {trial.number}: {str(trial_error)}")
                return penalty_value
        
        # Run optimization
        if HAS_OPTUNA:
            return self._optuna_optimization(
                objective, validated_space, n_trials, optimization_direction, 
                cluster_id, cluster_data, study_name, timeout
            )
        else:
            return self._simplified_optimization_with_constraints(
                objective, validated_space, n_trials, optimization_direction,
                cluster_id, cluster_data, constraint_function
            )
    
    def _optuna_optimization(
        self, 
        objective: Callable,
        validated_space: Dict,
        n_trials: int,
        optimization_direction: str,
        cluster_id: int,
        cluster_data: pd.DataFrame,
        study_name: Optional[str],
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Run optimization using Optuna."""
        # Create study with enhanced configuration
        direction = "maximize" if optimization_direction == "maximize" else "minimize"
        
        # Adaptive sampler configuration
        n_startup_trials = min(10, max(5, n_trials // 10))
        sampler = TPESampler(
            seed=self.random_state, 
            n_startup_trials=n_startup_trials,
            n_ei_candidates=24,
            gamma=None,  # Let TPE decide
            constant_liar=True  # For parallel optimization
        )
        
        # Pruner configuration
        pruner = MedianPruner(
            n_startup_trials=max(5, n_startup_trials // 2),
            n_warmup_steps=max(5, n_trials // 20),
            interval_steps=1
        )
        
        study_name = study_name or f"cluster_{cluster_id}_optimization_{self.random_state}"
        
        # Create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Run optimization with error handling
        try:
            print(f"Starting optimization for cluster {cluster_id}...")
            print(f"Parameters to optimize: {list(validated_space.keys())}")
            print(f"Direction: {optimization_direction}")
            print(f"Number of trials: {n_trials}")
            
            self.study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=1,  # Force single job to avoid multiprocessing issues
                show_progress_bar=True,
                catch=(Exception,)  # Catch exceptions and continue
            )
            
            # Check if optimization found any valid trials
            valid_trials = [t for t in self.study.trials if t.value is not None]
            
            if not valid_trials:
                raise RuntimeError("No valid trials found during optimization")
            
            # Store results
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            # Calculate success rate
            success_rate = len(valid_trials) / len(self.study.trials) if self.study.trials else 0
            
            # Compile comprehensive results
            results = {
                'best_parameters': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(self.study.trials),
                'n_valid_trials': len(valid_trials),
                'success_rate': success_rate,
                'optimization_direction': optimization_direction,
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_data),
                'optimized_parameters': list(validated_space.keys()),
                'study': self.study,
                'parameter_importance': self._calculate_parameter_importance(),
                'convergence_plot_data': self._get_convergence_data(),
                'parameter_relationships': self._analyze_parameter_relationships(),
                'optimization_summary': self._create_optimization_summary(),
                'method': 'optuna'
            }
            
            print(f"Optimization completed successfully!")
            print(f"Best value: {self.best_value:.4f}")
            print(f"Success rate: {success_rate:.2%}")
            
            return results
            
        except Exception as e:
            error_msg = f"Optimization failed for cluster {cluster_id}: {str(e)}"
            print(f"Error: {error_msg}")
            
            # Return partial results if any trials were completed
            if hasattr(self, 'study') and self.study and self.study.trials:
                valid_trials = [t for t in self.study.trials if t.value is not None]
                if valid_trials:
                    # Find best trial manually
                    if optimization_direction == "maximize":
                        best_trial = max(valid_trials, key=lambda t: t.value)
                    else:
                        best_trial = min(valid_trials, key=lambda t: t.value)
                    
                    return {
                        'error': error_msg,
                        'partial_results': True,
                        'best_parameters': best_trial.params,
                        'best_value': best_trial.value,
                        'n_trials': len(self.study.trials),
                        'n_valid_trials': len(valid_trials),
                        'cluster_id': cluster_id,
                        'method': 'optuna_partial'
                    }
            
            raise RuntimeError(error_msg)
    
    def _simplified_optimization_with_constraints(
        self,
        objective: Callable,
        validated_space: Dict,
        n_trials: int,
        optimization_direction: str,
        cluster_id: int,
        cluster_data: pd.DataFrame,
        constraint_function: Optional[Callable]
    ) -> Dict[str, Any]:
        """Run simplified optimization with constraint handling."""
        result = self._simplified_optimization(
            objective, validated_space, n_trials, optimization_direction
        )
        
        # Add additional metadata
        result.update({
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_data),
            'optimized_parameters': list(validated_space.keys()),
            'parameter_importance': {},  # Not available in simplified mode
            'convergence_plot_data': self._get_simplified_convergence_data(result['trial_history']),
            'parameter_relationships': {},  # Not available in simplified mode
            'optimization_summary': self._create_simplified_optimization_summary(result)
        })
        
        return result
    
    def _get_simplified_convergence_data(self, trial_history: List[Dict]) -> Dict[str, List]:
        """Get convergence data from simplified optimization."""
        if not trial_history:
            return {}
        
        trial_numbers = [t['trial'] + 1 for t in trial_history]
        values = [t['value'] for t in trial_history]
        
        # Calculate best value so far
        best_values = []
        current_best = float('-inf')  # Assume maximize for simplicity
        
        for value in values:
            if not (np.isnan(value) or np.isinf(value)):
                current_best = max(current_best, value)
            best_values.append(current_best)
        
        return {
            'trial_numbers': trial_numbers,
            'values': values,
            'best_values': best_values
        }
    
    def _create_simplified_optimization_summary(self, results: Dict) -> Dict[str, Any]:
        """Create summary for simplified optimization."""
        trial_history = results.get('trial_history', [])
        valid_trials = [t for t in trial_history if not (np.isnan(t['value']) or np.isinf(t['value']))]
        
        if not valid_trials:
            return {'status': 'no_valid_trials'}
        
        values = [t['value'] for t in valid_trials]
        
        return {
            'status': 'completed',
            'total_trials': len(trial_history),
            'valid_trials': len(valid_trials),
            'best_value': results['best_value'],
            'value_statistics': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            },
            'error_analysis': {
                'constraint_violations': 0,  # Not tracked in simplified mode
                'model_errors': 0,  # Not tracked in simplified mode
                'trial_errors': 0,  # Not tracked in simplified mode
                'success_rate': len(valid_trials) / len(trial_history) if trial_history else 0
            }
        }
    
    def optimize_multiple_clusters(
        self,
        clusters_data: Dict[int, pd.DataFrame],
        models: Dict,
        parameter_space: Dict[str, Dict[str, Any]],
        target_column: str,
        n_trials: int = 100,
        optimization_direction: str = "maximize",
        constraint_function: Optional[Callable] = None,
        parallel: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Optimize parameters for multiple clusters.
        
        Parameters
        ----------
        clusters_data : Dict[int, pd.DataFrame]
            Dictionary mapping cluster IDs to their data
        models : Dict
            Dictionary of trained models
        parameter_space : Dict[str, Dict[str, Any]]
            Parameter space configuration
        target_column : str
            Target column to optimize
        n_trials : int, optional
            Number of optimization trials per cluster, by default 100
        optimization_direction : str, optional
            Direction of optimization, by default "maximize"
        constraint_function : Optional[Callable], optional
            Function to check constraints, by default None
        parallel : bool, optional
            Whether to run optimizations in parallel, by default False
            
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Optimization results for each cluster
        """
        results = {}
        
        if parallel and self.n_jobs > 1 and HAS_OPTUNA:
            # Parallel optimization using joblib (only if Optuna is available)
            from joblib import Parallel, delayed
            
            def optimize_single_cluster(cluster_id, cluster_data):
                return cluster_id, self.optimize_cluster_parameters(
                    cluster_data=cluster_data,
                    models=models,
                    parameter_space=parameter_space,
                    target_column=target_column,
                    n_trials=n_trials,
                    optimization_direction=optimization_direction,
                    constraint_function=constraint_function,
                    study_name=f"cluster_{cluster_id}_optimization"
                )
            
            parallel_results = Parallel(n_jobs=self.n_jobs)(
                delayed(optimize_single_cluster)(cluster_id, cluster_data)
                for cluster_id, cluster_data in clusters_data.items()
            )
            
            results = dict(parallel_results)
            
        else:
            # Sequential optimization
            for cluster_id, cluster_data in clusters_data.items():
                print(f"Optimizing parameters for cluster {cluster_id}...")
                
                cluster_results = self.optimize_cluster_parameters(
                    cluster_data=cluster_data,
                    models=models,
                    parameter_space=parameter_space,
                    target_column=target_column,
                    n_trials=n_trials,
                    optimization_direction=optimization_direction,
                    constraint_function=constraint_function,
                    study_name=f"cluster_{cluster_id}_optimization"
                )
                
                results[cluster_id] = cluster_results
        
        return results
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance from the study."""
        if not HAS_OPTUNA or self.study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception:
            return {}
    
    def _get_convergence_data(self) -> Dict[str, List]:
        """Get convergence data for plotting."""
        if not HAS_OPTUNA or self.study is None:
            return {}
        
        trials = self.study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        trial_numbers = list(range(1, len(values) + 1))
        
        # Calculate best value so far
        best_values = []
        current_best = float('-inf') if self.study.direction.name == 'MAXIMIZE' else float('inf')
        
        for value in values:
            if self.study.direction.name == 'MAXIMIZE':
                current_best = max(current_best, value)
            else:
                current_best = min(current_best, value)
            best_values.append(current_best)
        
        return {
            'trial_numbers': trial_numbers,
            'values': values,
            'best_values': best_values
        }
    
    def _analyze_parameter_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between parameters and objective value."""
        if not HAS_OPTUNA or self.study is None:
            return {}
        
        df = self.study.trials_dataframe()
        if df.empty:
            return {}
        
        # Get parameter columns
        param_columns = [col for col in df.columns if col.startswith('params_')]
        
        if not param_columns:
            return {}
        
        # Calculate correlations
        correlations = {}
        for col in param_columns:
            param_name = col.replace('params_', '')
            if df[col].dtype in ['float64', 'int64']:
                correlation = df[col].corr(df['value'])
                if not pd.isna(correlation):
                    correlations[param_name] = correlation
        
        return {
            'parameter_correlations': correlations,
            'trials_dataframe': df
        }
    
    def _create_optimization_summary(self) -> Dict[str, Any]:
        """Create a summary of the optimization process."""
        if not HAS_OPTUNA or not self.study:
            return {}
        
        trials = self.study.trials
        valid_trials = [t for t in trials if t.value is not None]
        
        if not valid_trials:
            return {'status': 'no_valid_trials'}
        
        values = [t.value for t in valid_trials]
        
        summary = {
            'status': 'completed',
            'total_trials': len(trials),
            'valid_trials': len(valid_trials),
            'best_value': self.study.best_value,
            'value_statistics': {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        }
        
        # Count constraint violations and errors
        constraint_violations = sum(1 for t in trials if t.user_attrs.get('constraint_violated', False))
        model_errors = sum(1 for t in trials if 'model_error' in t.user_attrs)
        trial_errors = sum(1 for t in trials if 'trial_error' in t.user_attrs)
        
        summary['error_analysis'] = {
            'constraint_violations': constraint_violations,
            'model_errors': model_errors,
            'trial_errors': trial_errors,
            'success_rate': len(valid_trials) / len(trials) if trials else 0
        }
        
        return summary
    
    def create_constraint_function(
        self, 
        constraints: Dict[str, Dict[str, Any]]
    ) -> Callable:
        """
        Create a constraint function from configuration.
        
        Parameters
        ----------
        constraints : Dict[str, Dict[str, Any]]
            Constraint configuration
            Example:
            {
                'energy_budget': {'type': 'max', 'value': 1000, 'parameters': ['heating', 'cooling']},
                'comfort_range': {'type': 'range', 'min': 20, 'max': 25, 'parameter': 'temperature'}
            }
            
        Returns
        -------
        Callable
            Constraint function
        """
        def constraint_check(params: Dict[str, Any]) -> bool:
            for constraint_name, config in constraints.items():
                constraint_type = config['type']
                
                if constraint_type == 'max':
                    # Maximum sum constraint
                    param_names = config['parameters']
                    total = sum(params.get(p, 0) for p in param_names)
                    if total > config['value']:
                        return False
                        
                elif constraint_type == 'min':
                    # Minimum sum constraint
                    param_names = config['parameters']
                    total = sum(params.get(p, 0) for p in param_names)
                    if total < config['value']:
                        return False
                        
                elif constraint_type == 'range':
                    # Range constraint for single parameter
                    param_name = config['parameter']
                    value = params.get(param_name, 0)
                    if not (config['min'] <= value <= config['max']):
                        return False
                        
                elif constraint_type == 'equality':
                    # Equality constraint
                    param_names = config['parameters']
                    values = [params.get(p, 0) for p in param_names]
                    if len(set(values)) > 1:  # All values should be equal
                        return False
                        
                elif constraint_type == 'custom':
                    # Custom constraint function
                    custom_func = config['function']
                    if not custom_func(params):
                        return False
            
            return True
        
        return constraint_check
    
    def optimize(
        self,
        clusters: Dict,
        models: Dict,
        objective_function: Callable,
        parameter_space: Dict[str, Dict[str, Any]],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Main optimization interface for backward compatibility.
        
        Parameters
        ----------
        clusters : Dict
            Cluster data dictionary
        models : Dict
            Trained models dictionary
        objective_function : Callable
            Custom objective function
        parameter_space : Dict[str, Dict[str, Any]]
            Parameter space configuration
        n_trials : int, optional
            Number of trials, by default 100
            
        Returns
        -------
        Dict[str, Any]
            Optimization results
        """
        if HAS_OPTUNA:
            # Create a simple study
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_function, n_trials=n_trials)
            
            return {
                'best_parameters': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study': study,
                'method': 'optuna'
            }
        else:
            # Use simplified optimization
            return self._simplified_optimization(
                objective_function, parameter_space, n_trials, "maximize"
            )
    
    def save_optimization_results(
        self, 
        results: Dict[str, Any], 
        filepath: str,
        include_study: bool = False
    ):
        """
        Save optimization results to file.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results
        filepath : str
            Path to save the results
        include_study : bool, optional
            Whether to include the full study object, by default False
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare results for saving
        save_data = results.copy()
        
        if not include_study and 'study' in save_data:
            # Remove study object to reduce file size
            del save_data['study']
        
        # Save using joblib
        joblib.dump(save_data, filepath)
        print(f"Optimization results saved to: {filepath}")
    
    def load_optimization_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimization results from file.
        
        Parameters
        ----------
        filepath : str
            Path to the results file
            
        Returns
        -------
        Dict[str, Any]
            Loaded optimization results
        """
        return joblib.load(filepath)
    
    def plot_optimization_history(self, results: Dict[str, Any]):
        """
        Plot optimization history.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results containing convergence data
        """
        convergence_data = results.get('convergence_plot_data', {})
        
        if not convergence_data:
            print("No convergence data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot optimization history
        trial_numbers = convergence_data['trial_numbers']
        values = convergence_data['values']
        best_values = convergence_data['best_values']
        
        ax1.plot(trial_numbers, values, 'b.', alpha=0.6, label='Trial values')
        ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best value so far')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True)
        
        # Plot parameter importance
        importance = results.get('parameter_importance', {})
        if importance:
            params = list(importance.keys())
            importances = list(importance.values())
            
            ax2.barh(params, importances)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
            ax2.grid(True, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No importance data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_relationships(self, results: Dict[str, Any]):
        """
        Plot parameter relationships and correlations.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results containing relationship data
        """
        relationships = results.get('parameter_relationships', {})
        correlations = relationships.get('parameter_correlations', {})
        
        if not correlations:
            print("No parameter correlation data available for plotting")
            return
        
        # Plot correlations
        params = list(correlations.keys())
        corr_values = list(correlations.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(params, corr_values)
        
        # Color bars based on correlation strength
        for i, (bar, corr) in enumerate(zip(bars, corr_values)):
            if corr > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
            bar.set_alpha(abs(corr))
        
        plt.xlabel('Parameters')
        plt.ylabel('Correlation with Objective')
        plt.title('Parameter-Objective Correlations')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_optimization_report(
        self, 
        results: Dict[str, Any], 
        cluster_id: Optional[int] = None
    ) -> str:
        """
        Generate a text report of optimization results.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Optimization results
        cluster_id : Optional[int], optional
            Cluster ID for the report, by default None
            
        Returns
        -------
        str
            Formatted optimization report
        """
        report = []
        
        if cluster_id is not None:
            report.append(f"OPTIMIZATION REPORT - CLUSTER {cluster_id}")
        else:
            report.append("OPTIMIZATION REPORT")
        
        report.append("=" * 50)
        report.append("")
        
        # Best results
        best_params = results.get('best_parameters', {})
        best_value = results.get('best_value', 'N/A')
        
        report.append("BEST RESULTS:")
        report.append(f"Best objective value: {best_value}")
        report.append("Best parameters:")
        for param, value in best_params.items():
            report.append(f"  {param}: {value}")
        report.append("")
        
        # Optimization info
        n_trials = results.get('n_trials', 'N/A')
        direction = results.get('optimization_direction', 'N/A')
        method = results.get('method', 'unknown')
        
        report.append("OPTIMIZATION INFO:")
        report.append(f"Method: {method}")
        report.append(f"Number of trials: {n_trials}")
        report.append(f"Optimization direction: {direction}")
        report.append("")
        
        # Parameter importance (only available with Optuna)
        importance = results.get('parameter_importance', {})
        if importance:
            report.append("PARAMETER IMPORTANCE:")
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for param, imp in sorted_importance:
                report.append(f"  {param}: {imp:.4f}")
            report.append("")
        
        # Parameter correlations (only available with Optuna)
        relationships = results.get('parameter_relationships', {})
        correlations = relationships.get('parameter_correlations', {})
        if correlations:
            report.append("PARAMETER CORRELATIONS:")
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for param, corr in sorted_corr:
                report.append(f"  {param}: {corr:.4f}")
        
        # Success rate
        success_rate = results.get('success_rate', 'N/A')
        if success_rate != 'N/A':
            report.append("")
            report.append(f"SUCCESS RATE: {success_rate:.2%}")
        
        return "\n".join(report)