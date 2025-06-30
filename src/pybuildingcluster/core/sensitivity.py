"""
Sensitivity Analysis Module - Refactored

This module provides comprehensive sensitivity analysis functionality using
the functions created in the main sensitivity_analysis method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
import os
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer   
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import qmc
import itertools
from tqdm.notebook import tqdm
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D

class SensitivityAnalyzer:
    """
    A comprehensive sensitivity analyzer for building energy performance models.
    
    This class integrates the advanced functionality from the sensitivity_analysis method
    and provides a unified interface for all types of sensitivity analysis.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sensitivity analyzer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        self.random_state = random_state
        self.sensitivity_results = {}
        self.baseline_predictions = {}
        
    def create_parameter_variations(
        self,
        parameters: Dict[str, Dict[str, Any]],
        method: str = "grid",
        n_samples: int = 100
    ) -> Dict[str, List]:
        """
        Create parameter variations for sensitivity analysis.
        
        This method supports multiple sampling strategies for parameter space exploration.
        """
        variations = {}
        
        if method == "grid":
            # Grid-based variations
            for param_name, config in parameters.items():
                steps = config.get('steps', 20)
                variations[param_name] = np.linspace(
                    config['min'], config['max'], steps
                ).tolist()
                    
        elif method == "random":
            # Random sampling
            np.random.seed(self.random_state)
            for param_name, config in parameters.items():
                variations[param_name] = np.random.uniform(
                    config['min'], config['max'], n_samples
                ).tolist()
                
        elif method == "latin_hypercube":
            # Latin Hypercube Sampling
            try:
                
                sampler = qmc.LatinHypercube(d=len(parameters), seed=self.random_state)
                samples = sampler.random(n_samples)
                
                for idx, (param_name, config) in enumerate(parameters.items()):
                    scaled_samples = config['min'] + samples[:, idx] * (config['max'] - config['min'])
                    variations[param_name] = scaled_samples.tolist()
                    
            except ImportError:
                warnings.warn("SciPy not available for Latin Hypercube Sampling. Using random sampling.")
                return self.create_parameter_variations(parameters, method="random", n_samples=n_samples)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return variations

    def preprocess_data_prediction(self, df_: pd.DataFrame, target_column: str, apply_preprocess: bool = True):   
        """
        Preprocess data for prediction
        """
        if df_ is None:
            raise ValueError("The dataset is None")
            
        if target_column not in df_.columns:
            raise ValueError(f"The target column '{target_column}' does not exist in the dataset")
            
        # Separate feature and target
        X = df_.drop(columns=[target_column])
        y = df_[target_column]
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Define transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        if apply_preprocess:
            # Apply preprocessing
            X_processed = preprocessor.fit_transform(X)
        else:
            X_processed = X
                
        return X_processed
    
    def sensitivity_analysis(
        self, 
        cluster_df, 
        sensitivity_vars, 
        target, 
        feature_columns,
        modello=None, 
        n_points=20, 
        normalize_=True, 
        plot_3d=False, 
        cluster_id=None):
        """
        Analyze how the variation of selected parameters influences the target value within a cluster.
        
        Args:
            cluster_df: DataFrame containing the cluster data to analyze
            sensitivity_vars: List of parameters to vary in the sensitivity analysis
            target: Name of the target column to predict
            modello: Pre-trained model to use (if None, uses self.best_model)
            n_points: Number of points to use in the variation interval for each parameter
            normalize_: If normalize the variation parameters (min-max)
            plot_3d: If generate 3D plot when varying 2 parameters
            cluster_id: Optional[int] - Specific cluster to analyze, by default None (analyze all data)
            
        Returns:
            DataFrame with the sensitivity analysis results
        """

        
        # Filter data by cluster if specified
        if cluster_id is not None:
            if 'cluster' not in cluster_df.columns:
                print(f"⚠️ Warning: 'cluster' column not found in data. Using all data.")
                filtered_df = cluster_df.copy()
            else:
                filtered_df = cluster_df[cluster_df['cluster'] == cluster_id].copy()
                if len(filtered_df) == 0:
                    raise ValueError(f"No data found for cluster {cluster_id}")
                print(f"📊 Analyzing cluster {cluster_id} with {len(filtered_df)} samples")
        else:
            filtered_df = cluster_df.copy()
            print(f"📊 Analyzing all data with {len(filtered_df)} samples")
        
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("No model available. Train a model first.")
        
        if len(sensitivity_vars) > 3:
            print("Warning: more than 3 parameters specified. "
                "The analysis may take a long time and the results may be difficult to visualize.")
        
        # Create a "base point" using the mean values of the cluster for all features
        X_base = filtered_df.drop(columns=[target] if target in filtered_df.columns else [])
        # X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        # X_processed.columns = X_base.columns

        base_point = X_base.mean().to_dict()
        print(f"Base point (mean values of the cluster):")
        for k, v in base_point.items():
            if k in sensitivity_vars:
                print(f"  {k}: {v:.4f}")
        
        # Determine the range for the parameters to vary
        ranges = {}
        for param in sensitivity_vars:
            if param not in X_base.columns:
                raise ValueError(f"Parameter '{param}' not found in DataFrame")
            
            min_val = X_base[param].min()
            max_val = X_base[param].max()
            
            ranges[param] = np.linspace(min_val, max_val, n_points)
            print(f"Range for {param}: {min_val:.4f} - {max_val:.4f}")
        
        # Prepare the data structure for the results
        if len(sensitivity_vars) == 1:
            # Monodimensional case
            param = sensitivity_vars[0]
            results = []
            
            for val in tqdm(ranges[param], desc=f"Variation of {param}"):
                # Create a test point based on the base point
                test_point = base_point.copy()
                test_point[param] = val
                
                # Create DataFrame for the point
                df_test = pd.DataFrame([test_point])
                
                # Preprocessing e predizione
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                # pred_result = self._predict_with_model(modello, X_processed.values)
                pred_result = self.predict(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param: val,
                    'prediction': prediction,
                    'prediction_min': ci_lower,
                    'prediction_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # Visualization
            plt.figure(figsize=(10, 6))
            plt.plot(results_df[param], results_df['prediction'], 'b-', label='Prediction')
            
            if 'prediction_min' in results_df.columns and results_df['prediction_min'].notna().all():
                plt.fill_between(
                    results_df[param], 
                    results_df['prediction_min'], 
                    results_df['prediction_max'], 
                    alpha=0.2, 
                    color='blue', 
                    label='Confidence interval'
                )
            
            plt.xlabel(param)
            plt.ylabel(target)
            title = f"Effect of variation of {param} on {target}"
            if cluster_id is not None:
                title += f" (Cluster {cluster_id})"
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            
            # Calculate the sensitivity
            param_range = results_df[param].max() - results_df[param].min()
            pred_range = results_df['prediction'].max() - results_df['prediction'].min()
            
            if normalize_ and param_range > 0 and pred_range > 0:
                # Normalized sensitivity (variation % of the output / variation % of the input)
                param_mid = results_df[param].median()
                pred_mid = results_df['prediction'].median()
                
                elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                print(f"Normalized sensitivity of {target} with respect to {param}: {elasticity:.4f}")
                print(f"Interpretation: A variation of 1% in {param} causes a variation of {elasticity:.4f}% in {target}")
        
        elif len(sensitivity_vars) == 2:
            # Bidimensional case
            param1, param2 = sensitivity_vars
            results = []
            
            for val1, val2 in tqdm(list(itertools.product(ranges[param1], ranges[param2])), 
                                desc=f"Variation of {param1} and {param2}"):
                # Create a test point
                test_point = base_point.copy()
                # test_point = X_processed.mean().to_dict()
                test_point[param1] = val1
                test_point[param2] = val2
                
                # Create DataFrame for the point
                df_test = pd.DataFrame([test_point])
                
                # Preprocessing and prediction
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                # feature_columns_regression = ['degree_days', 'average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
                X_processed = X_processed[feature_columns]
                # pred_result = self._predict_with_model(modello, X_processed.values)
                pred_result = self.predict(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param1: val1,
                    param2: val2,
                    'prediction': prediction,
                    'prediction_min': ci_lower,
                    'prediction_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # Visualization of the heatmap
            pivot_table = results_df.pivot_table(
                index=param1, 
                columns=param2, 
                values='prediction'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap='viridis', annot=False, cbar_kws={'label': target})
            title = f"Effect of variation of {param1} and {param2} on {target}"
            if cluster_id is not None:
                title += f" (Cluster {cluster_id})"
            plt.title(title)
            plt.tight_layout()
            plt.show()
            
            # If requested, generate a 3D plot
            if plot_3d:
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                x = results_df[param1].values
                y = results_df[param2].values
                z = results_df['prediction'].values
                
                # Create a trisurf surface
                surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.7)
                
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_zlabel(target)   
                title = f"3D response surface: Effect of {param1} and {param2} on {target}"
                if cluster_id is not None:
                    title += f" (Cluster {cluster_id})"
                ax.set_title(title)
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=target)
                plt.tight_layout()
                plt.show()
                
            # Calculate partial derivatives (gradient)
            print("\nSensitivity analysis:")
            
            for param in [param1, param2]:
                other_param = param2 if param == param1 else param1
                other_param_med = results_df[other_param].median()
                
                # Filter the results with the other parameter close to the median
                filtered = results_df[np.isclose(results_df[other_param], other_param_med, rtol=0.1)]
                
                if len(filtered) >= 2:
                    param_range = filtered[param].max() - filtered[param].min()
                    pred_range = filtered['prediction'].max() - filtered['prediction'].min()
                    
                    if normalize_ and param_range > 0 and pred_range > 0:
                        # Normalized sensitivity
                        param_mid = filtered[param].median()
                        pred_mid = filtered['prediction'].median()
                        
                        elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                        print(f"Normalized sensitivity of {target} with respect to {param} (with {other_param} ≈ {other_param_med:.4f}): {elasticity:.4f}")
        
        else:
            # Multidimensional case - here we perform a one-at-a-time analysis
            results = []
            
            for param in sensitivity_vars:
                print(f"\nSensitivity analysis for {param}:")
                
                for val in tqdm(ranges[param], desc=f"Variation of {param}"):
                    # Create a test point
                    # test_point = base_point.copy()
                    test_point = X_processed.mean().to_dict()
                    test_point[param] = val
                    
                    # Create DataFrame for the point
                    df_test = pd.DataFrame([test_point])
                    
                    # Preprocessing and prediction
                    X_processed = self.preprocess_data_prediction(df_test, target_column=target, apply_preprocess=False)
                    X_processed = X_processed[feature_columns]
                    # pred_result = self._predict_with_model(modello, X_processed.values)
                    pred_result = self.predict(X_processed, modello)
                    
                    prediction = pred_result['predictions'][0]
                    ci_lower = pred_result.get('ci_lower', [None])[0]
                    ci_upper = pred_result.get('ci_upper', [None])[0]
                    
                    result_record = {'variated_parameter': param}
                    result_record.update({p: base_point[p] for p in sensitivity_vars})
                    result_record[param] = val
                    result_record['prediction'] = prediction
                    result_record['prediction_min'] = ci_lower
                    result_record['prediction_max'] = ci_upper
                    
                    results.append(result_record)
                
                # Visualize the graph for this parameter
                param_results = pd.DataFrame([r for r in results if r['variated_parameter'] == param])
                
                plt.figure(figsize=(10, 6))
                plt.plot(param_results[param], param_results['prediction'], 'b-', label='Prediction')
                
                if 'prediction_min' in param_results.columns and param_results['prediction_min'].notna().all():
                    plt.fill_between(
                        param_results[param], 
                        param_results['prediction_min'], 
                        param_results['prediction_max'], 
                        alpha=0.2, 
                        color='blue', 
                        label='Confidence interval'
                    )
                    
                plt.xlabel(param)
                plt.ylabel(target)
                title = f"Effect of variation of {param} on {target}"
                if cluster_id is not None:
                    title += f" (Cluster {cluster_id})"
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()
                
                # Calculate the sensitivity
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['prediction'].max() - param_results['prediction'].min()
                
                if normalize_ and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['prediction'].median()
                    
                    elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                    print(f"Normalized sensitivity of {target} with respect to {param}: {elasticity:.4f}")
            
            # Create a summary table of sensitivity
            sensitivity_summary = []
            
            for param in sensitivity_vars:
                param_results = pd.DataFrame([r for r in results if r['variated_parameter'] == param])
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['prediction'].max() - param_results['prediction'].min()
                
                if normalize_ and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['prediction'].median()
                    elasticity = abs((pred_range / pred_mid) / (param_range / param_mid))
                else:
                    elasticity = abs(pred_range / param_range) if param_range > 0 else 0
                    
                sensitivity_summary.append({
                    'parameter': param,
                    'elasticity': elasticity,
                    'absolute_variation': pred_range
                })
            
            sensitivity_df = pd.DataFrame(sensitivity_summary).sort_values('elasticity', ascending=False)
            print("\nSummary of sensitivity (ordered by importance):")
            print(sensitivity_df)
            
            # Bar plot of the sensitivity
            plt.figure(figsize=(10, 6))
            sns.barplot(x='parameter', y='elasticity', data=sensitivity_df)
            title = f"Sensitivity of {target} with respect to the parameters"
            if cluster_id is not None:
                title += f" (Cluster {cluster_id})"
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        return results_df

    def compare_scenarios(
        self, 
        cluster_df, 
        scenarios, 
        target, 
        feature_columns,
        modello=None):
        """
        Compares different parameter scenarios and their effect on the target.
        
        Args:
            cluster_df: DataFrame containing the cluster data to be analyzed
            scenari: List of dictionaries, each representing a scenario {'name': 'Scenario 1', 'parameters': {'param1': val1, ...}}
            target: Name of the target column to predict
            feature_columns: List of feature columns to use for prediction
            modello: Pre-trained model to use (if None, uses self.best_model)
            
        Returns:
            DataFrame: Comparison of scenarios with predicted target values
        """
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("No model available. Train a model first.")
        
        #   Creare un punto base usando i valori medi del cluster per tutte le features
        XY_copy = cluster_df.copy()
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        base_point = X_base.mean().to_dict()
        X_processed = pd.DataFrame(self.preprocess_data_prediction(XY_copy, target_column=target, apply_preprocess=False))
        X_processed.columns = X_base.columns

        # Prepare the base point as a scenario
        scenarios_con_base = [{'name': 'Base (mean cluster)', 'parameters': {}}] + scenarios
        
        # List for results
        results = []
        
        # Evaluate each scenario
        for scenario in scenarios_con_base:
            # Create a test point based on the base point
            test_point = base_point.copy()
            # test_point = X_processed.mean().to_dict()
            
            # Update with scenario parameters (for the base scenario, no changes are made)
            test_point.update(scenario['parameters'])
            
            # Create DataFrame for the point
            df_test = pd.DataFrame([test_point])
            
            # Preprocessing and prediction
            # X_processed = self.preprocess_data_prediction(df_test)
            X_processed = df_test
            X_processed = X_processed[feature_columns]
            pred_result = self.predict(X_processed, modello)
            
            prediction = pred_result['predictions'][0]
            ci_lower = pred_result.get('ci_lower', [None])[0]
            ci_upper = pred_result.get('ci_upper', [None])[0]
            
            # Result for this scenario
            result = {
                'scenario': scenario['name'],
                'prediction': prediction,
                'prediction_min': ci_lower,
                'prediction_max': ci_upper
            }
            
            # Add parameter values
            for param, value in test_point.items():
                if param in set().union(*[s['parameters'].keys() for s in scenarios if 'parameters' in s]):
                    result[f"param_{param}"] = value
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate percentage variations relative to the base scenario
        base_prediction = results_df.loc[results_df['scenario'] == 'Base (mean cluster)', 'prediction'].values[0]
        results_df['variation_pct'] = ((results_df['prediction'] - base_prediction) / base_prediction) * 100
        
        # Visualize the results
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='scenario', y='prediction', data=results_df)
        
        # Add error bars if available
        if 'prediction_min' in results_df.columns and results_df['prediction_min'].notna().all():
            for i, row in results_df.iterrows():
                ax.errorbar(
                    i, row['prediction'], 
                    yerr=[[row['prediction']-row['prediction_min']], [row['prediction_max']-row['prediction']]],
                    fmt='none', capsize=5, color='black', alpha=0.7
                )
        
        plt.title(f"Scenarios comparison - Effect on {target}")
        plt.ylabel(target)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above the bars
        for i, v in enumerate(results_df['prediction']):
            var_pct = results_df['variation_pct'].iloc[i]
            var_text = f" ({var_pct:+.1f}%)" if i > 0 else ""
            ax.text(i, v + (results_df['prediction'].max() * 0.01), 
                    f"{v:.2f}{var_text}", 
                    ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Show also a table of results
        print("\nResults table:")
        display_df = results_df[['scenario', 'prediction', 'variation_pct']].copy()
        display_df['prediction'] = display_df['prediction'].round(4)
        display_df['variation_pct'] = display_df['variation_pct'].round(2)
        display_df = display_df.rename(columns={'variation_pct': 'variation (%)'})
        print(display_df)
        return results_df
    
    def analyze(
        self,
        model,
        data: pd.DataFrame,
        scenarios: List[Dict],
        feature_columns: List[str],
        target_column: str,
        sensitivity_vars: List[str] = None,
        n_points: int = 20,
        normalize_: bool = True,
        plot_3d: bool = False,
        cluster_id: int = None,
        save_results: bool = True,
        results_dir: str = "sensitivity_results",
        create_html_report: bool = True
    ) -> Dict[str, Any]:
        """
        Main interface for comprehensive sensitivity analysis using existing class methods.
        
        This method integrates sensitivity_analysis and compare_scenarios to provide 
        unified results and comprehensive reporting.
        
        Parameters
        ----------
        model : object
            Trained model to use for predictions
        data : pd.DataFrame
            Input data for analysis
        scenarios : List[Dict]
            List of scenario definitions in format:
            [{'name': 'Scenario 1', 'parameters': {'param1': val1, ...}}, ...]
        feature_columns : List[str]
            List of feature columns to use for prediction
        target_column : str
            Name of the target column
        sensitivity_vars : List[str], optional
            Parameters for sensitivity analysis. If None, extracted from scenarios
        n_points : int, default=20
            Number of points for sensitivity analysis
        normalize_ : bool, default=True
            Whether to normalize sensitivity parameters
        plot_3d : bool, default=False
            Whether to generate 3D plots for 2-parameter sensitivity
        cluster_id : int, optional
            Specific cluster to analyze
        save_results : bool, default=True
            Whether to save results to files
        results_dir : str, default="sensitivity_results"
            Directory to save results
        create_html_report : bool, default=True
            Whether to create HTML report
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results containing:
            - scenario_analysis: Results from compare_scenarios
            - sensitivity_analysis: Results from sensitivity_analysis (if sensitivity_vars provided)
            - html_report_path: Path to HTML report (if created)
            - summary_statistics: Key metrics and insights
        """
        
        print(f"🚀 Starting comprehensive sensitivity and scenario analysis...")
        print(f"🎯 Target: {target_column}")
        print(f"📊 Total scenarios: {len(scenarios)}")
        print(f"📈 Feature columns: {len(feature_columns)}")
        
        # Store model reference
        self.best_model = model
        self.models = {target_column: model}
        
        # Initialize results dictionary
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'target_column': target_column,
            'feature_columns': feature_columns,
            'total_scenarios': len(scenarios),
            'cluster_id': cluster_id
        }
        
        # 1. SCENARIO ANALYSIS using compare_scenarios
        print(f"\n📋 Step 1: Running scenario comparison analysis...")
        try:
            scenario_results_df = self.compare_scenarios(
                cluster_df=data,
                scenarios=scenarios,
                target=target_column,
                feature_columns=feature_columns,
                modello=model
            )
            
            results['scenario_analysis'] = {
                'results_df': scenario_results_df,
                'method': 'compare_scenarios',
                'scenarios_definition': scenarios,
                'baseline_prediction': scenario_results_df[
                    scenario_results_df['scenario'].str.contains('Base', case=False, na=False)
                ]['prediction'].iloc[0] if not scenario_results_df.empty else None
            }
            
            # Extract key metrics
            non_baseline_df = scenario_results_df[
                ~scenario_results_df['scenario'].str.contains('Base', case=False, na=False)
            ]
            
            if not non_baseline_df.empty:
                results['scenario_analysis'].update({
                    'best_scenario': non_baseline_df.loc[non_baseline_df['variation_pct'].idxmin(), 'scenario'],
                    'worst_scenario': non_baseline_df.loc[non_baseline_df['variation_pct'].idxmax(), 'scenario'],
                    'max_positive_impact': non_baseline_df['variation_pct'].max(),
                    'max_negative_impact': non_baseline_df['variation_pct'].min(),
                    'avg_impact': non_baseline_df['variation_pct'].mean(),
                    'impact_std': non_baseline_df['variation_pct'].std(),
                    'positive_scenarios_count': len(non_baseline_df[non_baseline_df['variation_pct'] > 0]),
                    'negative_scenarios_count': len(non_baseline_df[non_baseline_df['variation_pct'] < 0])
                })
            
            print(f"✅ Scenario analysis completed successfully")
            
        except Exception as e:
            print(f"❌ Error in scenario analysis: {str(e)}")
            results['scenario_analysis'] = {'error': str(e)}
        
        # 2. SENSITIVITY ANALYSIS using sensitivity_analysis (if sensitivity_vars provided)
        if sensitivity_vars is not None:
            print(f"\n📈 Step 2: Running sensitivity analysis for {len(sensitivity_vars)} parameters...")
            try:
                sensitivity_results_df = self.sensitivity_analysis(
                    cluster_df=data,
                    sensitivity_vars=sensitivity_vars,
                    target=target_column,
                    feature_columns=feature_columns,
                    modello=model,
                    n_points=n_points,
                    normalize_=normalize_,
                    plot_3d=plot_3d,
                    cluster_id=cluster_id
                )
                
                results['sensitivity_analysis'] = {
                    'results_df': sensitivity_results_df,
                    'method': 'one_at_a_time_sensitivity',
                    'parameters_analyzed': sensitivity_vars,
                    'n_points': n_points,
                    'normalized': normalize_
                }
                
                print(f"✅ Sensitivity analysis completed successfully")
                
            except Exception as e:
                print(f"❌ Error in sensitivity analysis: {str(e)}")
                results['sensitivity_analysis'] = {'error': str(e)}
        else:
            print(f"⏭️ Step 2: Skipping sensitivity analysis (no sensitivity_vars provided)")
            
            # Extract sensitivity parameters from scenarios if not explicitly provided
            all_params = set()
            for scenario in scenarios:
                if 'parameters' in scenario:
                    all_params.update(scenario['parameters'].keys())
            
            if all_params:
                results['available_parameters'] = list(all_params)
                print(f"💡 Available parameters for future sensitivity analysis: {', '.join(all_params)}")
        
        # 3. GENERATE SUMMARY STATISTICS
        print(f"\n📊 Step 3: Generating summary statistics...")
        
        summary_stats = {
            'analysis_type': 'scenario_comparison',
            'total_scenarios_analyzed': len(scenarios) + 1,  # +1 for baseline
            'has_confidence_intervals': False,
            'analysis_scope': f"Cluster {cluster_id}" if cluster_id is not None else "All data"
        }
        
        # Add scenario-specific statistics
        if 'scenario_analysis' in results and 'error' not in results['scenario_analysis']:
            scenario_data = results['scenario_analysis']
            
            summary_stats.update({
                'baseline_prediction': scenario_data.get('baseline_prediction'),
                'best_scenario': scenario_data.get('best_scenario'),
                'worst_scenario': scenario_data.get('worst_scenario'),
                'impact_range': {
                    'min': scenario_data.get('max_negative_impact'),
                    'max': scenario_data.get('max_positive_impact')
                },
                'scenario_distribution': {
                    'improvement_scenarios': scenario_data.get('negative_scenarios_count', 0),
                    'degradation_scenarios': scenario_data.get('positive_scenarios_count', 0)
                }
            })
            
            # Check for confidence intervals
            scenario_df = scenario_data['results_df']
            has_ci = ('prediction_min' in scenario_df.columns and 
                    'prediction_max' in scenario_df.columns and 
                    scenario_df[['prediction_min', 'prediction_max']].notna().all().all())
            summary_stats['has_confidence_intervals'] = has_ci
        
        # Add sensitivity-specific statistics
        if 'sensitivity_analysis' in results and 'error' not in results['sensitivity_analysis']:
            sensitivity_data = results['sensitivity_analysis']
            summary_stats.update({
                'sensitivity_parameters': sensitivity_data.get('parameters_analyzed'),
                'sensitivity_method': sensitivity_data.get('method')
            })
        
        results['summary_statistics'] = summary_stats
        
        # 4. CREATE HTML REPORT
        if create_html_report and 'scenario_analysis' in results and 'error' not in results['scenario_analysis']:
            print(f"\n📄 Step 4: Creating HTML report...")
            try:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                html_filename = f"scenario_analysis_report_{target_column}_{timestamp}.html"
                html_path = os.path.join(results_dir, html_filename) if results_dir else html_filename
                
                # Create results directory if it doesn't exist
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                
                report_path = self.create_scenario_report_html(
                    results_df=results['scenario_analysis']['results_df'],
                    scenarios=scenarios,
                    target=target_column,
                    feature_columns=feature_columns,
                    output_path=html_path
                )
                
                results['html_report_path'] = report_path
                print(f"✅ HTML report created: {report_path}")
                
            except Exception as e:
                print(f"❌ Error creating HTML report: {str(e)}")
                results['html_report_error'] = str(e)
        else:
            print(f"⏭️ Step 4: Skipping HTML report creation")
        
        # 5. SAVE RESULTS (if requested)
        if save_results:
            print(f"\n💾 Step 5: Saving analysis results...")
            try:
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                
                # Save scenario results as CSV
                if 'scenario_analysis' in results and 'error' not in results['scenario_analysis']:
                    scenario_csv_path = os.path.join(results_dir, f"scenario_results_{target_column}_{timestamp}.csv")
                    results['scenario_analysis']['results_df'].to_csv(scenario_csv_path, index=False)
                    results['scenario_csv_path'] = scenario_csv_path
                    print(f"📊 Scenario results saved: {scenario_csv_path}")
                
                # Save sensitivity results as CSV (if available)
                if 'sensitivity_analysis' in results and 'error' not in results['sensitivity_analysis']:
                    sensitivity_csv_path = os.path.join(results_dir, f"sensitivity_results_{target_column}_{timestamp}.csv")
                    results['sensitivity_analysis']['results_df'].to_csv(sensitivity_csv_path, index=False)
                    results['sensitivity_csv_path'] = sensitivity_csv_path
                    print(f"📈 Sensitivity results saved: {sensitivity_csv_path}")
                
                # Save complete results as pickle
                results_pickle_path = os.path.join(results_dir, f"complete_analysis_{target_column}_{timestamp}.pkl")
                joblib.dump(results, results_pickle_path)
                results['pickle_path'] = results_pickle_path
                print(f"🔄 Complete results saved: {results_pickle_path}")
                
            except Exception as e:
                print(f"❌ Error saving results: {str(e)}")
                results['save_error'] = str(e)
        
        # 6. DISPLAY COMPREHENSIVE SUMMARY
        print(f"\n{'='*60}")
        print(f"🎉 COMPREHENSIVE ANALYSIS COMPLETED")
        print(f"{'='*60}")
        
        print(f"📊 **Analysis Summary:**")
        print(f"   • Target variable: {target_column}")
        print(f"   • Total scenarios: {summary_stats.get('total_scenarios_analyzed', 'N/A')}")
        print(f"   • Analysis scope: {summary_stats.get('analysis_scope', 'N/A')}")
        
        if 'scenario_analysis' in results and 'error' not in results['scenario_analysis']:
            scenario_stats = results['scenario_analysis']
            print(f"\n🎭 **Scenario Analysis Results:**")
            print(f"   • Baseline prediction: {scenario_stats.get('baseline_prediction', 'N/A'):.4f}")
            print(f"   • Best scenario: {scenario_stats.get('best_scenario', 'N/A')} ({scenario_stats.get('max_negative_impact', 0):.1f}%)")
            print(f"   • Worst scenario: {scenario_stats.get('worst_scenario', 'N/A')} (+{scenario_stats.get('max_positive_impact', 0):.1f}%)")
            print(f"   • Improvement scenarios: {scenario_stats.get('negative_scenarios_count', 0)}")
            print(f"   • Risk scenarios: {scenario_stats.get('positive_scenarios_count', 0)}")
            
            if summary_stats.get('has_confidence_intervals'):
                print(f"   • Confidence intervals: Available ✅")
        
        if 'sensitivity_analysis' in results and 'error' not in results['sensitivity_analysis']:
            print(f"\n📈 **Sensitivity Analysis Results:**")
            print(f"   • Parameters analyzed: {len(sensitivity_vars)}")
            print(f"   • Analysis points per parameter: {n_points}")
            print(f"   • Normalized analysis: {'Yes' if normalize_ else 'No'}")
        
        if 'html_report_path' in results:
            print(f"\n📄 **Generated Reports:**")
            print(f"   • HTML report: {results['html_report_path']}")
        
        if save_results:
            print(f"\n💾 **Saved Files:**")
            for key in ['scenario_csv_path', 'sensitivity_csv_path', 'pickle_path']:
                if key in results:
                    print(f"   • {key.replace('_path', '').replace('_', ' ').title()}: {results[key]}")
        
        print(f"\n💡 **Next Steps:**")
        if 'available_parameters' in results and sensitivity_vars is None:
            print(f"   • Run sensitivity analysis with: {results['available_parameters']}")
        print(f"   • Review HTML report for detailed insights")
        print(f"   • Consider implementing best scenario parameters")
        print(f"   • Monitor risk scenarios during operation")
        
        print(f"{'='*60}")
        
        return results
    
    def _predict_with_model(self, model, X_features):
        """
        Enhanced prediction method that handles different model formats.
        """
        if hasattr(model, 'predict'):
            return model.predict(X_features)
        elif isinstance(model, dict) and 'model' in model:
            return model['model'].predict(X_features)
        else:
            raise ValueError("Invalid model format")

    def predict(self, input_data, best_model, apply_preprocessing=True, confidence_interval=0.95):
        """
        Make predictions on new data with confidence interval
        
        Enhanced prediction method that handles different model formats and provides 
        comprehensive uncertainty estimation.
        
        Args:
            input_data: DataFrame or array with data to make predictions
            best_model: Model to use for predictions (supports multiple formats)
            apply_preprocessing: Whether to apply preprocessing to the data
            confidence_interval: Livello di confidenza per l'intervallo (default: 0.95 per 95%)
            
        Returns:
            dict: Predictions of the model with confidence intervals
        """
       
        
        # Enhanced model handling - support multiple model formats
        if hasattr(best_model, 'predict'):
            model = best_model
        elif isinstance(best_model, dict) and 'model' in best_model:
            model = best_model['model']
        elif isinstance(best_model, dict) and 'best_model' in best_model:
            model = best_model['best_model']
        else:
            raise ValueError("Invalid model format. Model must have 'predict' method or be a dict with 'model'/'best_model' key")
        
        # Prepare the input data
        # if isinstance(input_data, pd.DataFrame):
        #     # Verify that all necessary columns are present
        #     if hasattr(self, 'X_train') and isinstance(self.X_train, pd.DataFrame):
        #         missing_cols = set(self.X_train.columns) - set(input_data.columns)
        #         if missing_cols:
        #             raise ValueError(f"I dati di input mancano delle seguenti colonne: {missing_cols}")
            
        #     # Applica lo stesso preprocessing usato per i dati di training se richiesto
        #     if apply_preprocessing:
        #         print("Applicazione preprocessing ai dati...")
        #         dati_processati = self.preprocessor.transform(input_data)
        #     else:
        #         # Se non è richiesto preprocessing o non abbiamo un preprocessor, 
        #         # assumiamo che i dati siano già formattati correttamente
        #         dati_processati = input_data.values if hasattr(input_data, 'values') else input_data
        # else:
            # Assume that the data is already preprocessed correctly
        dati_processati = input_data
        
        # Make predictions with enhanced error handling
        try:
            predictions = model.predict(dati_processati)
        except Exception as e:
            raise ValueError(f"Error making predictions with model: {str(e)}")
        
        result = {
            'predictions': predictions,
            'model_type': type(model).__name__
        }
        
        # For classification problems, also offer probabilities if the model supports it
        if hasattr(self, 'problem_type') and self.problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(dati_processati)
                result['probabilities'] = probabilities
            except Exception as e:
                print(f"Warning: Could not calculate probabilities: {e}")
        
        # For regression problems, calculate confidence intervals
        if hasattr(self, 'problem_type') and self.problem_type == 'regressione':
            try:
                # Method 1: For ensemble models that support uncertainty calculation
                if hasattr(model, 'estimators_') and len(getattr(model, 'estimators_', [])) > 1:
                    # Make predictions with all ensemble estimators
                    preds_per_estimator = np.array([tree.predict(dati_processati) for tree in model.estimators_])
                    
                    # Calculate mean and standard deviation of predictions
                    mean_prediction = np.mean(preds_per_estimator, axis=0)
                    std_prediction = np.std(preds_per_estimator, axis=0)
                    
                    # Calculate confidence interval using normal distribution
                    alpha = 1 - confidence_interval
                    z_score = st.norm.ppf(1 - alpha/2)  # Two-tailed z-score
                    
                    ci_lower = mean_prediction - z_score * std_prediction
                    ci_upper = mean_prediction + z_score * std_prediction
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
                    result['uncertainty_method'] = 'ensemble_variance'
                    
                # Method 2: For XGBoost and LightGBM models
                elif any(isinstance(model, model_type) for model_type in [xgb.XGBRegressor, lgb.LGBMRegressor]):
                    # Use training error for uncertainty estimation
                    if hasattr(self, 'y_train') and hasattr(self, 'X_train_processed'):
                        # Calculate error on the training set
                        y_train_pred = model.predict(self.X_train_processed)
                        errors = np.abs(self.y_train - y_train_pred)
                        
                        # Estimate error for a certain confidence level
                        error_percentile = np.percentile(errors, confidence_interval * 100)
                        
                        # Apply this error to the new predictions
                        ci_lower = predictions - error_percentile
                        ci_upper = predictions + error_percentile
                        
                        result['ci_lower'] = ci_lower
                        result['ci_upper'] = ci_upper
                        result['uncertainty_method'] = 'training_error_percentile'
                        
                    elif hasattr(self, 'y_test') and hasattr(self, 'X_test_processed'):
                        # Fallback to test RMSE
                        rmse = np.sqrt(mean_squared_error(self.y_test, model.predict(self.X_test_processed)))
                        alpha = 1 - confidence_interval
                        z_score = st.norm.ppf(1 - alpha/2)
                        
                        ci_lower = predictions - z_score * rmse
                        ci_upper = predictions + z_score * rmse
                        
                        result['ci_lower'] = ci_lower
                        result['ci_upper'] = ci_upper
                        result['uncertainty_method'] = 'test_rmse'
                        
                # Method 3: For other models, use RMSE-based estimation
                else:
                    rmse = None
                    
                    # Try to get RMSE from test set first
                    if hasattr(self, 'y_test') and hasattr(self, 'X_test_processed'):
                        try:
                            test_pred = model.predict(self.X_test_processed)
                            rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
                            uncertainty_method = 'test_rmse'
                        except:
                            pass
                    
                    # Fallback to training set RMSE
                    if rmse is None and hasattr(self, 'y_train') and hasattr(self, 'X_train_processed'):
                        try:
                            train_pred = model.predict(self.X_train_processed)
                            rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
                            uncertainty_method = 'train_rmse'
                        except:
                            pass
                    
                    # Final fallback: use a default uncertainty estimate
                    if rmse is None:
                        # Estimate RMSE as 10% of prediction range (rough heuristic)
                        pred_range = np.max(predictions) - np.min(predictions)
                        rmse = pred_range * 0.1 if pred_range > 0 else np.std(predictions) if len(predictions) > 1 else abs(np.mean(predictions)) * 0.1
                        uncertainty_method = 'heuristic_estimate'
                        print(f"Warning: Using heuristic uncertainty estimate (RMSE ≈ {rmse:.4f})")
                    
                    if rmse is not None and rmse > 0:
                        alpha = 1 - confidence_interval
                        z_score = st.norm.ppf(1 - alpha/2)
                        
                        ci_lower = predictions - z_score * rmse
                        ci_upper = predictions + z_score * rmse
                        
                        result['ci_lower'] = ci_lower
                        result['ci_upper'] = ci_upper
                        result['uncertainty_method'] = uncertainty_method
                        result['rmse_used'] = rmse
            
            except Exception as e:
                print(f"Warning: Could not calculate confidence intervals: {e}")
                # Provide basic uncertainty bounds as fallback
                pred_std = np.std(predictions) if len(predictions) > 1 else abs(np.mean(predictions)) * 0.1
                alpha = 1 - confidence_interval
                z_score = st.norm.ppf(1 - alpha/2)
                
                result['ci_lower'] = predictions - z_score * pred_std
                result['ci_upper'] = predictions + z_score * pred_std
                result['uncertainty_method'] = 'fallback_std'
                result['warning'] = f"Basic uncertainty estimate used due to error: {str(e)}"
        
        # Add summary statistics
        result.update({
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'n_predictions': len(predictions)
        })
        
        # Add confidence interval statistics if available
        if 'ci_lower' in result and 'ci_upper' in result:
            ci_width = result['ci_upper'] - result['ci_lower']
            result.update({
                'mean_ci_width': np.mean(ci_width),
                'confidence_level': confidence_interval
            })
    
        return result
    
    def _create_sensitivity_scenarios(self, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create default scenarios based on parameter ranges."""
        scenarios = {}
        
        # Conservative scenario (all parameters at minimum)
        scenarios['conservative'] = {
            param: config['min'] for param, config in parameters.items()
        }
        
        # Aggressive scenario (all parameters at maximum)
        scenarios['aggressive'] = {
            param: config['max'] for param, config in parameters.items()
        }
        
        # Balanced scenario (all parameters at midpoint)
        scenarios['balanced'] = {
            param: (config['min'] + config['max']) / 2 
            for param, config in parameters.items()
        }
        
        # Individual parameter scenarios
        for param_name, config in parameters.items():
            # High value for this parameter, balanced for others
            scenarios[f'{param_name}_optimized'] = {
                p: config['max'] if p == param_name else (c['min'] + c['max']) / 2
                for p, c in parameters.items()
            }
            
            # Low value for this parameter, balanced for others
            scenarios[f'{param_name}_conservative'] = {
                p: config['min'] if p == param_name else (c['min'] + c['max']) / 2
                for p, c in parameters.items()
            }
        
        return scenarios
    
    def _plot_scenario_results(self, results_df: pd.DataFrame, target: str):
        """
        Plot comprehensive scenario analysis results based on compare_scenarios output.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame returned by compare_scenarios function containing:
            - scenario: scenario names
            - prediction: predicted values
            - prediction_min: lower confidence interval (optional)
            - prediction_max: upper confidence interval (optional)
            - variation_pct: percentage variation from baseline
            - param_*: parameter values used in each scenario
        target : str
            Target variable name for labeling
        scenarios : Dict, optional
            Original scenario definitions for additional analysis
        """
        if results_df.empty:
            print("⚠️ No scenario data to plot")
            return
        
        # Extract data from results DataFrame
        scenario_names = results_df['scenario'].tolist()
        predictions = results_df['prediction'].tolist()
        variations_pct = results_df['variation_pct'].tolist()
        
        # Check for confidence intervals
        has_ci = ('prediction_min' in results_df.columns and 
                'prediction_max' in results_df.columns and 
                results_df['prediction_min'].notna().all() and 
                results_df['prediction_max'].notna().all())
        
        # Get baseline values (first row should be baseline)
        baseline_idx = results_df[results_df['scenario'].str.contains('Base', case=False, na=False)].index
        if len(baseline_idx) > 0:
            baseline_prediction = results_df.loc[baseline_idx[0], 'prediction']
        else:
            baseline_prediction = predictions[0]  # Fallback to first prediction
        
        # Create comprehensive subplot layout
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Main comparison plot with error bars (matching compare_scenarios style)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Create bar plot
        x_pos = range(len(scenario_names))
        colors = ['#3498db' if 'Base' in name else '#2ecc71' if var >= 0 else '#e74c3c' 
                for name, var in zip(scenario_names, variations_pct)]
        
        bars = ax1.bar(x_pos, predictions, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
        
        # Add error bars if available
        if has_ci:
            yerr_lower = results_df['prediction'] - results_df['prediction_min']
            yerr_upper = results_df['prediction_max'] - results_df['prediction']
            ax1.errorbar(x_pos, predictions, 
                        yerr=[yerr_lower, yerr_upper],
                        fmt='none', capsize=5, color='black', alpha=0.7)
        
        # Add baseline reference line
        ax1.axhline(y=baseline_prediction, color='#e74c3c', linestyle='--', 
                    linewidth=2, alpha=0.8, label=f'Baseline: {baseline_prediction:.2f}')
        
        ax1.set_xlabel('Scenarios')
        ax1.set_ylabel(f'{target} Prediction')
        ax1.set_title(f'📊 Scenario Comparison - Effect on {target}')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace('_', '\n') for name in scenario_names], 
                            rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        
        # Add value labels above bars (matching compare_scenarios style)
        for i, (bar, pred, var) in enumerate(zip(bars, predictions, variations_pct)):
            height = bar.get_height()
            var_text = f" ({var:+.1f}%)" if i > 0 else ""  # Skip variation for baseline
            y_offset = (max(predictions) - min(predictions)) * 0.02
            ax1.text(bar.get_x() + bar.get_width()/2., height + y_offset, 
                    f"{pred:.2f}{var_text}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Percentage variation plot
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Filter out baseline for variation plot
        non_baseline_idx = [i for i, name in enumerate(scenario_names) if 'Base' not in name]
        if non_baseline_idx:
            var_names = [scenario_names[i] for i in non_baseline_idx]
            var_values = [variations_pct[i] for i in non_baseline_idx]
            var_colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in var_values]
            
            bars2 = ax2.bar(range(len(var_names)), var_values, color=var_colors, 
                            alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Scenarios')
            ax2.set_ylabel('Variation from Baseline (%)')
            ax2.set_title('📈 Percentage Changes')
            ax2.set_xticks(range(len(var_names)))
            ax2.set_xticklabels([name.replace('_', '\n') for name in var_names], 
                            rotation=45, ha='right', fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, v in zip(bars2, var_values):
                height = bar.get_height()
                y_offset = (max(var_values) - min(var_values)) * 0.02 if var_values else 0.1
                y_pos = height + y_offset if height >= 0 else height - y_offset
                ax2.text(bar.get_x() + bar.get_width()/2., y_pos, f'{v:.1f}%', 
                        ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=8, fontweight='bold')
        
        # 3. Impact ranking (horizontal bar chart)
        ax3 = fig.add_subplot(gs[1, :2])
        
        if non_baseline_idx:
            # Sort scenarios by absolute impact
            ranking_data = [(var_names[i], var_values[i]) for i in range(len(var_names))]
            ranking_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            ranked_names, ranked_values = zip(*ranking_data)
            y_pos = range(len(ranked_names))
            colors_ranked = ['#27ae60' if x >= 0 else '#e74c3c' for x in ranked_values]
            
            bars3 = ax3.barh(y_pos, ranked_values, color=colors_ranked, alpha=0.7,
                            edgecolor='black', linewidth=0.5)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([name.replace('_', ' ').title() for name in ranked_names])
            ax3.set_xlabel('Variation from Baseline (%)')
            ax3.set_title('🏆 Scenario Impact Ranking')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, v in zip(bars3, ranked_values):
                width = bar.get_width()
                x_offset = (max(ranked_values) - min(ranked_values)) * 0.01
                x_pos = width + x_offset if width >= 0 else width - x_offset
                ax3.text(x_pos, bar.get_y() + bar.get_height()/2., f'{v:.1f}%', 
                        ha='left' if width >= 0 else 'right', va='center', 
                        fontsize=9, fontweight='bold')
        
        # 4. Parameter heatmap (if parameter columns exist)
        ax4 = fig.add_subplot(gs[1, 2])
        
        param_columns = [col for col in results_df.columns if col.startswith('param_')]
        if param_columns:
            # Create parameter matrix
            param_data = results_df[param_columns].values
            param_names = [col.replace('param_', '').replace('_', ' ').title() 
                        for col in param_columns]
            
            # Create heatmap
            im = ax4.imshow(param_data.T, cmap='RdYlBu_r', aspect='auto', 
                        interpolation='nearest')
            
            ax4.set_xticks(range(len(scenario_names)))
            ax4.set_xticklabels([name.replace('_', '\n') for name in scenario_names], 
                            rotation=45, ha='right', fontsize=8)
            ax4.set_yticks(range(len(param_names)))
            ax4.set_yticklabels(param_names, fontsize=8)
            ax4.set_title('🎛️ Parameter Values')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
            cbar.set_label('Parameter Value', rotation=270, labelpad=15)
            
            # Add text annotations
            for i in range(len(param_names)):
                for j in range(len(scenario_names)):
                    text = f'{param_data[j, i]:.2f}'
                    ax4.text(j, i, text, ha="center", va="center", 
                            fontsize=7, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No parameter data\navailable', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('🎛️ Parameter Values')
        
        # 5. Statistical summary
        ax5 = fig.add_subplot(gs[2, :])
        
        # Create summary table
        summary_data = results_df[['scenario', 'prediction', 'variation_pct']].copy()
        summary_data['prediction'] = summary_data['prediction'].round(4)
        summary_data['variation_pct'] = summary_data['variation_pct'].round(2)
        
        # Remove axis and add table
        ax5.axis('off')
        
        # Create table
        table_data = []
        for _, row in summary_data.iterrows():
            table_data.append([row['scenario'], f"{row['prediction']:.4f}", 
                            f"{row['variation_pct']:.2f}%" if row['variation_pct'] != 0 else "Baseline"])
        
        table = ax5.table(cellText=table_data,
                        colLabels=['Scenario', f'{target} Prediction', 'Variation (%)'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.3, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#3498db')
                    cell.set_text_props(weight='bold', color='white')
                elif 'Base' in table_data[i-1][0]:  # Baseline row
                    cell.set_facecolor('#f8f9fa')
                else:
                    cell.set_facecolor('#ffffff')
        
        ax5.set_title('📋 Results Summary Table', fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle(f'🎭 Comprehensive Scenario Analysis for {target}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add summary statistics
        if non_baseline_idx:
            best_idx = non_baseline_idx[var_values.index(max(var_values))]
            worst_idx = non_baseline_idx[var_values.index(min(var_values))]
            
            insights = [
                f"🏆 Best: {scenario_names[best_idx]} (+{variations_pct[best_idx]:.1f}%)",
                f"⚠️ Worst: {scenario_names[worst_idx]} ({variations_pct[worst_idx]:.1f}%)",
                f"📊 Range: {min(var_values):.1f}% to {max(var_values):.1f}%"
            ]
            
            fig.text(0.02, 0.02, ' | '.join(insights), fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary (matching compare_scenarios style)
        print(f"\n📊 SCENARIO ANALYSIS SUMMARY:")
        print(f"{'='*50}")
        print(f"Total scenarios analyzed: {len(scenario_names)}")
        print(f"Baseline prediction: {baseline_prediction:.4f}")
        
        if non_baseline_idx:
            best_scenario = scenario_names[non_baseline_idx[var_values.index(max(var_values))]]
            worst_scenario = scenario_names[non_baseline_idx[var_values.index(min(var_values))]]
            
            print(f"Best performing scenario: {best_scenario} (+{max(var_values):.1f}%)")
            print(f"Worst performing scenario: {worst_scenario} ({min(var_values):.1f}%)")
            print(f"Average impact: {np.mean(var_values):.1f}% ± {np.std(var_values):.1f}%")
            print(f"Impact range: {min(var_values):.1f}% to {max(var_values):.1f}%")
        
        return results_df   

    def create_scenario_report_html(self, results_df: pd.DataFrame, scenarios, 
                                target: str, feature_columns: list, 
                                output_path: str = "scenario_analysis_report.html"):
        """
        Create a comprehensive HTML report for scenario analysis results from compare_scenarios.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame returned by compare_scenarios function containing:
            - scenario: scenario names
            - prediction: predicted values  
            - prediction_min/max: confidence intervals (optional)
            - variation_pct: percentage variation from baseline
            - param_*: parameter values used in each scenario
        scenarios
            Original scenario definitions. Can be either:
            - Dict[str, Dict]: {'scenario_name': {'param1': val1, ...}}
            - List[Dict]: [{'name': 'scenario_name', 'parameters': {'param1': val1, ...}}, ...]
        target : str
            Target variable name
        feature_columns : list
            List of feature columns used in analysis
        output_path : str
            Path for the output HTML file
            
        Returns
        -------
        str
            Path to the generated HTML report
        """
        
        # Extract key metrics from results_df
        total_scenarios = len(results_df)
        baseline_row = results_df[results_df['scenario'].str.contains('Base', case=False, na=False)]
        baseline_prediction = baseline_row['prediction'].iloc[0] if not baseline_row.empty else results_df['prediction'].iloc[0]
        
        # Filter non-baseline scenarios for analysis
        non_baseline_df = results_df[~results_df['scenario'].str.contains('Base', case=False, na=False)]
        
        # Calculate key statistics
        if not non_baseline_df.empty:
            max_positive_impact = non_baseline_df['variation_pct'].max()
            max_negative_impact = non_baseline_df['variation_pct'].min()
            avg_impact = non_baseline_df['variation_pct'].mean()
            impact_std = non_baseline_df['variation_pct'].std()
            
            # For targets where lower is better, negative variation is actually positive impact
            best_scenario = non_baseline_df.loc[non_baseline_df['variation_pct'].idxmin(), 'scenario']  # Most negative = best
            worst_scenario = non_baseline_df.loc[non_baseline_df['variation_pct'].idxmax(), 'scenario']  # Most positive = worst
        else:
            max_positive_impact = max_negative_impact = avg_impact = impact_std = 0
            best_scenario = worst_scenario = "N/A"
        
        # Check for confidence intervals
        has_ci = ('prediction_min' in results_df.columns and 
                'prediction_max' in results_df.columns and 
                results_df[['prediction_min', 'prediction_max']].notna().all().all())
        
        # Get parameter columns
        param_columns = [col for col in results_df.columns if col.startswith('param_')]
        param_names = [col.replace('param_', '') for col in param_columns]
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Scenario Analysis Report - {target}</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{ color: #2c3e50; margin-top: 30px; }}
                h1 {{ text-align: center; color: #34495e; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                .summary {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 25px; 
                    border-radius: 10px; 
                    margin: 20px 0;
                }}
                .metric-card {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px 20px; 
                    background-color: #ecf0f1; 
                    border-radius: 8px; 
                    border-left: 4px solid #3498db;
                    min-width: 180px;
                    vertical-align: top;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                .metric-positive {{ color: #27ae60; }}
                .metric-negative {{ color: #e74c3c; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    border: none; 
                    padding: 12px 15px; 
                    text-align: left; 
                }}
                th {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    font-weight: 600;
                    font-size: 14px;
                }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #e3f2fd; transition: background-color 0.3s; }}
                .baseline-row {{ background-color: #fff3cd !important; border-left: 4px solid #ffc107; }}
                .positive-impact {{ color: #27ae60; font-weight: bold; }}
                .negative-impact {{ color: #e74c3c; font-weight: bold; }}
                .neutral-impact {{ color: #6c757d; }}
                .best-scenario {{ background-color: #d4edda !important; }}
                .worst-scenario {{ background-color: #f8d7da !important; }}
                .parameter-section {{ 
                    margin: 25px 0; 
                    padding: 20px; 
                    border-left: 4px solid #3498db; 
                    background-color: #f8f9fa;
                    border-radius: 0 8px 8px 0;
                }}
                .recommendation {{ 
                    background-color: #d1ecf1; 
                    border: 1px solid #bee5eb; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 20px 0;
                }}
                .recommendation h3 {{ margin-top: 0; color: #0c5460; }}
                .recommendation ul {{ margin: 0; }}
                .recommendation li {{ margin: 8px 0; }}
                .footer {{ 
                    margin-top: 40px; 
                    padding: 20px; 
                    background-color: #34495e; 
                    color: white; 
                    border-radius: 8px; 
                    text-align: center;
                }}
                .progress-bar {{
                    background-color: #ecf0f1;
                    border-radius: 10px;
                    overflow: hidden;
                    height: 20px;
                    margin: 5px 0;
                    position: relative;
                }}
                .progress-fill {{
                    height: 100%;
                    transition: width 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                }}
                .progress-positive {{ background: linear-gradient(90deg, #27ae60, #2ecc71); }}
                .progress-negative {{ background: linear-gradient(90deg, #e74c3c, #c0392b); }}
                .emoji {{ font-size: 1.2em; }}
                .section-header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 15px 25px; 
                    margin: 30px -30px 20px -30px; 
                    border-radius: 0;
                }}
                .ci-info {{ font-size: 12px; color: #6c757d; }}
                .param-chip {{ 
                    display: inline-block; 
                    background-color: #e9ecef; 
                    padding: 3px 8px; 
                    margin: 2px; 
                    border-radius: 12px; 
                    font-size: 11px; 
                    color: #495057;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1> Scenario Analysis Report</h1>
                <p style="text-align: center; color: #6c757d; margin-top: -10px;">
                    Comprehensive analysis of parameter scenarios for <strong>{target}</strong>
                </p>
                
                <div class="summary">
                    <h2><span class="emoji">📊</span> Analysis Overview</h2>
                    <div class="metric-card">
                        <div class="metric-label">Target Variable</div>
                        <div class="metric-value">{target}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Scenarios</div>
                        <div class="metric-value">{total_scenarios}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Parameters Tested</div>
                        <div class="metric-value">{len(param_names)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Baseline Prediction</div>
                        <div class="metric-value">{baseline_prediction:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Best Impact (Lower is Better)</div>
                        <div class="metric-value metric-positive">{max_negative_impact:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Worst Impact (Higher is Worse)</div>
                        <div class="metric-value metric-negative">+{max_positive_impact:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Impact Range</div>
                        <div class="metric-value">{max_negative_impact:.1f}% to +{max_positive_impact:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">{"Confidence Intervals" if has_ci else "Features Used"}</div>
                        <div class="metric-value">{"Available" if has_ci else len(feature_columns)}</div>
                    </div>
                </div>
        """
        
        # Main results table
        html_content += """
        <div class="section-header">
            <h2><span class="emoji">📈</span> Scenario Results Summary</h2>
        </div>
        """
        
        # Create table headers
        table_headers = ["Scenario", f"{target} Prediction", "Change from Baseline", "Relative Impact", "Visual Impact"]
        if has_ci:
            table_headers.insert(2, "Confidence Interval")
        if param_names:
            table_headers.append("Parameters")
        
        html_content += f"""
        <table>
            <tr>
                {"".join(f"<th>{header}</th>" for header in table_headers)}
            </tr>
        """
        
        # Add table rows
        for _, row in results_df.iterrows():
            scenario_name = row['scenario']
            prediction = row['prediction']
            variation = row['variation_pct']
            
            # Determine row class
            row_class = ""
            if 'Base' in scenario_name:
                row_class = "baseline-row"
            elif scenario_name == best_scenario:
                row_class = "best-scenario"
            elif scenario_name == worst_scenario:
                row_class = "worst-scenario"
            
            # Format variation (for targets where lower is better)
            if variation < 0:  # Negative variation = improvement
                variation_class = "positive-impact"
                variation_text = f"{variation:.1f}% (Better)"
            elif variation > 0:  # Positive variation = worse
                variation_class = "negative-impact"
                variation_text = f"+{variation:.1f}% (Worse)"
            else:
                variation_class = "neutral-impact"
                variation_text = "Baseline"
            
            # Create progress bar for visual impact (inverted logic for lower-is-better targets)
            if variation != 0:
                bar_width = min(abs(variation) / max(abs(max_positive_impact), abs(max_negative_impact)) * 100, 100)
                bar_class = "progress-positive" if variation < 0 else "progress-negative"  # Negative = better = green
                progress_bar = f'''
                    <div class="progress-bar">
                        <div class="progress-fill {bar_class}" style="width: {bar_width}%">
                            {variation:.1f}%
                        </div>
                    </div>
                '''
            else:
                progress_bar = '<span class="neutral-impact">Baseline</span>'
            
            # Build table row
            row_html = f'<tr class="{row_class}">'
            row_html += f'<td><strong>{scenario_name}</strong></td>'
            row_html += f'<td>{prediction:.4f}</td>'
            
            # Add confidence interval if available
            if has_ci:
                ci_min = row.get('prediction_min', 'N/A')
                ci_max = row.get('prediction_max', 'N/A')
                if ci_min != 'N/A' and ci_max != 'N/A':
                    ci_text = f'[{ci_min:.4f}, {ci_max:.4f}]'
                else:
                    ci_text = 'N/A'
                row_html += f'<td class="ci-info">{ci_text}</td>'
            
            # Change from baseline
            if variation != 0:
                abs_change = prediction - baseline_prediction
                row_html += f'<td>{abs_change:+.4f}</td>'
            else:
                row_html += f'<td>-</td>'
            
            row_html += f'<td class="{variation_class}">{variation_text}</td>'
            row_html += f'<td>{progress_bar}</td>'
            
            # Add parameters if available
            if param_names:
                param_values = []
                for param in param_names:
                    param_col = f'param_{param}'
                    if param_col in row and pd.notna(row[param_col]):
                        param_values.append(f'<span class="param-chip">{param}={row[param_col]:.2f}</span>')
                row_html += f'<td>{"".join(param_values) if param_values else "-"}</td>'
            
            row_html += '</tr>'
            html_content += row_html
        
        html_content += "</table>"
        
        # Best and Worst Scenarios Analysis
        if not non_baseline_df.empty:
            html_content += """
            <div class="section-header">
                <h2><span class="emoji">🏆</span> Best & Worst Scenarios</h2>
            </div>
            """
            
            best_row = results_df[results_df['scenario'] == best_scenario].iloc[0]
            worst_row = results_df[results_df['scenario'] == worst_scenario].iloc[0]
            
            html_content += f"""
            <div style="display: flex; gap: 20px; margin: 20px 0;">
                <div style="flex: 1; padding: 20px; background-color: #d4edda; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h3 style="color: #155724; margin-top: 0;"><span class="emoji">🥇</span> Best Performing Scenario</h3>
                    <p><strong>Scenario:</strong> {best_scenario}</p>
                    <p><strong>Prediction:</strong> {best_row['prediction']:.4f}</p>
                    <p><strong>Impact:</strong> <span class="positive-impact">{best_row['variation_pct']:.1f}% (Better)</span></p>
                    <p><strong>Absolute Change:</strong> {best_row['prediction'] - baseline_prediction:.4f}</p>
                </div>
                <div style="flex: 1; padding: 20px; background-color: #f8d7da; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <h3 style="color: #721c24; margin-top: 0;"><span class="emoji">⚠️</span> Worst Performing Scenario</h3>
                    <p><strong>Scenario:</strong> {worst_scenario}</p>
                    <p><strong>Prediction:</strong> {worst_row['prediction']:.4f}</p>
                    <p><strong>Impact:</strong> <span class="negative-impact">+{worst_row['variation_pct']:.1f}% (Worse)</span></p>
                    <p><strong>Absolute Change:</strong> +{worst_row['prediction'] - baseline_prediction:.4f}</p>
                </div>
            </div>
            """
        
        # Recommendations
        html_content += """
        <div class="recommendation">
            <h3><span class="emoji">💡</span> Key Findings & Recommendations</h3>
        """
        
        # Generate insights based on the data
        insights = []
        
        if not non_baseline_df.empty:
            positive_scenarios = non_baseline_df[non_baseline_df['variation_pct'] < 0]  # Negative = better
            negative_scenarios = non_baseline_df[non_baseline_df['variation_pct'] > 0]  # Positive = worse
            
            insights.append(f"<strong>Impact Range:</strong> Scenarios show impacts ranging from {max_negative_impact:.1f}% (best) to +{max_positive_impact:.1f}% (worst)")
            
            if len(positive_scenarios) > 0:
                insights.append(f"<strong>Improvement Scenarios:</strong> {len(positive_scenarios)} out of {len(non_baseline_df)} scenarios show improvement (negative variation)")
            
            if len(negative_scenarios) > 0:
                insights.append(f"<strong>Risk Scenarios:</strong> {len(negative_scenarios)} scenarios show degradation (positive variation) requiring attention")
            
            if impact_std > 0:
                insights.append(f"<strong>Variability:</strong> Standard deviation of {impact_std:.1f}% indicates {'high' if impact_std > 5 else 'moderate' if impact_std > 2 else 'low'} scenario variability")
        
        for insight in insights:
            html_content += f"<p><span class='emoji'>📍</span> {insight}</p>"
        
        html_content += f"""
            <h4><span class="emoji">📋</span> Recommended Actions:</h4>
            <ul>
                <li><span class="emoji">🎯</span> <strong>Implement best scenario:</strong> Focus on parameters from "{best_scenario}" scenario</li>
                <li><span class="emoji">⚠️</span> <strong>Avoid risk scenarios:</strong> Monitor conditions that lead to "{worst_scenario}" outcomes</li>
                <li><span class="emoji">📊</span> <strong>Parameter optimization:</strong> Fine-tune the {len(param_names)} analyzed parameters for optimal {target}</li>
                <li><span class="emoji">🔍</span> <strong>Sensitivity monitoring:</strong> Track parameters with highest impact on {target}</li>
                <li><span class="emoji">📈</span> <strong>Scenario planning:</strong> Use this analysis for strategic decision-making</li>
                <li><span class="emoji">🔄</span> <strong>Regular updates:</strong> Re-run analysis with new data to validate findings</li>
                {"<li><span class='emoji'>📏</span> <strong>Confidence intervals:</strong> Consider prediction uncertainty in decision-making</li>" if has_ci else ""}
            </ul>
        </div>
        
        <div class="footer">
            <p><span class="emoji">📄</span> <strong>Report generated on:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><span class="emoji">🔬</span> <strong>Analysis method:</strong> Scenario Comparison Analysis</p>
            <p><span class="emoji">📊</span> <strong>Total scenarios analyzed:</strong> {total_scenarios} (including baseline)</p>
            <p><span class="emoji">🎯</span> <strong>Target variable:</strong> {target}</p>
        </div>
        
        </div>
        </body>
        </html>
        """
        
        # Save HTML file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📄 Scenario analysis HTML report saved to: {output_path}")
            print(f"🎭 Report includes {total_scenarios} scenarios with detailed analysis")
            
            return output_path
        
        except Exception as e:
            print(f"❌ Error saving HTML report: {str(e)}")
            return None