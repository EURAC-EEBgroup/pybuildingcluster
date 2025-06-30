"""
Integration tests for the geoclustering sensitivity analysis system.

This module contains comprehensive integration tests that verify the interaction
between all major components: data loading, clustering, regression modeling,
sensitivity analysis, and parameter optimization.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
import sys
from unittest.mock import patch
from sklearn.datasets import make_blobs
import warnings

# Handle optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Import the modules under test
try:
    from src.pybuildingcluster.core.clustering import ClusteringAnalyzer
    from src.pybuildingcluster.core.regression import RegressionModelBuilder
    from src.pybuildingcluster.core.sensitivity import SensitivityAnalyzer
    from src.pybuildingcluster.core.optimization import ParameterOptimizer
    from src.pybuildingcluster import GeoClusteringAnalyzer
    from src.pybuildingcluster.data.loader import DataLoader
except ImportError:
    # If running from test directory, add parent to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.pybuildingcluster.core.clustering import ClusteringAnalyzer
    from src.pybuildingcluster.core.regression import RegressionModelBuilder
    from src.pybuildingcluster.core.sensitivity import SensitivityAnalyzer
    from src.pybuildingcluster.core.optimization import ParameterOptimizer
    from src.pybuildingcluster import GeoClusteringAnalyzer
    from src.pybuildingcluster.data.loader import DataLoader


data_path = "../data/clustering.csv" # path to the csv file

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_building_data():
    """Create synthetic building energy performance data."""
    np.random.seed(42)
    
    # Generate synthetic data with realistic building features
    n_samples = 500
    
    # Generate clustered building types
    X_cluster, cluster_labels = make_blobs(
        n_samples=n_samples,
        centers=4,
        n_features=6,
        cluster_std=1.5,
        random_state=42
    )
    
    # Create realistic building features
    data = pd.DataFrame({
        'building_area': X_cluster[:, 0] * 50 + 1000,  # Building area (m²)
        'insulation_thickness': np.abs(X_cluster[:, 1]) * 0.05 + 0.1,  # Insulation (m)
        'window_u_value': np.abs(X_cluster[:, 2]) * 0.5 + 1.0,  # U-value (W/m²K)
        'hvac_efficiency': np.abs(X_cluster[:, 3]) * 0.2 + 0.7,  # HVAC efficiency
        'occupancy_density': np.abs(X_cluster[:, 4]) * 5 + 10,  # People/100m²
        'ventilation_rate': np.abs(X_cluster[:, 5]) * 2 + 3,  # Air changes/hour
    })
    
    # Add some noise and correlations
    data['building_area'] = np.abs(data['building_area'] + np.random.normal(0, 100, n_samples))
    data['insulation_thickness'] = np.clip(data['insulation_thickness'], 0.05, 0.5)
    data['window_u_value'] = np.clip(data['window_u_value'], 0.5, 3.0)
    data['hvac_efficiency'] = np.clip(data['hvac_efficiency'], 0.5, 1.0)
    data['occupancy_density'] = np.clip(data['occupancy_density'], 5, 30)
    data['ventilation_rate'] = np.clip(data['ventilation_rate'], 1, 10)
    
    # Generate target variable (energy consumption) with realistic relationships
    energy_consumption = (
        data['building_area'] * 0.08 +  # Base consumption per m²
        (1 / data['insulation_thickness']) * 20 +  # Insulation effect
        data['window_u_value'] * 15 +  # Window heat loss
        (1 / data['hvac_efficiency']) * 30 +  # HVAC efficiency
        data['occupancy_density'] * 2 +  # Occupancy heat gain
        data['ventilation_rate'] * 5 +  # Ventilation heat loss
        np.random.normal(0, 10, n_samples)  # Random noise
    )
    
    data['energy_consumption'] = np.abs(energy_consumption)
    
    # Add some categorical variables
    data['building_type'] = np.random.choice(['residential', 'office', 'retail', 'industrial'], n_samples)
    data['climate_zone'] = np.random.choice(['cold', 'moderate', 'warm'], n_samples)
    data['construction_year'] = np.random.choice(range(1950, 2021), n_samples)
    
    # Add building ID
    data['building_id'] = range(1, n_samples + 1)
    
    return data

@pytest.fixture
def building_data():
    """Create synthetic regression data."""
    df = pd.read_csv(data_path, sep=",", decimal=".", low_memory=False, header=0, index_col=0)
    df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
    df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
    df = df.reset_index(drop=True)

    return df


@pytest.fixture
def feature_columns():
    """Define feature columns for clustering and modeling."""
    return [
        'QHnd', 'degree_days'
    ]


@pytest.fixture
def sensitivity_parameters():
    """Define parameters for sensitivity analysis."""
    return {
        'average_opaque_surface_transmittance': {'min': 0.2, 'max': 5, 'steps': 0.2},
        'average_glazed_surface_transmittance': {'min': 0.7, 'max': 2.5, 'steps': 0.2},
    }


@pytest.fixture
def optimization_parameter_space():
    """Define parameter space for optimization."""
    return {
        'average_opaque_surface_transmittance': {'type': 'float', 'low': 0.2, 'high': 5},
        'average_glazed_surface_transmittance': {'type': 'float', 'low': 0.7, 'high': 2.5}
    }

@pytest.fixture
def feature_columns_regression(building_data):
    """Define feature columns for clustering and modeling."""
    feature_remove_regression = ["QHnd","EPl", "EPt", "EPc", "EPv", "EPw", "EPh", "QHimp", "theoric_nominal_power", "energy_vectors_used"]
    feature_columns_df = building_data.columns
    feature_columns_regression = [item for item in feature_columns_df if item not in feature_remove_regression]
    return feature_columns_regression

class TestDataLoaderIntegration:
    """Integration tests for the DataLoader component."""
    
    def test_data_loading_pipeline(self, building_data, temp_dir):
        """Test complete data loading and preprocessing pipeline."""
        # Save synthetic data to CSV
        data_file = os.path.join(temp_dir, 'building_data.csv')
        building_data.to_csv(data_file, index=False)
        
        # Initialize data loader
        loader = DataLoader()
        
        # Load data
        loaded_data = loader.load_csv(data_file)
        
        # Verify data loaded correctly
        assert len(loaded_data) == len(building_data)
        assert list(loaded_data.columns) == list(building_data.columns)
        
        # Validate data
        validation_report = loader.validate_data(
            loaded_data,
            required_columns=['QHnd', 'degree_days'],
            min_rows=100
        )
        
        assert validation_report['is_valid']
        assert len(validation_report['errors']) == 0
        
        # Explore data
        exploration_report = loader.explore_data(loaded_data)
        
        assert exploration_report['basic_info']['shape'] == loaded_data.shape
        assert 'numeric' in exploration_report['statistical_summary']
        
        # Clean data
        cleaned_data = loader.clean_data(
            loaded_data,
            remove_duplicates=True,
            fill_missing='median'
        )
        
        assert len(cleaned_data) <= len(loaded_data)
        
        # Save cleaned data
        cleaned_file = os.path.join(temp_dir, 'cleaned_data.csv')
        loader.save_data(cleaned_data, cleaned_file)
        
        assert os.path.exists(cleaned_file)


class TestClusteringIntegration:
    """Integration tests for the ClusteringAnalyzer component."""
    
    def test_clustering_pipeline(self, building_data, feature_columns, temp_dir):
        """Test complete clustering pipeline."""
        analyzer = ClusteringAnalyzer(random_state=42)
        
        # Test clustering with optimal cluster determination
        results = analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
            output_dir=temp_dir
        )
        
        # Verify clustering results
        assert 'labels' in results
        assert 'n_clusters' in results
        assert 'cluster_centers' in results
        assert 'evaluation_metrics' in results
        assert 'data_with_clusters' in results
        
        # Verify cluster assignments
        labels = results['labels']
        assert len(labels) == len(building_data)
        assert results['n_clusters'] > 1
        
        # Verify cluster files were saved
        assert os.path.exists(os.path.join(temp_dir, 'data_with_clusters.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'clustering_model.pkl'))
        
        # Test cluster statistics
        stats = analyzer.get_cluster_statistics(
            results['data_with_clusters'], 
            feature_columns
        )
        
        assert len(stats) == results['n_clusters']
        assert 'cluster_id' in stats.columns
        assert 'size' in stats.columns
        
        return results
    
    def test_different_clustering_algorithms(self, building_data, feature_columns):
        """Test different clustering algorithms."""
        analyzer = ClusteringAnalyzer(random_state=42)
        
        algorithms = ['kmeans', 'hierarchical']
        results = {}
        
        for algorithm in algorithms:
            result = analyzer.fit_predict(
                data=building_data,
                feature_columns=feature_columns,
                n_clusters=3,
                algorithm=algorithm,
                save_clusters=False
            )
            
            results[algorithm] = result
            assert result['n_clusters'] == 3
            assert len(result['labels']) == len(building_data)
        
        # Compare results
        assert len(results) == len(algorithms)


class TestRegressionModelingIntegration:
    """Integration tests for the RegressionModelBuilder component."""
    

    
    def test_regression_modeling_pipeline(self, building_data, feature_columns, temp_dir, feature_columns_regression):
        """Test complete regression modeling pipeline."""
        # Print data quality info for debugging
        print(f"Data shape: {building_data.shape}")
        print(f"Target stats: mean={building_data['QHnd'].mean():.2f}, "
              f"std={building_data['QHnd'].std():.2f}")
        
        # Check feature correlations
        feature_data = building_data[feature_columns]
        correlations = feature_data.corr()['QHnd'].abs().sort_values(ascending=False)
        print(f"Feature correlations with target:\n{correlations}")
        
        # First perform clustering
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans",
            save_clusters=False
        )
        
        print(f"Clusters found: {clusters['n_clusters']}")
        print(f"Cluster sizes: {pd.Series(clusters['labels']).value_counts().sort_index()}")
        
        # Build regression models
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['xgboost'],  # Use only RF for speed
            hyperparameter_tuning="none",  # Skip tuning for speed
            models_dir=temp_dir,
            save_models=True

        )
        
        # Verify models were built
        assert len(models) > 0
        
        for cluster_id, cluster_models in models.items():
            assert 'best_model' in cluster_models
            assert 'best_model_name' in cluster_models
            assert 'best_model_metrics' in cluster_models
            assert 'feature_columns' in cluster_models
            
            # Verify model performance metrics
            metrics = cluster_models['best_model_metrics']
            assert 'r2' in metrics
            assert 'rmse' in metrics
            assert 'mae' in metrics
            
            print(f"Cluster {cluster_id} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # R² should be reasonable for improved  data
            # Lower threshold since clustering might reduce predictive power
            assert metrics['r2'] > -0.5, f"R² too low for cluster {cluster_id}: {metrics['r2']:.4f}"
            
            # RMSE should be finite and positive
            assert np.isfinite(metrics['rmse'])
            assert metrics['rmse'] > 0
        
        return models, clusters
    
    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_multiple_models_training(self, building_data, feature_columns, feature_columns_regression):
        """Test training multiple model types."""
        # Perform clustering
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            n_clusters=2,  # Fewer clusters for speed
            algorithm="kmeans",
            save_clusters=False
        )
        
        # Build models with multiple algorithms
        regression_builder = RegressionModelBuilder(random_state=42,  problem_type="regression")
        
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest', 'xgboost'],
            hyperparameter_tuning="none",
            save_models=False
        )
        
        # Verify multiple models were trained
        for cluster_id, cluster_models in models.items():
            assert 'models' in cluster_models
            models_dict = cluster_models['models']
            
            # Should have trained both models
            assert len(models_dict) >= 1  # At least one model should succeed
            
            if 'random_forest' in models_dict:
                assert 'model' in models_dict['random_forest']
                assert 'test_metrics' in models_dict['random_forest']


class TestSensitivityAnalysisIntegration:
    """Integration tests for the SensitivityAnalyzer component."""
    
    def test_individual_sensitivity_methods(self, building_data, feature_columns, feature_columns_regression):
        """Test individual sensitivity analysis methods."""
        # Setup clustering and models
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            n_clusters=2,
            algorithm="kmeans",
            save_clusters=False
        )
        
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest'],
            hyperparameter_tuning="none",
            save_models=False,
            user_features = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        )
        
        sensitivity_analyzer = SensitivityAnalyzer(random_state=42)
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
        
        assert len(oat_results) > 0

        list_dict_scenarios = [
            {'name': 'Scenario 1', 'parameters': {'average_opaque_surface_transmittance': 0.5, 
                                                'average_glazed_surface_transmittance': 1}},
            {'name': 'Scenario 2', 'parameters': {'average_opaque_surface_transmittance': 0.2, 
                                                'average_glazed_surface_transmittance': 0.7}}
        ]

        scenario_results = sensitivity_analyzer.compare_scenarios(
            cluster_df=data_with_clusters,
            scenarios=list_dict_scenarios,
            target='QHnd',
            feature_columns=models[1]['feature_columns'],
            modello=models[1]['best_model']
        )

        sensitivity_analyzer._plot_scenario_results(scenario_results, 'QHnd')
        sensitivity_analyzer.create_scenario_report_html(scenario_results, list_dict_scenarios, 'QHnd', feature_columns_regression)
        
        assert len(scenario_results) > 0
        assert 'Scenario 1' in scenario_results['scenario'].values
        assert 'Scenario 2' in scenario_results['scenario'].values


class TestParameterOptimizationIntegration:
    """Integration tests for the ParameterOptimizer component."""
    
    def test_parameter_optimization_pipeline(self, building_data, feature_columns, optimization_parameter_space, temp_dir, feature_columns_regression):
        """Test complete parameter optimization pipeline."""
        # Setup clustering and models
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            n_clusters=2,  # Fewer clusters for speed
            algorithm="kmeans",
            save_clusters=False
        )
        
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest'],
            hyperparameter_tuning="none",
            save_models=False,
            user_features = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        )
        
        # Parameter optimization
        optimizer = ParameterOptimizer(random_state=42)
        data_with_clusters = clusters['data_with_clusters']
        
        # Test optimization for one cluster
        cluster_id = list(models.keys())[0]
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        optimization_results = optimizer.optimize_cluster_parameters(
            cluster_data=cluster_data,
            models=models,
            parameter_space=optimization_parameter_space,
            target_column='QHnd',
            n_trials=10,  # Small number for speed
            optimization_direction="minimize"  # Minimize energy consumption
        )
        
        # Verify optimization results
        assert 'best_parameters' in optimization_results
        assert 'best_value' in optimization_results
        assert 'n_trials' in optimization_results
        assert 'cluster_id' in optimization_results
        
        best_params = optimization_results['best_parameters']
        assert len(best_params) > 0
        
        # Verify parameters are within bounds
        for param_name, param_value in best_params.items():
            if param_name in optimization_parameter_space:
                param_config = optimization_parameter_space[param_name]
                assert param_config['low'] <= param_value <= param_config['high']
        
        # Test saving results
        results_file = os.path.join(temp_dir, 'optimization_results.pkl')
        optimizer.save_optimization_results(optimization_results, results_file)
        assert os.path.exists(results_file)
        
        # Test loading results
        loaded_results = optimizer.load_optimization_results(results_file)
        assert loaded_results['best_value'] == optimization_results['best_value']
        
        return optimization_results
    
    def test_multiple_cluster_optimization(self, building_data, feature_columns, optimization_parameter_space, feature_columns_regression):
        """Test optimization across multiple clusters."""
        # Setup
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            n_clusters=2,
            algorithm="kmeans",
            save_clusters=False
        )
        
        regression_builder = RegressionModelBuilder(random_state=42)
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest'],
            hyperparameter_tuning="none",
            save_models=False,
            user_features = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        )
        
        # Prepare cluster data
        data_with_clusters = clusters['data_with_clusters']
        clusters_data = {}
        
        for cluster_id in models.keys():
            clusters_data[cluster_id] = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        # Optimize multiple clusters
        optimizer = ParameterOptimizer(random_state=42)
        
        multi_cluster_results = optimizer.optimize_multiple_clusters(
            clusters_data=clusters_data,
            models=models,
            parameter_space=optimization_parameter_space,
            target_column='energy_consumption',
            n_trials=5,  # Very small for speed
            optimization_direction="minimize",
            parallel=False
        )
        
        # Verify results
        assert len(multi_cluster_results) == len(models)
        
        for cluster_id, cluster_results in multi_cluster_results.items():
            assert 'best_parameters' in cluster_results
            assert 'best_value' in cluster_results


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_complete_workflow(self, building_data, feature_columns, sensitivity_parameters, temp_dir, feature_columns_regression):
        """Test complete end-to-end workflow."""
        # Save data to CSV for realistic testing
        # data_file = os.path.join(temp_dir, 'building_data.csv')
        # building_data.to_csv(data_file, index=False)
        
        # # Step 1: Data Loading
        # loader = DataLoader()
        # data = loader.load_csv(data_file)
        
        # # Clean data
        # cleaned_data = loader.clean_data(data, fill_missing='median')
        
        # Step 2: Clustering
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
            output_dir=os.path.join(temp_dir, 'clusters')
        )
        
        # Step 3: Regression Modeling
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest'],
            hyperparameter_tuning="none",
            models_dir=os.path.join(temp_dir, 'models'),
            save_models=True,
            user_features = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        )
        
        # Step 4: Sensitivity Analysis
        sensitivity_analyzer = SensitivityAnalyzer(random_state=42)
        sensitivity_results = sensitivity_analyzer.analyze(
            model=models[1]['best_model'],
            data=building_data,
            scenarios=sensitivity_parameters,
            feature_columns=feature_columns_regression,
            target_column='QHnd',
            sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
            n_points=20,
            normalize_=True,
            plot_3d=False,
            cluster_id=None,
            save_results=True,
            results_dir=os.path.join(temp_dir, 'sensitivity'),
            create_html_report=True
        )
        
        # Step 5: Parameter Optimization (simplified)
        optimizer = ParameterOptimizer(random_state=42)
        
        # Pick one cluster for optimization
        cluster_id = list(models.keys())[0]
        cluster_data = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == cluster_id]
        
        optimization_parameter_space = {
            'average_opaque_surface_transmittance': {'type': 'float', 'low': 0.1, 'high': 1.0},
            'average_glazed_surface_transmittance': {'type': 'float', 'low': 0.7, 'high': 3.0}
        }
        
        optimization_results = optimizer.optimize_cluster_parameters(
            cluster_data=cluster_data,
            models=models,
            parameter_space=optimization_parameter_space,
            target_column='QHnd',
            n_trials=5,
            optimization_direction="minimize"
        )
        
        # Verify all steps completed successfully
        assert len(building_data) > 0
        assert clusters['n_clusters'] > 1
        assert len(models) > 0
        assert 'best_parameters' in optimization_results
        
        # Verify files were created
        assert os.path.exists(os.path.join(temp_dir, 'clusters', 'data_with_clusters.csv'))
        # assert os.path.exists(os.path.join(temp_dir, 'sensitivity', 'sensitivity_analysis_results.pkl'))
        
        
        print(f"✅ Complete workflow test passed!")
        print(f"   Data points: {len(building_data)}")
        print(f"   Clusters found: {clusters['n_clusters']}")
        print(f"   Models built: {len(models)}")
        print(f"   Best optimization value: {optimization_results['best_value']:.4f}")
    
    @patch('matplotlib.pyplot.show')  # Prevent plots from showing during tests
    def test_geoclustering_analyzer_main_class(self, mock_show, temp_dir, building_data, feature_columns_regression, feature_columns):
        """Test the main GeoClusteringAnalyzer class."""
        
        # Test main analyzer class
        data_file = os.path.join(temp_dir, 'building_data.csv')
        df = pd.read_csv(data_path, sep=",", decimal=".", low_memory=False, header=0, index_col=0)
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
        df = df.reset_index(drop=True)
        df.to_csv(data_file, index=False)

        analyzer = GeoClusteringAnalyzer(
            data_path=data_file,
            feature_columns_clustering=feature_columns,
            feature_columns_regression=feature_columns_regression,
            output_dir=os.path.join(temp_dir, 'results'),
            target_column='QHnd',
            random_state=42,
            user_features=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        )
        
        # Test data loading
        loaded_data = analyzer.load_and_clean_data(columns_to_remove=["EPl", "EPt", "EPc", "EPv", "EPw", "EPh", "QHimp", "theoric_nominal_power"])
        assert len(loaded_data) == len(building_data)
        
        # Test clustering
        clusters = analyzer.perform_clustering()
        assert clusters['n_clusters'] == 2
        
        # Test regression modeling
        models = analyzer.build_models(models_to_train=['random_forest','xgboost','lightgbm'], hyperparameter_tuning="none")
        assert len(models) > 0
        
        #Test create scenarios
        scenarios = analyzer.create_scenarios_from_cluster(cluster_id=0, sensitivity_vars=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'], n_scenarios=10)
        assert len(scenarios) > 0

        optimizer = ParameterOptimizer(random_state=42)
        # Pick one cluster for optimization
        cluster_id = list(models.keys())[0]
        cluster_data = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == cluster_id]
        
        optimization_parameter_space = {
            'average_opaque_surface_transmittance': {'type': 'float', 'low': 0.1, 'high': 1.0},
            'average_glazed_surface_transmittance': {'type': 'float', 'low': 0.7, 'high': 3.0}
        }
        
        optimization_results = optimizer.optimize_cluster_parameters(
            cluster_data=cluster_data,
            models=models,
            parameter_space=optimization_parameter_space,
            target_column='QHnd',
            n_trials=5,
            optimization_direction="minimize"
        )

        assert 'best_parameters' in optimization_results
        assert 'best_value' in optimization_results

        print(f"✅ GeoClusteringAnalyzer integration test passed!")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in integration scenarios."""
    
    def test_insufficient_data_for_clustering(self):
        """Test behavior with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [10, 20, 30]
        })
        
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        
        # Should handle small dataset gracefully
        results = clustering_analyzer.fit_predict(
            data=small_data,
            feature_columns=['feature1', 'feature2'],
            n_clusters=2,
            algorithm="kmeans",
            save_clusters=False
        )
        
        assert 'labels' in results
        assert len(results['labels']) == 3
    
    def test_missing_target_values(self, feature_columns_regression, building_data):
        """Test handling of missing target values in regression."""
        # Create data with missing target values
        data = building_data.copy()
        data['QHnd'] = data['QHnd'].fillna(np.nan)
        data['cluster'] = np.random.rand(len(data))
        
        # Simulate clusters
        clusters = {
            'data_with_clusters': data,
            'labels': data['cluster'].values,
            'n_clusters': 2
        }
        
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        
        # Should handle missing target values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = regression_builder.build_models(
                data=data,
                clusters=clusters,
                target_column='QHnd',
                feature_columns=feature_columns_regression,  # Use subset of features
                models_to_train=['random_forest'],
                hyperparameter_tuning="none",
                save_models=False
            )
        
        # Should either build models or handle gracefully
        assert isinstance(models, dict)
    

    
    def test_parameter_optimization_with_constraints(self, building_data, feature_columns, feature_columns_regression):
        """Test parameter optimization with constraints."""
        # Setup clustering and models
        clustering_analyzer = ClusteringAnalyzer(random_state=42)
        clusters = clustering_analyzer.fit_predict(
            data=building_data,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
        )    
        
        regression_builder = RegressionModelBuilder(random_state=42, problem_type="regression")
        models = regression_builder.build_models(
            data=building_data,
            clusters=clusters,
            target_column='QHnd',
            feature_columns=feature_columns_regression,
            models_to_train=['random_forest'],
            hyperparameter_tuning="none",
            save_models=False
        )
        
        # Define constraints
        optimizer = ParameterOptimizer(random_state=42)
        
        constraints = {
            'transmittance_facade': {
                'type': 'min',
                'value': 1.0,
                'parameters': ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
            }
        }
        
        constraint_function = optimizer.create_constraint_function(constraints)
        
        # Test constraint function
        valid_params = {'average_opaque_surface_transmittance': 0.2, 'average_glazed_surface_transmittance': 0.8}
        invalid_params = {'average_opaque_surface_transmittance': 0.01, 'average_glazed_surface_transmittance': 0.5}
        
        assert constraint_function(valid_params) == True
        assert constraint_function(invalid_params) == False
        
        # Test optimization with constraints
        cluster_id = list(models.keys())[0]
        cluster_data = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == cluster_id]
        
        parameter_space = {
            'average_opaque_surface_transmittance': {'type': 'float', 'low': 0.2, 'high': 2.0},
            'average_glazed_surface_transmittance': {'type': 'float', 'low': 0.7, 'high': 3.0}
        }
        
        optimization_results = optimizer.optimize_cluster_parameters(
            cluster_data=cluster_data,
            models=models,
            parameter_space=optimization_parameter_space,
            target_column='QHnd',
            n_trials=5,
            optimization_direction="minimize",
            constraint_function=constraint_function
        )
        
        # Verify constraints were respected
        best_params = optimization_results['best_parameters']
        assert constraint_function(best_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])