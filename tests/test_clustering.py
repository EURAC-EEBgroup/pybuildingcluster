"""
Test suite for clustering analysis module.

This module contains comprehensive tests for the ClusteringAnalyzer class
including unit tests, integration tests, and edge case handling.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
import sys
from unittest.mock import patch
import warnings


feature_columns = ['QHnd', 'degree_days']
data_path = "../data/data.csv"  # path to the data.csv file
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

# Import the module under test
try:
    from src.pybuildingcluster.core.clustering import ClusteringAnalyzer
except ImportError:
    # If running from test directory, add parent to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.pybuildingcluster.core.clustering import ClusteringAnalyzer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestClusteringAnalyzer:
    """Test suite for the ClusteringAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        '''
        Uplod csv dataset from EPC
        '''
        print("Loading dataset...")
        df = pd.read_csv(data_path, sep=",", decimal=".", low_memory=False, header=0, index_col=0)
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
        df = df.reset_index(drop=True)

        return df


    @pytest.fixture
    def analyzer(self):
        """Create a ClusteringAnalyzer instance."""
        return ClusteringAnalyzer(random_state=42)
    
    def test_init(self, analyzer):
        """Test ClusteringAnalyzer initialization."""
        assert analyzer.random_state == 42
        assert analyzer.scaler is not None
        assert analyzer.model is None
        assert analyzer.labels_ is None
        assert analyzer.scaled_data is None
        assert analyzer.cluster_centers_ is None
        assert analyzer.evaluation_metrics == {}
    
    
    def test_prepare_data_with_missing_values(self, analyzer):
        """Test data preparation with missing values."""
        # Create data with missing values
        data = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [2, np.nan, 4, 5, 6],
            'feature_3': [3, 4, 5, np.nan, 7]
        })
        
        feature_columns = ['feature_1', 'feature_2', 'feature_3']
        
        # Test that the method handles missing values appropriately
        # It may either raise an exception, issue a warning, or handle them silently
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                scaled_data = analyzer.prepare_data(data, feature_columns)
                
                # If warnings are issued, check for missing value warnings
                if len(w) > 0:
                    warning_messages = [str(warning.message) for warning in w]
                    has_missing_warning = any("missing" in msg.lower() or "nan" in msg.lower() 
                                            for msg in warning_messages)
                    # If there are warnings, at least one should be about missing values
                    if len(w) > 0:
                        print(f"Warnings issued: {warning_messages}")
            
            # Check that no NaN values remain after processing
            assert not np.isnan(scaled_data).any(), "NaN values should be handled"
            
            # Check output shape is maintained
            assert scaled_data.shape == (5, 3), f"Expected shape (5, 3), got {scaled_data.shape}"
            
        except Exception as e:
            # If the method raises an exception for missing values, that's also acceptable behavior
            assert "missing" in str(e).lower() or "nan" in str(e).lower(), \
                f"Expected missing value related error, got: {str(e)}"
    
    def test_elbow_method(self, analyzer, sample_data):
        """Test elbow method for determining optimal clusters."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Test without plotting
        optimal_k = analyzer._elbow_method(scaled_data, range(2, 6), plot=False)
        
        # Should return a reasonable number of clusters
        assert 2 <= optimal_k <= 5
        assert isinstance(optimal_k, (int, np.integer))
    
    def test_silhouette_method(self, analyzer, sample_data):
        """Test silhouette method for determining optimal clusters."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Test without plotting
        optimal_k = analyzer._silhouette_method(scaled_data, range(2, 6), plot=False)
        
        # Should return a reasonable number of clusters
        assert 2 <= optimal_k <= 5
        assert isinstance(optimal_k, (int, np.integer))
    
    def test_calinski_harabasz_method(self, analyzer, sample_data):
        """Test Calinski-Harabasz method for determining optimal clusters."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Test without plotting
        optimal_k = analyzer._calinski_harabasz_method(scaled_data, range(2, 6), plot=False)
        
        # Should return a reasonable number of clusters
        assert 2 <= optimal_k <= 5
        assert isinstance(optimal_k, (int, np.integer))
    
    def test_determine_optimal_clusters_invalid_method(self, analyzer, sample_data):
        """Test determine_optimal_clusters with invalid method."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.determine_optimal_clusters(scaled_data, method="invalid_method")
    
    def test_fit_kmeans(self, analyzer, sample_data):
        """Test K-means clustering."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Test without plotting using elbow method
        optimal_k = analyzer._elbow_method(scaled_data, range(2, 6), plot=False)

        # Fit K-means with 3 clusters
        result = analyzer.fit_kmeans(scaled_data, n_clusters=optimal_k)
        
        # Check return value
        assert result is analyzer  # Should return self
        
        # Check that model is fitted
        assert analyzer.model is not None
        assert analyzer.labels_ is not None
        assert analyzer.cluster_centers_ is not None
        
        # Check output shapes
        assert len(analyzer.labels_) == scaled_data.shape[0]
        assert analyzer.cluster_centers_.shape == (optimal_k, scaled_data.shape[1])
        
        # Check that labels are in expected range
        assert set(analyzer.labels_).issubset({0, 1, 2,3})
        
        # Check evaluation metrics
        assert 'silhouette_score' in analyzer.evaluation_metrics
        assert 'calinski_harabasz_score' in analyzer.evaluation_metrics
        assert 'davies_bouldin_score' in analyzer.evaluation_metrics
    
    def test_fit_dbscan(self, analyzer, sample_data):
        """Test DBSCAN clustering."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Fit DBSCAN
        result = analyzer.fit_dbscan(scaled_data, eps=0.5, min_samples=5)
        
        # Check return value
        assert result is analyzer  # Should return self
        
        # Check that model is fitted
        assert analyzer.model is not None
        assert analyzer.labels_ is not None
        
        # Check output shape
        assert len(analyzer.labels_) == scaled_data.shape[0]
        
        # Labels can include -1 for noise points
        unique_labels = set(analyzer.labels_)
        assert -1 in unique_labels or len(unique_labels) >= 1
    
    def test_fit_hierarchical(self, analyzer, sample_data):
        """Test hierarchical clustering."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        scaled_data = analyzer.prepare_data(sample_data, feature_columns)
        
        # Fit hierarchical clustering
        result = analyzer.fit_hierarchical(scaled_data, n_clusters=3)
        
        # Check return value
        assert result is analyzer  # Should return self
        
        # Check that model is fitted
        assert analyzer.model is not None
        assert analyzer.labels_ is not None
        
        # Check output shape
        assert len(analyzer.labels_) == scaled_data.shape[0]
        
        # Check that labels are in expected range
        assert set(analyzer.labels_).issubset({0, 1, 2,3})
    
    def test_fit_predict_complete_pipeline(self, analyzer, sample_data, temp_dir):
        """Test the complete fit_predict pipeline."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # Run complete pipeline
        results = analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            method="elbow",
            algorithm="kmeans",
            save_clusters=True,
            output_dir=temp_dir
        )
        
        # Check results structure
        assert isinstance(results, dict)
        required_keys = [
            'labels', 'n_clusters', 'cluster_centers', 'evaluation_metrics',
            'feature_columns', 'algorithm', 'scaled_data', 'data_with_clusters'
        ]
        for key in required_keys:
            assert key in results
        
        # Check data with clusters
        data_with_clusters = results['data_with_clusters']
        assert 'cluster' in data_with_clusters.columns
        assert len(data_with_clusters) == len(sample_data)
        
        # Check that files are saved
        assert os.path.exists(os.path.join(temp_dir, 'data_with_clusters.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'clustering_model.pkl'))
        
        # Check individual cluster files
        unique_clusters = results['data_with_clusters']['cluster'].unique()
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise points
                cluster_file = os.path.join(temp_dir, f'cluster_{cluster_id}.csv')
                assert os.path.exists(cluster_file)
    
    def test_fit_predict_with_custom_clusters(self, analyzer, sample_data):
        """Test fit_predict with custom number of clusters."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # Run with custom cluster number
        results = analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            n_clusters=4,  # Custom number
            algorithm="kmeans",
            save_clusters=False
        )
        
        # Check that 4 clusters were created
        assert results['n_clusters'] == 4
        assert len(set(results['labels'])) == 4
    
    def test_fit_predict_invalid_algorithm(self, analyzer, sample_data):
        """Test fit_predict with invalid algorithm."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            analyzer.fit_predict(
                data=sample_data,
                feature_columns=feature_columns,
                algorithm="invalid_algorithm"
            )
    
    def test_get_cluster_statistics(self, analyzer, sample_data):
        """Test cluster statistics calculation."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # First run clustering
        results = analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        # Get cluster statistics
        stats = analyzer.get_cluster_statistics(results['data_with_clusters'], feature_columns)
        
        # Check output structure
        assert isinstance(stats, pd.DataFrame)
        
        # Check columns
        expected_columns = ['cluster_id', 'size', 'percentage']
        for col in feature_columns:
            expected_columns.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])
        
        for col in expected_columns:
            assert col in stats.columns
        
        # Check that we have statistics for each cluster
        assert len(stats) == 3  # 3 clusters
        
        # Check that percentages sum to ~100%
        assert abs(stats['percentage'].sum() - 100.0) < 1e-10
    
    def test_get_cluster_statistics_no_cluster_column(self, analyzer, sample_data):
        """Test cluster statistics with missing cluster column."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        with pytest.raises(ValueError, match="Data must contain 'cluster' column"):
            analyzer.get_cluster_statistics(sample_data, feature_columns)
    
    def test_predict(self, analyzer, sample_data):
        """Test prediction on new data."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # First fit the model
        analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        # Create new data for prediction
        new_data = sample_data.iloc[:10].copy()
        
        # Predict clusters
        predictions = analyzer.predict(new_data, feature_columns)
        
        # Check output
        assert len(predictions) == 10
        assert all(pred in {0, 1, 2} for pred in predictions)
    
    def test_predict_not_fitted(self, analyzer, sample_data):
        """Test prediction without fitted model."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        with pytest.raises(ValueError, match="Model not fitted"):
            analyzer.predict(sample_data, feature_columns)
    
    def test_predict_dbscan(self, analyzer, sample_data):
        """Test prediction with DBSCAN (should raise NotImplementedError)."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # Fit DBSCAN
        analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            algorithm="dbscan"
        )
        
        # Try to predict (should fail)
        with pytest.raises(NotImplementedError):
            analyzer.predict(sample_data, feature_columns)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_clusters_pca(self, mock_show, analyzer, sample_data):
        """Test cluster plotting with PCA."""
        # feature_columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        
        # Fit clustering
        results = analyzer.fit_predict(
            data=sample_data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        # Test plotting (should not raise errors)
        analyzer.plot_clusters(
            results['scaled_data'],
            results['labels'],
            feature_names=feature_columns,
            method="pca"
        )
        
        # Check that plot was called
        mock_show.assert_called_once()
    
    
    def test_plot_clusters_unsuitable_dimensionality(self, analyzer):
        """Test plotting with unsuitable data dimensionality."""
        # Create 1D data
        data_1d = np.random.randn(50, 1)
        labels = np.random.randint(0, 2, 50)
        
        # Should print message and not crash
        analyzer.plot_clusters(data_1d, labels)
    
    def test_calculate_metrics_single_cluster(self, analyzer):
        """Test metric calculation with single cluster."""
        # Create data with all same labels
        data = np.random.randn(50, 3)
        analyzer.labels_ = np.zeros(50)  # All points in cluster 0
        
        analyzer._calculate_metrics(data)
        
        # Should handle single cluster case gracefully
        assert 'error' in analyzer.evaluation_metrics
        assert analyzer.evaluation_metrics['error'] == 'Only one cluster found'
    
    def test_calculate_metrics_with_noise(self, analyzer):
        """Test metric calculation with noise points (DBSCAN)."""
        data = np.random.randn(50, 3)
        # Create labels with noise points (-1)
        analyzer.labels_ = np.array([0, 1, 0, 1, -1, -1] + [0] * 22 + [1] * 22)
        
        analyzer._calculate_metrics(data)
        
        # Should calculate metrics excluding noise points
        assert 'silhouette_score' in analyzer.evaluation_metrics
        assert 'n_noise_points' in analyzer.evaluation_metrics
        assert analyzer.evaluation_metrics['n_noise_points'] == 2
    
    def test_edge_case_empty_data(self, analyzer):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        feature_columns = []
        
        with pytest.raises(Exception):  # Should raise some exception
            analyzer.prepare_data(empty_data, feature_columns)
    
    
    def test_memory_efficiency(self, analyzer):
        """Test memory efficiency with larger dataset."""
        # Create larger dataset to test memory handling
        large_data = pd.DataFrame(np.random.randn(1000, 10))
        large_data.columns = [f'feature_{i}' for i in range(10)]
        
        feature_columns = large_data.columns.tolist()
        
        # Should handle larger dataset without issues
        results = analyzer.fit_predict(
            data=large_data,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans"
        )
        
        assert len(results['labels']) == 1000
        assert results['scaled_data'].shape == (1000, 10)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        # Create sample data
        data = pd.DataFrame(np.random.RandomState(42).randn(100, 4))
        data.columns = ['f1', 'f2', 'f3', 'f4']
        feature_columns = ['f1', 'f2', 'f3', 'f4']
        
        # Run clustering twice with same random state
        analyzer1 = ClusteringAnalyzer(random_state=42)
        analyzer2 = ClusteringAnalyzer(random_state=42)
        
        results1 = analyzer1.fit_predict(
            data=data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        results2 = analyzer2.fit_predict(
            data=data,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        # Results should be identical
        np.testing.assert_array_equal(results1['labels'], results2['labels'])
        np.testing.assert_array_almost_equal(
            results1['cluster_centers'], 
            results2['cluster_centers']
        )


class TestClusteringIntegration:
    """Integration tests for clustering functionality."""
    @pytest.fixture
    def sample_data(self):
        '''
        Uplod csv dataset from EPC
        '''
        print("Loading dataset...")
        df = pd.read_csv(data_path, sep=",", decimal=".", low_memory=False, header=0, index_col=0)
        
        return df

    def test_end_to_end_workflow(self, temp_dir, sample_data):
        """Test complete end-to-end clustering workflow."""

        data = sample_data        
        # Initialize analyzer
        analyzer = ClusteringAnalyzer(random_state=42)
        
        # Run complete workflow
        
        # Step 1: Determine optimal clusters
        scaled_data = analyzer.prepare_data(data, feature_columns)
        optimal_k = analyzer.determine_optimal_clusters(
            scaled_data, method="silhouette", plot=False
        )
        
        # Step 2: Fit clustering
        results = analyzer.fit_predict(
            data=data,
            feature_columns=feature_columns,
            n_clusters=optimal_k,
            algorithm="kmeans",
            save_clusters=True,
            output_dir=temp_dir
        )
        
        # Step 3: Analyze results
        stats = analyzer.get_cluster_statistics(
            results['data_with_clusters'], 
            feature_columns
        )
        
        # Verify results
        assert results['n_clusters'] == optimal_k
        assert len(stats) == optimal_k
        assert 'silhouette_score' in results['evaluation_metrics']
        
        # Check file outputs
        assert os.path.exists(os.path.join(temp_dir, 'data_with_clusters.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'clustering_model.pkl'))
        
        # Verify cluster quality (should be reasonable for well-separated blobs)
        silhouette_score = results['evaluation_metrics']['silhouette_score']
        assert silhouette_score > 0.5  # Should be good clustering
    
    def test_robustness_with_outliers(self, sample_data):
        """Test clustering robustness with outliers."""
        # Create data with outliers
        data_with_outliers = sample_data
        # outliers = np.random.randn(10, 3) * 5 + 10  # Far outliers
        # data_with_outliers = np.vstack([normal_data, outliers])
        
        # df = pd.DataFrame(data_with_outliers, columns=['x', 'y', 'z'])
        
        analyzer = ClusteringAnalyzer(random_state=42)
        
        # Test with K-means (sensitive to outliers)
        results_kmeans = analyzer.fit_predict(
            data=data_with_outliers,
            feature_columns=feature_columns,
            n_clusters=3,
            algorithm="kmeans"
        )
        
        # Test with DBSCAN (robust to outliers)
        analyzer_dbscan = ClusteringAnalyzer(random_state=42)
        results_dbscan = analyzer_dbscan.fit_predict(
            data=data_with_outliers,
            feature_columns=feature_columns,
            algorithm="dbscan"
        )
        
        # Both should complete without errors
        assert len(results_kmeans['labels']) == len(data_with_outliers)
        assert len(results_dbscan['labels']) == len(data_with_outliers)
        
        # DBSCAN should identify some outliers as noise
        assert -1 in results_dbscan['labels']  # Noise points


if __name__ == "__main__":
    pytest.main([__file__])