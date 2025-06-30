"""
Clustering Analysis Module

This module provides clustering functionality for building energy performance data
using K-means and other clustering algorithms with automatic cluster number determination.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Union
import warnings
import os
from pathlib import Path
import joblib


class ClusteringAnalyzer:
    """
    A comprehensive clustering analyzer for building energy performance data.
    
    This class provides methods for clustering data using various algorithms,
    determining optimal cluster numbers, and evaluating cluster quality.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the clustering analyzer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        self.scaled_data = None
        self.cluster_centers_ = None
        self.evaluation_metrics = {}
        
    def prepare_data(self, data: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Prepare and scale data for clustering.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_columns : List[str]
            Columns to use for clustering
            
        Returns
        -------
        np.ndarray
            Scaled feature matrix
        """
        # Select features and handle missing values
        features = data[feature_columns].copy()
        
        # Handle missing values
        if features.isnull().any().any():
            print("Warning: Missing values detected. Filling with median values.")
            features = features.fillna(features.median())
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(features)
        
        return self.scaled_data
    
    def determine_optimal_clusters(
        self, 
        data: np.ndarray, 
        method: str = "elbow",
        k_range: Tuple[int, int] = (2, 15),
        plot: bool = True
    ) -> int:
        """
        Determine optimal number of clusters using various methods.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled data for clustering
        method : str, optional
            Method to use ('elbow', 'silhouette'), by default "elbow"
        k_range : Tuple[int, int], optional
            Range of k values to test, by default (2, 15)
        plot : bool, optional
            Whether to plot the results, by default True
            
        Returns
        -------
        int
            Optimal number of clusters
        """
        k_min, k_max = k_range
        k_values = range(k_min, k_max + 1)
        
        if method == "elbow":
            return self._elbow_method(data, k_values, plot)
        elif method == "silhouette":
            return self._silhouette_method(data, k_values, plot)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'elbow', 'silhouette'")
    
    def _elbow_method(self, data: np.ndarray, k_values: range, plot: bool = True) -> int:
        """Determine optimal clusters using elbow method."""
        inertias = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Calculate rate of change
        rate_of_change = []
        for i in range(1, len(inertias)):
            rate_of_change.append(inertias[i-1] - inertias[i])
        
        # Find elbow point (maximum rate of change decrease)
        elbow_point = 2  # Start with minimum k
        max_decrease = 0
        
        for i in range(1, len(rate_of_change)):
            decrease = rate_of_change[i-1] - rate_of_change[i]
            if decrease > max_decrease:
                max_decrease = decrease
                elbow_point = k_values[i+1]
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(k_values, inertias, 'bo-')
            plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Optimal k={elbow_point}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(k_values[1:], rate_of_change, 'go-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Rate of Change in Inertia')
            plt.title('Rate of Change')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        return elbow_point
    
    def _silhouette_method(self, data: np.ndarray, k_values: range, plot: bool = True) -> int:
        """Determine optimal clusters using silhouette analysis."""
        silhouette_scores = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_values[np.argmax(silhouette_scores)]
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(k_values, silhouette_scores, 'bo-')
            plt.axvline(x=optimal_k, color='red', linestyle='--', 
                       label=f'Optimal k={optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return optimal_k
    
    def _calinski_harabasz_method(self, data: np.ndarray, k_values: range, plot: bool = True) -> int:
        """Determine optimal clusters using Calinski-Harabasz index."""
        ch_scores = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            ch_score = calinski_harabasz_score(data, cluster_labels)
            ch_scores.append(ch_score)
        
        optimal_k = k_values[np.argmax(ch_scores)]
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(k_values, ch_scores, 'bo-')
            plt.axvline(x=optimal_k, color='red', linestyle='--', 
                       label=f'Optimal k={optimal_k}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski-Harabasz Score')
            plt.title('Calinski-Harabasz Analysis')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return optimal_k
    
    def fit_kmeans(
        self, 
        data: np.ndarray, 
        n_clusters: int, 
        **kwargs
    ) -> 'ClusteringAnalyzer':
        """
        Fit K-means clustering model.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled data for clustering
        n_clusters : int
            Number of clusters
        **kwargs
            Additional arguments for KMeans
            
        Returns
        -------
        ClusteringAnalyzer
            Self for method chaining
        """
        self.model = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state,
            n_init=10,
            **kwargs
        )
        
        self.labels_ = self.model.fit_predict(data)
        self.cluster_centers_ = self.model.cluster_centers_
        
        # Calculate evaluation metrics
        self._calculate_metrics(data)
        
        return self
    
    def fit_dbscan(
        self, 
        data: np.ndarray, 
        eps: float = 0.5, 
        min_samples: int = 5,
        **kwargs
    ) -> 'ClusteringAnalyzer':
        """
        Fit DBSCAN clustering model.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled data for clustering
        eps : float, optional
            Maximum distance between samples, by default 0.5
        min_samples : int, optional
            Minimum number of samples in a neighborhood, by default 5
        **kwargs
            Additional arguments for DBSCAN
            
        Returns
        -------
        ClusteringAnalyzer
            Self for method chaining
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.labels_ = self.model.fit_predict(data)
        
        # Calculate evaluation metrics
        self._calculate_metrics(data)
        
        return self
    
    def fit_hierarchical(
        self, 
        data: np.ndarray, 
        n_clusters: int,
        linkage: str = 'ward',
        **kwargs
    ) -> 'ClusteringAnalyzer':
        """
        Fit Agglomerative (Hierarchical) clustering model.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled data for clustering
        n_clusters : int
            Number of clusters
        linkage : str, optional
            Linkage criterion, by default 'ward'
        **kwargs
            Additional arguments for AgglomerativeClustering
            
        Returns
        -------
        ClusteringAnalyzer
            Self for method chaining
        """
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        
        self.labels_ = self.model.fit_predict(data)
        
        # Calculate evaluation metrics
        self._calculate_metrics(data)
        
        return self
    
    def _calculate_metrics(self, data: np.ndarray):
        """Calculate clustering evaluation metrics."""
        if len(np.unique(self.labels_)) > 1:
            # Filter out noise points for DBSCAN
            valid_mask = self.labels_ != -1
            if np.sum(valid_mask) > 1:
                valid_data = data[valid_mask]
                valid_labels = self.labels_[valid_mask]
                
                if len(np.unique(valid_labels)) > 1:
                    self.evaluation_metrics = {
                        'silhouette_score': silhouette_score(valid_data, valid_labels),
                        'calinski_harabasz_score': calinski_harabasz_score(valid_data, valid_labels),
                        'davies_bouldin_score': davies_bouldin_score(valid_data, valid_labels),
                        'n_clusters': len(np.unique(valid_labels)),
                        'n_noise_points': np.sum(self.labels_ == -1)
                    }
                else:
                    self.evaluation_metrics = {'error': 'Only one cluster found'}
            else:
                self.evaluation_metrics = {'error': 'No valid clusters found'}
        else:
            self.evaluation_metrics = {'error': 'Only one cluster found'}
    
    def fit_predict(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        method: str = "elbow",
        n_clusters: Optional[int] = None,
        algorithm: str = "kmeans",
        save_clusters: bool = False,
        output_dir: str = "data/clusters"
    ) -> Dict:
        """
        Complete clustering pipeline: prepare data, determine clusters, and fit model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_columns : List[str]
            Columns to use for clustering
        method : str, optional
            Method for determining optimal clusters, by default "elbow"
        n_clusters : Optional[int], optional
            Fixed number of clusters (overrides automatic determination), by default None
        algorithm : str, optional
            Clustering algorithm to use, by default "kmeans"
        save_clusters : bool, optional
            Whether to save cluster results, by default False
        output_dir : str, optional
            Directory to save cluster results, by default "data/clusters"
            
        Returns
        -------
        Dict
            Dictionary containing cluster results and metadata
        """
        # Prepare data
        scaled_data = self.prepare_data(data, feature_columns)
        
        # Determine optimal clusters if not provided
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(scaled_data, method=method)
            print(f"Optimal number of clusters determined: {n_clusters}")
        
        # Fit clustering model
        if algorithm == "kmeans":
            self.fit_kmeans(scaled_data, n_clusters)
        elif algorithm == "dbscan":
            # For DBSCAN, we need to determine eps and min_samples
            self.fit_dbscan(scaled_data)
        elif algorithm == "hierarchical":
            self.fit_hierarchical(scaled_data, n_clusters)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create results dictionary
        results = {
            'labels': self.labels_,
            'n_clusters': len(np.unique(self.labels_[self.labels_ != -1])),
            'cluster_centers': self.cluster_centers_ if hasattr(self, 'cluster_centers_') else None,
            'evaluation_metrics': self.evaluation_metrics,
            'feature_columns': feature_columns,
            'algorithm': algorithm,
            'scaled_data': scaled_data
        }
        
        # Add cluster labels to original data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = self.labels_
        results['data_with_clusters'] = data_with_clusters
        
        # Save clusters if requested
        if save_clusters:
            self._save_clusters(data_with_clusters, output_dir, feature_columns)
        
        return results
    
    def _save_clusters(self, data_with_clusters: pd.DataFrame, output_dir: str, feature_columns: List[str]):
        """Save cluster results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete dataset with clusters
        data_with_clusters.to_csv(
            os.path.join(output_dir, 'data_with_clusters.csv'), 
            index=False
        )
        
        # Save individual cluster datasets
        unique_clusters = data_with_clusters['cluster'].unique()
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise points
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                cluster_data.to_csv(
                    os.path.join(output_dir, f'cluster_{cluster_id}.csv'),
                    index=False
                )
        
        # Save clustering model and metadata
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'evaluation_metrics': self.evaluation_metrics,
            'feature_columns': feature_columns
        }
        
        joblib.dump(model_data, os.path.join(output_dir, 'clustering_model.pkl'))
        
        print(f"Cluster results saved to: {output_dir}")
    
    def plot_clusters(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        feature_names: List[str] = None,
        method: str = "pca"
    ):
        """
        Plot clustering results.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled data
        labels : np.ndarray
            Cluster labels
        feature_names : List[str], optional
            Names of features, by default None
        method : str, optional
            Dimensionality reduction method for visualization, by default "pca"
        """
        if method == "pca" and data.shape[1] > 2:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2, random_state=self.random_state)
            data_2d = pca.fit_transform(data)
            explained_variance = pca.explained_variance_ratio_
            
            plt.figure(figsize=(12, 5))
            
            # Plot clusters
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.xlabel(f'First Principal Component ({explained_variance[0]:.2%} variance)')
            plt.ylabel(f'Second Principal Component ({explained_variance[1]:.2%} variance)')
            plt.title('Clustering Results (PCA Projection)')
            plt.colorbar(scatter)
            
            # Plot cluster centers if available
            if self.cluster_centers_ is not None:
                centers_2d = pca.transform(self.cluster_centers_)
                plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                          marker='x', s=200, linewidths=3, color='red', label='Centroids')
                plt.legend()
            
            # Plot explained variance
            plt.subplot(1, 2, 2)
            plt.bar(range(1, len(explained_variance) + 1), explained_variance)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            
            plt.tight_layout()
            plt.show()
            
        elif data.shape[1] == 2:
            # Direct 2D plot
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
            
            if feature_names:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            
            plt.title('Clustering Results')
            plt.colorbar(scatter)
            
            if self.cluster_centers_ is not None:
                plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], 
                          marker='x', s=200, linewidths=3, color='red', label='Centroids')
                plt.legend()
            
            plt.show()
        
        else:
            print("Cannot plot clusters: data dimensionality not suitable for visualization")
    
    def get_cluster_statistics(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with cluster labels
        feature_columns : List[str]
            Feature columns to analyze
            
        Returns
        -------
        pd.DataFrame
            Statistics for each cluster
        """
        if 'cluster' not in data.columns:
            raise ValueError("Data must contain 'cluster' column")
        
        cluster_stats = []
        
        for cluster_id in sorted(data['cluster'].unique()):
            if cluster_id != -1:  # Skip noise points
                cluster_data = data[data['cluster'] == cluster_id]
                
                stats = {
                    'cluster_id': cluster_id,
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(data) * 100
                }
                
                for col in feature_columns:
                    stats[f'{col}_mean'] = cluster_data[col].mean()
                    stats[f'{col}_std'] = cluster_data[col].std()
                    stats[f'{col}_min'] = cluster_data[col].min()
                    stats[f'{col}_max'] = cluster_data[col].max()
                
                cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def predict(self, new_data: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to predict
        feature_columns : List[str]
            Feature columns (must match training features)
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        # Prepare new data
        features = new_data[feature_columns]
        scaled_features = self.scaler.transform(features)
        
        # Predict clusters
        if hasattr(self.model, 'predict'):
            return self.model.predict(scaled_features)
        else:
            # For DBSCAN, we need to use a different approach
            raise NotImplementedError("Prediction not supported for this clustering algorithm")