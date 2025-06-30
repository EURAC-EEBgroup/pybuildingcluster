"""
Regression Modeling Module

This module provides regression modeling functionality for building energy performance
prediction using Random Forest, XGBoost, and LightGBM with automatic model selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge,  LassoCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import (
    train_test_split, cross_val_score,
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
)
from sklearn.preprocessing import StandardScaler

# Handle optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RegressionModelBuilder:
    """
    Enhanced regression model builder with aggressive overfitting prevention.
    
    This class implements multiple strategies to prevent overfitting:
    1. Conservative feature selection
    2. Strict validation criteria
    3. Learning curve analysis
    4. Regularization emphasis
    5. Cross-validation based model selection
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, problem_type='regression'):
        """
        Initialize the anti-overfitting model builder.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        problem_type : str, optional
            Type of problem ('regression'), by default 'regression'
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.best_models = {}
        self.problem_type = problem_type
        self.selected_features = {}  # Store selected features per cluster
        
    def prepare_data_with_feature_selection(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str],
        test_size: float = 0.3,  # Larger test set for better validation
        scale_features: bool = True,
        max_features_ratio: float = 0.3,  # Maximum 30% of available features
        min_features: int = 3,
        max_features_absolute: int = 15,  # Hard limit on features
        user_features: Optional[List[str]] = None  # User-specified features
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data with conservative feature selection to prevent overfitting.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_column : str
            Name of the target column
        feature_columns : List[str]
            List of feature column names
        test_size : float, optional
            Proportion of data for testing, by default 0.3
        scale_features : bool, optional
            Whether to scale features, by default True
        max_features_ratio : float, optional
            Maximum ratio of features to samples, by default 0.3
        min_features : int, optional
            Minimum number of features to select, by default 3
        max_features_absolute : int, optional
            Absolute maximum number of features, by default 15
        user_features : Optional[List[str]], optional
            User-specified features to include in selection, by default None
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
            X_train, X_test, y_train, y_test, selected_feature_names
        """
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Check if feature columns exist
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        
        # Extract features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            print("Warning: Missing values in features. Filling with median values.")
            X = X.fillna(X.median(numeric_only=True))
        
        if y.isnull().any():
            print("Warning: Missing values in target. Dropping corresponding rows.")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Calculate maximum features based on sample size and constraints
        n_samples = len(X)
        max_features_by_ratio = max(min_features, int(n_samples * max_features_ratio))
        max_features = min(max_features_by_ratio, max_features_absolute, len(feature_columns))
        
        print(f"Feature Selection Strategy:")
        print(f"  Total available features: {len(feature_columns)}")
        print(f"  Sample size: {n_samples}")
        print(f"  Max features by ratio (30%): {max_features_by_ratio}")
        print(f"  Max features (absolute limit): {max_features_absolute}")
        print(f"  Selected max features: {max_features}")
        
        # Split data first to avoid data leakage in feature selection
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Handle user features
        if user_features:
            # Validate user features exist in dataset
            valid_user_features = [f for f in user_features if f in X_train_full.columns]
            if valid_user_features:
                print(f"  User-specified features provided: {len(valid_user_features)} valid features")
                selected_features = self._select_best_features(
                    X_train_full, y_train, max_features, min_features
                ) + valid_user_features
                selected_features = list(set(selected_features))
            else:
                print("  Warning: No valid user features found, using automated selection")
                selected_features = self._select_best_features(
                    X_train_full, y_train, max_features, min_features
                )
        else:
            # Perform automated feature selection
            selected_features = self._select_best_features(
                X_train_full, y_train, max_features, min_features
            )
        
        print(f"  Selected {len(selected_features)} features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        
        # Apply feature selection
        X_train = X_train_full[selected_features]
        X_test = X_test_full[selected_features]
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler for later use
            self.scaler = scaler
            
            return X_train_scaled, X_test_scaled, y_train.values, y_test.values, selected_features
        else:
            self.scaler = None
            return X_train.values, X_test.values, y_train.values, y_test.values, selected_features
    
    def _select_best_features(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        max_features: int, 
        min_features: int
    ) -> List[str]:
        """
        Select the best features using multiple methods and conservative criteria.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        max_features : int
            Maximum number of features to select
        min_features : int
            Minimum number of features to select
            
        Returns
        -------
        List[str]
            List of selected feature names
        """
        
        print("  Performing conservative feature selection...")
        
        # Method 1: Statistical significance (Univariate)
        selector_stats = SelectKBest(score_func=f_regression, k=max_features)
        selector_stats.fit(X_train, y_train)
        features_stats = X_train.columns[selector_stats.get_support()].tolist()
        scores_stats = selector_stats.scores_
        
        # Method 2: Random Forest Feature Importance
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        rf.fit(X_train, y_train)
        importance_rf = rf.feature_importances_
        
        # Get top features by importance
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance_rf
        }).sort_values('importance', ascending=False)
        features_rf = feature_importance_df.head(max_features)['feature'].tolist()
        
        # Method 3: Lasso for automatic feature selection
        try:
            lasso_cv = LassoCV(
                cv=5, 
                random_state=self.random_state,
                max_iter=2000,
                n_jobs=self.n_jobs
            )
            lasso_cv.fit(X_train, y_train)
            lasso_coef = np.abs(lasso_cv.coef_)
            
            # Select features with non-zero coefficients
            features_lasso = X_train.columns[lasso_coef > 0].tolist()
            
            # If too many features, take top ones by coefficient magnitude
            if len(features_lasso) > max_features:
                lasso_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'coef': lasso_coef
                }).sort_values('coef', ascending=False)
                features_lasso = lasso_importance.head(max_features)['feature'].tolist()
        
        except Exception as e:
            print(f"    Warning: Lasso selection failed: {e}")
            features_lasso = []
        
        # Method 4: Recursive Feature Elimination with Cross-Validation
        try:
            # Use a simple model for RFE to avoid overfitting
            
            estimator = Ridge(random_state=self.random_state)
            
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=5,
                scoring='r2',
                min_features_to_select=min_features,
                n_jobs=self.n_jobs
            )
            rfecv.fit(X_train, y_train)
            features_rfe = X_train.columns[rfecv.support_].tolist()
            
            # Limit to max_features
            if len(features_rfe) > max_features:
                features_rfe = features_rfe[:max_features]
                
        except Exception as e:
            print(f"    Warning: RFE selection failed: {e}")
            features_rfe = []
        
        # Combine results from all methods
        all_selected_features = set()
        method_counts = {}
        
        for feature_list, method_name in [
            (features_stats, 'stats'),
            (features_rf, 'rf'),
            (features_lasso, 'lasso'),
            (features_rfe, 'rfe')
        ]:
            if feature_list:  # Only if method succeeded
                all_selected_features.update(feature_list)
                for feature in feature_list:
                    method_counts[feature] = method_counts.get(feature, 0) + 1
        
        # Select features that appear in multiple methods (consensus approach)
        consensus_features = [
            feature for feature, count in method_counts.items() 
            if count >= 2  # Feature must be selected by at least 2 methods
        ]
        
        # If not enough consensus features, add top features from individual methods
        if len(consensus_features) < min_features:
            # Add top RF features
            for feature in features_rf:
                if feature not in consensus_features:
                    consensus_features.append(feature)
                    if len(consensus_features) >= min_features:
                        break
        
        # Limit to max_features
        if len(consensus_features) > max_features:
            # Prioritize by method count, then by RF importance
            feature_priority = []
            for feature in consensus_features:
                count = method_counts.get(feature, 0)
                rf_importance = importance_rf[X_train.columns.get_loc(feature)]
                feature_priority.append((feature, count, rf_importance))
            
            # Sort by count (descending), then by importance (descending)
            feature_priority.sort(key=lambda x: (x[1], x[2]), reverse=True)
            consensus_features = [f[0] for f in feature_priority[:max_features]]
        
        print(f"    Feature selection summary:")
        print(f"    - Statistical: {len(features_stats)} features")
        print(f"    - Random Forest: {len(features_rf)} features")
        print(f"    - Lasso: {len(features_lasso)} features")
        print(f"    - RFE: {len(features_rfe)} features")
        print(f"    - Consensus (≥2 methods): {len(consensus_features)} features")
        
        return consensus_features
    
    def get_conservative_model_configurations(self) -> Dict[str, Dict]:
        """
        Get conservative model configurations designed to prevent overfitting.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary of conservative model configurations
        """
        
        
        configs = {
            # Random Forest with conservative parameters
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,  # Moderate number of trees
                    max_depth=8,       # Limited depth
                    min_samples_split=10,  # Require more samples to split
                    min_samples_leaf=5,    # Require more samples in leaves
                    max_features=0.7,      # Use subset of features
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                ),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 8, 10],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [3, 5, 8],
                    'max_features': [0.5, 0.7, 0.9]
                }
            },
            
        }
        
        # Add XGBoost if available
        try:
            import xgboost as xgb
            configs['xgboost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,      # Shallow trees
                    learning_rate=0.05,  # Slow learning
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,    # L1 regularization
                    reg_lambda=1.0,   # L2 regularization
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'reg_alpha': [0.5, 1.0, 2.0],
                    'reg_lambda': [0.5, 1.0, 2.0]
                }
            }
        except ImportError:
            pass
        
        # Add LightGBM if available
        try:
            import lightgbm as lgb
            configs['lightgbm'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    num_leaves=15,    # Few leaves
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbosity=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [10, 15, 20],
                    'reg_alpha': [0.5, 1.0, 2.0],
                    'reg_lambda': [0.5, 1.0, 2.0]
                }
            }
        except ImportError:
            pass
        
        return configs
    
    def validate_model_robustness(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Validate model robustness with strict overfitting detection.
        
        Parameters
        ----------
        model : sklearn model
            Trained model
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
        cv_folds : int, optional
            Number of CV folds, by default 5
            
        Returns
        -------
        Dict[str, float]
            Robustness metrics
        """
        # Training performance
        train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
        
        # Test performance
        test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Cross-validation performance
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        r2_cv_mean = cv_scores.mean()
        r2_cv_std = cv_scores.std()
        
        # Calculate overfitting indicators
        r2_diff = r2_train - r2_test
        r2_cv_diff = r2_train - r2_cv_mean
        
        # Stability metric: how much CV varies
        cv_stability = 1 - (r2_cv_std / max(abs(r2_cv_mean), 0.01))
        
        return {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'r2_cv_mean': r2_cv_mean,
            'r2_cv_std': r2_cv_std,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_diff_train_test': r2_diff,
            'r2_diff_train_cv': r2_cv_diff,
            'cv_stability': cv_stability
        }
    
    def select_best_model_conservative(
        self,
        model_results: Dict[str, Dict],
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        Select best model using conservative criteria to prevent overfitting.
        
        Parameters
        ----------
        model_results : Dict[str, Dict]
            Results from multiple models
        verbose : bool, optional
            Whether to print selection details, by default True
            
        Returns
        -------
        Tuple[str, Dict]
            Best model name and its results
        """
        if verbose:
            print("  Applying conservative model selection criteria...")
        
        best_model_name = None
        best_score = -np.inf
        best_result = None
        
        # Define conservative thresholds
        max_r2_diff = 0.15        # Max difference between train and test R²
        min_r2_test = 0.6         # Minimum acceptable test R²
        min_cv_stability = 0.7    # Minimum CV stability
        max_cv_std = 0.2          # Maximum CV standard deviation
        
        eligible_models = []
        
        for model_name, result in model_results.items():
            metrics = result['robustness_metrics']
            
            # Check conservative criteria
            criteria_met = {
                'r2_diff': metrics['r2_diff_train_test'] <= max_r2_diff,
                'r2_test': metrics['r2_test'] >= min_r2_test,
                'cv_stability': metrics['cv_stability'] >= min_cv_stability,
                'cv_std': metrics['r2_cv_std'] <= max_cv_std
            }
            
            all_criteria_met = all(criteria_met.values())
            
            if verbose:
                print(f"    {model_name}:")
                print(f"      R² Train: {metrics['r2_train']:.3f}")
                print(f"      R² Test: {metrics['r2_test']:.3f}")
                print(f"      R² Diff: {metrics['r2_diff_train_test']:.3f} ({'✓' if criteria_met['r2_diff'] else '✗'})")
                print(f"      CV Mean: {metrics['r2_cv_mean']:.3f}")
                print(f"      CV Std: {metrics['r2_cv_std']:.3f} ({'✓' if criteria_met['cv_std'] else '✗'})")
                print(f"      CV Stability: {metrics['cv_stability']:.3f} ({'✓' if criteria_met['cv_stability'] else '✗'})")
                print(f"      Conservative: {'✓' if all_criteria_met else '✗'}")
            
            if all_criteria_met:
                eligible_models.append((model_name, result, metrics))
        
        if eligible_models:
            # Among eligible models, select based on composite score
            for model_name, result, metrics in eligible_models:
                # Composite score: prioritize test performance and stability
                composite_score = (
                    0.4 * metrics['r2_test'] +           # 40% test R²
                    0.3 * metrics['r2_cv_mean'] +        # 30% CV mean
                    0.2 * metrics['cv_stability'] +      # 20% stability
                    0.1 * (1 - metrics['r2_cv_std'])     # 10% low variance
                )
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_name = model_name
                    best_result = result
        
        else:
            # No model meets conservative criteria, select least overfitted
            if verbose:
                print("    ⚠️ No model meets conservative criteria. Selecting least overfitted.")
            
            min_overfit = np.inf
            for model_name, result in model_results.items():
                metrics = result['robustness_metrics']
                overfit_score = metrics['r2_diff_train_test'] + metrics['r2_cv_std']
                
                if overfit_score < min_overfit and metrics['r2_test'] > 0:
                    min_overfit = overfit_score
                    best_model_name = model_name
                    best_result = result
        
        if verbose and best_model_name:
            metrics = best_result['robustness_metrics']
            print(f"  🏆 Selected: {best_model_name}")
            print(f"    Final R² Test: {metrics['r2_test']:.3f}")
            print(f"    Overfitting Risk: {'Low' if metrics['r2_diff_train_test'] <= 0.1 else 'Medium' if metrics['r2_diff_train_test'] <= 0.2 else 'High'}")
        
        return best_model_name, best_result
    
    def build_models(
        self,
        data: pd.DataFrame,
        clusters: Dict,
        target_column: str,
        feature_columns: List[str],
        models_to_train: Optional[List[str]] = None,
        hyperparameter_tuning: str = "grid",  # Use grid search for conservative tuning
        models_dir: str = "models",
        save_models: bool = False,
        user_features: Optional[List[str]] = None  # User-specified features
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build regression models with aggressive overfitting prevention.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with cluster labels
        clusters : Dict
            Clustering results dictionary
        target_column : str
            Name of the target column
        feature_columns : List[str]
            List of feature column names
        models_to_train : Optional[List[str]], optional
            List of models to train, by default None (trains conservative models)
        hyperparameter_tuning : str, optional
            Type of hyperparameter tuning, by default "grid"
        models_dir : str, optional
            Directory to save models, by default "models"
        save_models : bool, optional
            Whether to save trained models, by default False
        user_features : Optional[List[str]], optional
            User-specified features to use instead of automated selection, by default None
            
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary mapping cluster IDs to model results
        """
        # Get conservative model configurations
        available_models = list(self.get_conservative_model_configurations().keys())
        
        if models_to_train is None:
            models_to_train = available_models
        else:
            # Filter out unavailable models
            unavailable_models = [m for m in models_to_train if m not in available_models]
            if unavailable_models:
                warnings.warn(f"Models not available: {unavailable_models}. Available models: {available_models}")
                models_to_train = [m for m in models_to_train if m in available_models]
        
        if not models_to_train:
            raise ValueError(f"No available models to train. Available models: {available_models}")

        # Get data with clusters
        if 'data_with_clusters' in clusters:
            clustered_data = clusters['data_with_clusters']
        else:
            clustered_data = data.copy()
            if 'labels' in clusters:
                clustered_data['cluster'] = clusters['labels']
            elif 'cluster' not in clustered_data.columns:
                raise ValueError("No cluster information found in data or clusters dictionary")

        # Validate inputs
        if target_column not in clustered_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        missing_features = [col for col in feature_columns if col not in clustered_data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")

        # Check if there are any valid rows
        if clustered_data.empty:
            raise ValueError("Input data is empty")

        # Build models for each cluster
        cluster_models = {}
        unique_clusters = sorted(clustered_data['cluster'].unique())

        print(f"Building ANTI-OVERFITTING regression models for {len(unique_clusters)} clusters...")
        print(f"Models to train: {models_to_train}")
        print(f"Conservative approach: Strong regularization + Limited features + Strict validation")
        if user_features:
            print(f"User-specified features: {user_features}")
        print("-" * 80)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points from DBSCAN
                print(f"Skipping noise cluster (ID: -1)")
                continue
                
            print(f"\n🔍 Processing Cluster {cluster_id}")
            print("-" * 50)
            
            # Get cluster data
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id].copy()
            
            # More stringent minimum sample size for reliable validation
            min_samples_required = 30  # Increased for better validation
            if len(cluster_data) < min_samples_required:
                print(f"❌ Cluster {cluster_id} has only {len(cluster_data)} samples.")
                print(f"Minimum required: {min_samples_required}. Skipping this cluster.")
                continue
            
            # Check for target variance
            target_values = cluster_data[target_column].dropna()
            if len(target_values) == 0:
                print(f"❌ No valid target values in cluster {cluster_id}. Skipping.")
                continue
            
            if target_values.std() == 0:
                print(f"❌ No variance in target column for cluster {cluster_id}. Skipping.")
                continue
            
            print(f"✅ Cluster size: {len(cluster_data)} samples")
            print(f"📊 Target statistics: mean={target_values.mean():.3f}, std={target_values.std():.3f}")
            
            # Prepare data with conservative feature selection
            try:
                X_train, X_test, y_train, y_test, selected_features = self.prepare_data_with_feature_selection(
                    cluster_data, target_column, feature_columns,
                    test_size=0.3,  # Larger test set
                    scale_features=False,
                    max_features_ratio=0.25,  # Conservative feature ratio
                    min_features=3,
                    max_features_absolute=12,  # Lower absolute limit
                    user_features=user_features  # Pass user features
                )
                
                # Store selected features for this cluster
                self.selected_features[cluster_id] = selected_features
                
                print(f"📈 Training set: {len(X_train)} samples, {len(selected_features)} features")
                print(f"📉 Test set: {len(X_test)} samples")
                
            except Exception as e:
                print(f"❌ Error preparing data for cluster {cluster_id}: {str(e)}")
                continue
            
            # Train models for this cluster
            model_results = {}
            
            for model_name in models_to_train:
                print(f"\n  🤖 Training {model_name}...")
                
                try:
                    # Get model configuration
                    configs = self.get_conservative_model_configurations()
                    config = configs[model_name]
                    base_model = config['model']
                    
                    # Perform hyperparameter tuning with conservative approach
                    if hyperparameter_tuning == "grid":
                        
                        search = GridSearchCV(
                            base_model,
                            config['params'],
                            cv=5,  # Conservative CV
                            scoring='r2',
                            n_jobs=self.n_jobs,
                            verbose=0
                        )
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                    
                    elif hyperparameter_tuning == "randomized":
                        
                        search = RandomizedSearchCV(
                            base_model,
                            config['params'],
                            n_iter=20,  # Limited iterations to prevent overfitting to CV
                            cv=5,
                            scoring='r2',
                            n_jobs=self.n_jobs,
                            random_state=self.random_state,
                            verbose=0
                        )
                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                    
                    else:  # No hyperparameter tuning
                        best_model = base_model
                        best_model.fit(X_train, y_train)
                        best_params = {}
                    
                    # Validate model robustness
                    robustness_metrics = self.validate_model_robustness(
                        best_model, X_train, y_train, X_test, y_test, cv_folds=5
                    )
                    
                    # Calculate additional metrics
                    train_pred = best_model.predict(X_train)
                    test_pred = best_model.predict(X_test)
                    
                    train_metrics = {
                        'r2': robustness_metrics['r2_train'],
                        'rmse': robustness_metrics['rmse_train'],
                        'mae': mean_absolute_error(y_train, train_pred)
                    }
                    
                    test_metrics = {
                        'r2': robustness_metrics['r2_test'],
                        'rmse': robustness_metrics['rmse_test'],
                        'mae': mean_absolute_error(y_test, test_pred)
                    }
                    
                    # Store model results
                    model_results[model_name] = {
                        'model': best_model,
                        'model_name': model_name,
                        'best_params': best_params,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'robustness_metrics': robustness_metrics,
                        'selected_features': selected_features,
                        'train_predictions': train_pred,
                        'test_predictions': test_pred
                    }
                    
                    # Print performance with overfitting indicators
                    print(f"    ✅ {model_name} completed:")
                    print(f"      R² Train: {train_metrics['r2']:.4f}")
                    print(f"      R² Test: {test_metrics['r2']:.4f}")
                    print(f"      R² Diff: {robustness_metrics['r2_diff_train_test']:.4f}")
                    print(f"      CV Mean: {robustness_metrics['r2_cv_mean']:.4f} ± {robustness_metrics['r2_cv_std']:.4f}")
                    print(f"      RMSE Test: {test_metrics['rmse']:.4f}")
                    print(f"      CV Stability: {robustness_metrics['cv_stability']:.4f}")
                    
                    # Overfitting warning
                    if robustness_metrics['r2_diff_train_test'] > 0.15:
                        print(f"      ⚠️ HIGH OVERFITTING RISK (R² diff > 0.15)")
                    elif robustness_metrics['r2_diff_train_test'] > 0.1:
                        print(f"      ⚠️ Medium overfitting risk (R² diff > 0.1)")
                    else:
                        print(f"      ✅ Low overfitting risk")
                    
                except Exception as e:
                    print(f"    ❌ Error training {model_name}: {str(e)}")
                    continue
            
            # Select best model using conservative criteria
            if model_results:
                print(f"\n🏆 Model Selection for Cluster {cluster_id}:")
                best_model_name, best_result = self.select_best_model_conservative(
                    model_results, verbose=True
                )
                
                if best_model_name:
                    # Store scaler used for this cluster
                    scaler_key = f"cluster_{cluster_id}"
                    self.scalers[scaler_key] = self.scaler
                    
                    # Compile cluster model information
                    cluster_models[cluster_id] = {
                        'models': model_results,
                        'best_model': best_result['model'],
                        'best_model_name': best_model_name,
                        'best_model_metrics': best_result['test_metrics'],
                        'best_model_robustness': best_result['robustness_metrics'],
                        'best_params': best_result['best_params'],
                        'selected_features': selected_features,
                        'feature_columns': selected_features,
                        'target_column': target_column,
                        'cluster_size': len(cluster_data),
                        'data_split': {
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'train_target_mean': np.mean(y_train),
                            'train_target_std': np.std(y_train),
                            'test_target_mean': np.mean(y_test),
                            'test_target_std': np.std(y_test)
                        },
                        'hyperparameter_tuning': hyperparameter_tuning,
                        'scaler': self.scaler,
                        'overfitting_assessment': {
                            'risk_level': 'Low' if best_result['robustness_metrics']['r2_diff_train_test'] <= 0.1 
                                         else 'Medium' if best_result['robustness_metrics']['r2_diff_train_test'] <= 0.2 
                                         else 'High',
                            'r2_difference': best_result['robustness_metrics']['r2_diff_train_test'],
                            'cv_stability': best_result['robustness_metrics']['cv_stability']
                        }
                    }
                    
                    # Store model performance
                    self.model_performance[cluster_id] = {
                        'best_model': best_model_name,
                        'performance': best_result['test_metrics'],
                        'robustness': best_result['robustness_metrics']
                    }
                    
                else:
                    print(f"  ❌ No suitable model found for cluster {cluster_id}")
                    continue
            else:
                print(f"  ❌ No models successfully trained for cluster {cluster_id}")
                continue

        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("ANTI-OVERFITTING MODEL BUILDING SUMMARY")
        print(f"{'='*80}")

        if cluster_models:
            print(f"✅ Successfully built models for {len(cluster_models)} clusters")
            
            # Analyze overfitting risks
            risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
            model_counts = {}
            r2_test_scores = []
            r2_differences = []
            cv_stabilities = []
            feature_counts = []
            
            for cluster_id, cluster_info in cluster_models.items():
                # Risk assessment
                risk_level = cluster_info['overfitting_assessment']['risk_level']
                risk_counts[risk_level] += 1
                
                # Model distribution
                best_model = cluster_info['best_model_name']
                model_counts[best_model] = model_counts.get(best_model, 0) + 1
                
                # Performance metrics
                robustness = cluster_info['best_model_robustness']
                r2_test_scores.append(robustness['r2_test'])
                r2_differences.append(robustness['r2_diff_train_test'])
                cv_stabilities.append(robustness['cv_stability'])
                feature_counts.append(len(cluster_info['selected_features']))
            
            # Risk distribution
            total_clusters = len(cluster_models)
            print(f"\n📊 Overfitting Risk Assessment:")
            for risk_level, count in risk_counts.items():
                percentage = (count / total_clusters) * 100
                emoji = "✅" if risk_level == "Low" else "⚠️" if risk_level == "Medium" else "❌"
                print(f"  {emoji} {risk_level} Risk: {count} clusters ({percentage:.1f}%)")
            
            # Model distribution
            print(f"\n🤖 Best Model Distribution:")
            for model_name, count in sorted(model_counts.items()):
                percentage = (count / total_clusters) * 100
                print(f"  {model_name}: {count} clusters ({percentage:.1f}%)")
            
            # Feature usage analysis
            print(f"\n🔍 Feature Selection Analysis:")
            print(f"  Average features per cluster: {np.mean(feature_counts):.1f}")
            print(f"  Feature count range: [{np.min(feature_counts)}, {np.max(feature_counts)}]")
            print(f"  Feature reduction: {len(feature_columns)} → {np.mean(feature_counts):.1f} avg")
            
            # Performance statistics
            print(f"\n📈 Performance Statistics:")
            print(f"  R² Test    - Mean: {np.mean(r2_test_scores):.4f} ± {np.std(r2_test_scores):.4f}")
            print(f"  R² Test    - Range: [{np.min(r2_test_scores):.4f}, {np.max(r2_test_scores):.4f}]")
            print(f"  R² Diff    - Mean: {np.mean(r2_differences):.4f} ± {np.std(r2_differences):.4f}")
            print(f"  R² Diff    - Range: [{np.min(r2_differences):.4f}, {np.max(r2_differences):.4f}]")
            print(f"  CV Stability - Mean: {np.mean(cv_stabilities):.4f} ± {np.std(cv_stabilities):.4f}")
            
            # Conservative success metrics
            conservative_success = sum(1 for r in r2_differences if r <= 0.15)
            conservative_percentage = (conservative_success / total_clusters) * 100
            
            print(f"\n🎯 Conservative Success Metrics:")
            print(f"  Models with R² diff ≤ 0.15: {conservative_success}/{total_clusters} ({conservative_percentage:.1f}%)")
            print(f"  Average CV stability: {np.mean(cv_stabilities):.3f}")
            
            # Recommendations
            print(f"\n💡 Recommendations:")
            if np.mean(r2_differences) <= 0.1:
                print("  ✅ Excellent overfitting control achieved")
            elif np.mean(r2_differences) <= 0.15:
                print("  ✅ Good overfitting control achieved")
            else:
                print("  ⚠️ Consider further regularization or feature reduction")
            
            if np.mean(cv_stabilities) >= 0.8:
                print("  ✅ High model stability achieved")
            elif np.mean(cv_stabilities) >= 0.7:
                print("  ✅ Adequate model stability achieved")
            else:
                print("  ⚠️ Consider simpler models or more data")

        else:
            print("❌ No models were successfully built for any cluster")
            print("Possible causes:")
            print("  - Clusters too small (< 30 samples)")
            print("  - Insufficient feature diversity")
            print("  - Target variable lacks variance")
            print("  - Data quality issues")

        # Save models if requested
        if save_models and cluster_models:
            try:
                self._save_models(cluster_models, models_dir)
                print(f"\n✅ Models saved to: {models_dir}")
            except Exception as e:
                print(f"\n❌ Error saving models: {str(e)}")

        # Store results in class attributes
        self.best_models = cluster_models

        print(f"\n🎉 Anti-overfitting model building completed!")
        print(f"📋 Built {len(cluster_models)} robust models with conservative validation")
        
        return cluster_models
    
    def _save_models(self, cluster_models: Dict, models_dir: str):
        """Save trained models to disk with additional metadata."""
        import os
        import joblib
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Save each cluster's models
        for cluster_id, cluster_info in cluster_models.items():
            cluster_dir = os.path.join(models_dir, f'cluster_{cluster_id}')
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Save best model
            model_file = os.path.join(cluster_dir, 'best_model.pkl')
            joblib.dump(cluster_info['best_model'], model_file)
            
            # Save scaler
            if cluster_info.get('scaler'):
                scaler_file = os.path.join(cluster_dir, 'scaler.pkl')
                joblib.dump(cluster_info['scaler'], scaler_file)
            
            # Save comprehensive metadata
            metadata = {
                'best_model_name': cluster_info['best_model_name'],
                'best_model_metrics': cluster_info['best_model_metrics'],
                'robustness_metrics': cluster_info['best_model_robustness'],
                'selected_features': cluster_info['selected_features'],
                'target_column': cluster_info['target_column'],
                'cluster_size': cluster_info['cluster_size'],
                'best_params': cluster_info['best_params'],
                'overfitting_assessment': cluster_info['overfitting_assessment'],
                'data_split': cluster_info['data_split']
            }
            
            metadata_file = os.path.join(cluster_dir, 'metadata.pkl')
            joblib.dump(metadata, metadata_file)
        
        # Save overall summary with anti-overfitting analysis
        summary = {
            'total_clusters': len(cluster_models),
            'feature_selection_summary': {
                cluster_id: info['selected_features'] 
                for cluster_id, info in cluster_models.items()
            },
            'overfitting_analysis': {
                cluster_id: info['overfitting_assessment']
                for cluster_id, info in cluster_models.items()
            },
            'model_distribution': {},
            'performance_summary': {}
        }
        
        # Calculate summary statistics
        model_counts = {}
        for cluster_info in cluster_models.values():
            model_name = cluster_info['best_model_name']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        summary['model_distribution'] = model_counts
        
        summary_file = os.path.join(models_dir, 'anti_overfit_summary.pkl')
        joblib.dump(summary, summary_file)
    
    def predict_cluster(
        self,
        cluster_id: int,
        X_new: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions for new data using the trained model for a specific cluster.
        
        Parameters
        ----------
        cluster_id : int
            Cluster ID to use for prediction
        X_new : pd.DataFrame
            New data to predict
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if cluster_id not in self.best_models:
            raise ValueError(f"No model found for cluster {cluster_id}")
        
        cluster_info = self.best_models[cluster_id]
        model = cluster_info['best_model']
        selected_features = cluster_info['selected_features']
        scaler = cluster_info.get('scaler')
        
        # Select only the features used during training
        X_selected = X_new[selected_features]
        
        # Scale if scaler was used
        if scaler is not None:
            X_scaled = scaler.transform(X_selected)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X_selected.values)
        
        return predictions
