"""
Regression Modeling Module

This module provides regression modeling functionality for building energy performance
prediction using Random Forest, XGBoost, and LightGBM with automatic model selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error
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


class RegressionModelBuilder:
    """
    A comprehensive regression model builder for building energy performance prediction.
    
    This class provides methods for building and evaluating regression models using
    multiple algorithms with automatic hyperparameter tuning and model selection.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, problem_type='classification'):
        """
        Initialize the regression model builder.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        n_jobs : int, optional
            Number of parallel jobs, by default -1
        problem_type : str, optional
            Type of problem ('classification' or 'regression'), by default 'classification'
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.best_models = {}
        self.problem_type = problem_type
        
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str],
        test_size: float = 0.2,
        scale_features: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for modeling.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_column : str
            Name of the target column
        feature_columns : List[str]
            List of feature column names
        test_size : float, optional
            Proportion of data for testing, by default 0.2
        scale_features : bool, optional
            Whether to scale features, by default True
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
            X_train, X_test, y_train, y_test, feature_names
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
            X = X.fillna(X.median())
        
        if y.isnull().any():
            print("Warning: Missing values in target. Dropping corresponding rows.")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Identify categorical and numerical columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features if requested
        if scale_features:
            if self.problem_type == "classification":
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numerical_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ]
                )
                X_train_scaled = preprocessor.fit_transform(X_train)
                X_test_scaled = preprocessor.transform(X_test)
            else: 
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Store scaler for later use
                self.scaler = scaler
            
            return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_columns
        else:
            self.scaler = None
            return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns
    
    def get_model_configurations(self) -> Dict[str, Dict]:
        """
        Get default model configurations for hyperparameter tuning.
        Supports both regression and classification problems.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary of model configurations
        """
        configs = {}
    
        # Determine if it's classification or regression
        is_classification = hasattr(self, 'problem_type') and self.problem_type == "classification"
        
        if is_classification:
            # Classification models
            from sklearn.ensemble import RandomForestClassifier
            
            configs['random_forest'] = {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1,
                    class_weight='balanced'  # Handle imbalanced classes
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'criterion': ['gini', 'entropy']
                },
                'param_distributions': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, 0.9],
                    'criterion': ['gini', 'entropy']
                }
            }
            
            # Add XGBoost Classifier if available
            if HAS_XGBOOST:
                configs['xgboost'] = {
                    'model': xgb.XGBClassifier(
                        random_state=self.random_state,
                        n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1,
                        verbosity=0,
                        eval_metric='logloss'
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    'param_distributions': {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [3, 4, 5, 6, 7, 8, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5, 1.0],
                        'reg_lambda': [0, 0.1, 0.5, 1.0]
                    }
                }
            
            # Add LightGBM Classifier if available
            if HAS_LIGHTGBM:
                configs['lightgbm'] = {
                    'model': lgb.LGBMClassifier(
                        random_state=self.random_state,
                        n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1,
                        verbosity=-1,
                        class_weight='balanced'
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 70],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    'param_distributions': {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [3, 4, 5, 6, 7, 8, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                        'num_leaves': [15, 31, 50, 70, 100],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5, 1.0],
                        'reg_lambda': [0, 0.1, 0.5, 1.0]
                    }
                }
            
            # Add Support Vector Machine for classification
            from sklearn.svm import SVC
            configs['svm'] = {
                'model': SVC(
                    random_state=self.random_state,
                    probability=True,  # Enable probability estimates
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                },
                'param_distributions': {
                    'C': [0.01, 0.1, 1, 10, 50, 100, 200],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10]
                }
            }
            
            # Add Logistic Regression for classification
            from sklearn.linear_model import LogisticRegression
            configs['logistic_regression'] = {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                },
                'param_distributions': {
                    'C': [0.01, 0.1, 1, 10, 50, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Only used with elasticnet
                }
            }
            
        else:
            # Regression models (your existing code)
            from sklearn.ensemble import RandomForestRegressor
            
            configs['random_forest'] = {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                'param_distributions': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 6],
                    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, 0.9]
                }
            }
            
            # Add XGBoost Regressor if available
            if HAS_XGBOOST:
                configs['xgboost'] = {
                    'model': xgb.XGBRegressor(
                        random_state=self.random_state,
                        n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1,
                        verbosity=0
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    'param_distributions': {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [3, 4, 5, 6, 7, 8, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5, 1.0],
                        'reg_lambda': [0, 0.1, 0.5, 1.0]
                    }
                }
            
            # Add LightGBM Regressor if available
            if HAS_LIGHTGBM:
                configs['lightgbm'] = {
                    'model': lgb.LGBMRegressor(
                        random_state=self.random_state,
                        n_jobs=self.n_jobs if hasattr(self, 'n_jobs') else -1,
                        verbosity=-1
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 70],
                        'subsample': [0.8, 0.9, 1.0]
                    },
                    'param_distributions': {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [3, 4, 5, 6, 7, 8, 9],
                        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                        'num_leaves': [15, 31, 50, 70, 100],
                        'subsample': [0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0, 0.1, 0.5, 1.0],
                        'reg_lambda': [0, 0.1, 0.5, 1.0]
                    }
                }
            
            # Add Support Vector Regression
            from sklearn.svm import SVR
            configs['svr'] = {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.1, 0.2, 0.5]
                },
                'param_distributions': {
                    'C': [0.01, 0.1, 1, 10, 50, 100, 200],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1, 10],
                    'epsilon': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
                }
            }
            
            # Add Linear Regression variants
            from sklearn.linear_model import Ridge, Lasso, ElasticNet
            configs['ridge'] = {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1, 10, 100],
                    'solver': ['auto', 'saga', 'lsqr']
                },
                'param_distributions': {
                    'alpha': [0.01, 0.1, 1, 10, 50, 100, 200],
                    'solver': ['auto', 'saga', 'lsqr', 'sparse_cg']
                }
            }
            
            configs['lasso'] = {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                },
                'param_distributions': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]
                }
            }
            
            configs['elastic_net'] = {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.1, 1, 10, 100],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'param_distributions': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
                }
            }
        
        return configs
    
    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparameter_tuning: str = "false",
        cv_folds: int = 5,
        n_iter: int = 50,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Parameters
        ----------
        model_name : str
            Name of the model to train
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
        hyperparameter_tuning : str, optional
            Type of hyperparameter tuning ('grid', 'randomized', 'none'), by default "randomized"
        cv_folds : int, optional
            Number of cross-validation folds, by default 5
        n_iter : int, optional
            Number of iterations for randomized search, by default 50
            
        Returns
        -------
        Dict[str, Any]
            Model training results
        """
        configs = self.get_model_configurations()
        
        if model_name not in configs:
            available_models = list(configs.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        config = configs[model_name]
        base_model = config['model']
        
        print(f"Training {model_name}...")
        start_time = time.time()
        
        if hyperparameter_tuning == "grid":
            # Grid search
            search = GridSearchCV(
                base_model,
                config['params'],
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                #n_jobs=self.n_jobs,
                verbose=0
            )
        elif hyperparameter_tuning == "randomized":
            # Randomized search
            search = RandomizedSearchCV(
                base_model,
                config['param_distributions'],
                n_iter=n_iter,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                #n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
        else:
            # No hyperparameter tuning
            search = base_model
        
        # Fit the model
        if hyperparameter_tuning != "none":
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_score = -search.best_score_
        else:
            search.fit(X_train, y_train)
            best_model = search
            best_params = {}
            # Cross-validation evaluation
            if self.problem_type == 'classification':
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                print(f"Accuracy in cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            else:
                try:
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='neg_root_mean_squared_error')
                    print(f"RMSE in cross-validation: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except:
                    pass
            cv_score = -cv_scores.mean()
        
        # Make predictions
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, train_pred)
        test_metrics = self._calculate_metrics(y_test, test_pred)
        
        # Feature importance
        importance = self._get_feature_importance(best_model, model_name, feature_names)
        
        training_time = time.time() - start_time
        
        return {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_score': cv_score,
            'feature_importance': importance,
            'training_time': training_time,
            'train_predictions': train_pred,
            'test_predictions': test_pred
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def _get_feature_importance(self, model, model_name: str, feature_names: List[str] = None) -> Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]]:
        """
        Extract feature importance from model with enhanced support for different model types.
        
        Parameters
        ----------
        model : sklearn model
            Trained model instance
        model_name : str
            Name of the model
        feature_names : List[str], optional
            Names of the features, by default None
            
        Returns
        -------
        Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]]
            Dictionary containing feature importance data and metadata
        """
        try:
            importance_data = {}
            
            # Try different methods to extract feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost, LightGBM, etc.)
                importances = model.feature_importances_
                importance_type = 'feature_importances'
                
            elif hasattr(model, 'coef_'):
                # Linear models (Logistic Regression, Ridge, Lasso, etc.)
                coef = model.coef_
                
                # Handle different coefficient shapes
                if len(coef.shape) == 1:
                    # Binary classification or regression
                    importances = np.abs(coef)
                else:
                    # Multi-class classification - take mean across classes
                    importances = np.mean(np.abs(coef), axis=0)
                
                importance_type = 'coefficients'
                
            elif hasattr(model, 'dual_coef_') and hasattr(model, 'support_'):
                # Support Vector Machines
                if hasattr(model, 'coef_'):
                    # Linear SVM
                    coef = model.coef_
                    if len(coef.shape) == 1:
                        importances = np.abs(coef)
                    else:
                        importances = np.mean(np.abs(coef), axis=0)
                else:
                    # Non-linear SVM - feature importance not directly available
                    print(f"Warning: {model_name} (non-linear SVM) doesn't provide direct feature importance")
                    return None
                
                importance_type = 'svm_coefficients'
                
            elif model_name.lower() in ['xgboost', 'xgb'] and hasattr(model, 'get_booster'):
                # XGBoost specific method
                try:
                    booster = model.get_booster()
                    importance_dict = booster.get_score(importance_type='gain')
                    
                    # Create importance array in correct order
                    if feature_names:
                        importances = np.array([importance_dict.get(f'f{i}', 0.0) for i in range(len(feature_names))])
                    else:
                        importances = np.array(list(importance_dict.values()))
                    
                    importance_type = 'xgboost_gain'
                except Exception:
                    # Fallback to feature_importances_ if available
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        importance_type = 'feature_importances'
                    else:
                        return None
                        
            elif model_name.lower() in ['lightgbm', 'lgb'] and hasattr(model, 'booster_'):
                # LightGBM specific method
                try:
                    importances = model.feature_importances_
                    importance_type = 'lightgbm_importance'
                except Exception:
                    return None
                    
            else:
                # Model doesn't support feature importance
                print(f"Warning: {model_name} doesn't provide feature importance")
                return None
            
            # Validate importances
            if importances is None or len(importances) == 0:
                return None
                
            # Ensure importances is a numpy array
            importances = np.array(importances)
            
            # Create feature names if not provided
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            elif len(feature_names) != len(importances):
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            # Create DataFrame for easier manipulation
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Importance_Normalized': importances / np.sum(importances) if np.sum(importances) > 0 else importances
            }).sort_values(by='Importance', ascending=False)
            
            # Calculate additional statistics
            importance_stats = {
                'total_importance': np.sum(importances),
                'mean_importance': np.mean(importances),
                'std_importance': np.std(importances),
                'max_importance': np.max(importances),
                'min_importance': np.min(importances),
                'n_features': len(importances),
                'importance_type': importance_type
            }
            
            # Store results
            importance_data = {
                'importances': importances,
                'feature_names': feature_names,
                'importance_df': feature_importance_df,
                'importance_stats': importance_stats,
                'model_name': model_name,
                'importance_type': importance_type
            }
            
            return importance_data
            
        except Exception as e:
            print(f"Error extracting feature importance for {model_name}: {str(e)}")
            return None


    

    def build_models(
        self,
        data: pd.DataFrame,
        clusters: Dict,
        target_column: str,
        feature_columns: List[str],
        models_to_train: Optional[List[str]] = None,
        hyperparameter_tuning: str = "none",
        models_dir: str = "models",
        save_models: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build regression models for each cluster.

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
            List of models to train, by default None (trains all available)
        hyperparameter_tuning : str, optional
            Type of hyperparameter tuning, by default "randomized"
        models_dir : str, optional
            Directory to save models, by default "models"
        save_models : bool, optional
            Whether to save trained models, by default True
            
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary mapping cluster IDs to model results
        """
        # Get available models
        available_models = list(self.get_model_configurations().keys())
        
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

        print(f"Building regression models for {len(unique_clusters)} clusters...")
        print(f"Models to train: {models_to_train}")
        print(f"Hyperparameter tuning: {hyperparameter_tuning}")
        print("-" * 60)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points from DBSCAN
                print(f"Skipping noise cluster (ID: -1)")
                continue
                
            print(f"\nProcessing Cluster {cluster_id}")
            print("-" * 40)
            
            # Get cluster data
            cluster_data = clustered_data[clustered_data['cluster'] == cluster_id].copy()
            
            # Check minimum sample size
            min_samples_required = 20  # Minimum for reliable model training
            if len(cluster_data) < min_samples_required:
                print(f"Warning: Cluster {cluster_id} has only {len(cluster_data)} samples.")
                print(f"Minimum required: {min_samples_required}. Skipping this cluster.")
                continue
            
            # Check for target variance
            target_values = cluster_data[target_column].dropna()
            if len(target_values) == 0:
                print(f"Warning: No valid target values in cluster {cluster_id}. Skipping.")
                continue
            
            if target_values.std() == 0:
                print(f"Warning: No variance in target column for cluster {cluster_id}. Skipping.")
                continue
            
            print(f"Cluster size: {len(cluster_data)} samples")
            print(f"Target statistics: mean={target_values.mean():.3f}, std={target_values.std():.3f}")
            
            # Prepare data for this cluster
            try:
                X_train, X_test, y_train, y_test, features = self.prepare_data(
                    cluster_data, target_column, feature_columns,
                    test_size=0.2, scale_features=False
                )
                
                print(f"Training set: {len(X_train)} samples")
                print(f"Test set: {len(X_test)} samples")
                
            except Exception as e:
                print(f"Error preparing data for cluster {cluster_id}: {str(e)}")
                continue
            
            # Train models for this cluster
            cluster_results = {}
            model_performances = []
            
            for model_name in models_to_train:
                print(f"\nTraining {model_name}...")
                
                try:
                    model_result = self.train_single_model(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        hyperparameter_tuning=hyperparameter_tuning,
                        cv_folds=5,
                        n_iter=50 if hyperparameter_tuning == "randomized" else None,
                        feature_names=features,
                    )
                    
                    cluster_results[model_name] = model_result
                    
                    # Store performance for comparison
                    performance_info = {
                        'model': model_name,
                        'cv_score': model_result['cv_score'],
                        'test_r2': model_result['test_metrics']['r2'],
                        'test_rmse': model_result['test_metrics']['rmse'],
                        'test_mae': model_result['test_metrics']['mae'],
                        'training_time': model_result['training_time']
                    }
                    model_performances.append(performance_info)
                    
                    # Print performance
                    metrics = model_result['test_metrics']
                    print(f"  ✓ {model_name} completed:")
                    print(f"    R² Score: {metrics['r2']:.4f}")
                    print(f"    RMSE: {metrics['rmse']:.4f}")
                    print(f"    MAE: {metrics['mae']:.4f}")
                    print(f"    MAPE: {metrics['mape']:.2f}%")
                    print(f"    CV Score (RMSE): {model_result['cv_score']:.4f}")
                    print(f"    Training time: {model_result['training_time']:.2f}s")
                    
                except Exception as e:
                    print(f"  ✗ Error training {model_name}: {str(e)}")
                    continue
            
            # Select best model and compile cluster results
            if model_performances:
                # Select best model based on CV score (lowest RMSE)
                best_model_info = min(model_performances, key=lambda x: x['cv_score'])
                best_model_name = best_model_info['model']
                best_model_result = cluster_results[best_model_name]
                
                print(f"\n🏆 Best model for cluster {cluster_id}: {best_model_name}")
                print(f"   CV RMSE: {best_model_info['cv_score']:.4f}")
                print(f"   Test R²: {best_model_info['test_r2']:.4f}")
                
                # Store scaler used for this cluster
                scaler_key = f"cluster_{cluster_id}"
                self.scalers[scaler_key] = self.scaler
                
                # Compile cluster model information
                cluster_models[cluster_id] = {
                    'models': cluster_results,
                    'best_model': best_model_result['model'],
                    'best_model_name': best_model_name,
                    'best_model_metrics': best_model_result['test_metrics'],
                    'best_model_cv_score': best_model_info['cv_score'],
                    'model_comparison': model_performances,
                    'feature_columns': features,
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
                    'feature_importance': best_model_result.get('feature_importance'),
                    'hyperparameter_tuning': hyperparameter_tuning,
                    'best_params': best_model_result.get('best_params', {}),
                    'scaler': self.scaler
                }
                
                # Store feature importance for analysis
                # if best_model_result.get('feature_importance') is not None:
                #     importance_dict = dict(zip(features, best_model_result['feature_importance']))
                #     self.feature_importance[cluster_id] = importance_dict
                
                # Store model performance
                self.model_performance[cluster_id] = {
                    'best_model': best_model_name,
                    'performance': best_model_result['test_metrics'],
                    'cv_score': best_model_info['cv_score']
                }
                
            else:
                print(f"  ❌ No models successfully trained for cluster {cluster_id}")
                continue

        # Print overall summary
        print(f"\n{'='*60}")
        print("MODEL BUILDING SUMMARY")
        print(f"{'='*60}")

        if cluster_models:
            print(f"Successfully built models for {len(cluster_models)} clusters")
            
            # Best model distribution
            model_counts = {}
            total_r2_scores = []
            total_rmse_scores = []
            
            for cluster_id, cluster_data in cluster_models.items():
                best_model = cluster_data['best_model_name']
                model_counts[best_model] = model_counts.get(best_model, 0) + 1
                
                metrics = cluster_data['best_model_metrics']
                total_r2_scores.append(metrics['r2'])
                total_rmse_scores.append(metrics['rmse'])
            
            print(f"\nBest model distribution:")
            for model_name, count in sorted(model_counts.items()):
                percentage = (count / len(cluster_models)) * 100
                print(f"  {model_name}: {count} clusters ({percentage:.1f}%)")
            
            # Overall performance statistics
            if total_r2_scores:
                print(f"\nOverall performance statistics:")
                print(f"  R² Score  - Mean: {np.mean(total_r2_scores):.4f} ± {np.std(total_r2_scores):.4f}")
                print(f"  R² Score  - Range: [{np.min(total_r2_scores):.4f}, {np.max(total_r2_scores):.4f}]")
                print(f"  RMSE      - Mean: {np.mean(total_rmse_scores):.4f} ± {np.std(total_rmse_scores):.4f}")
                print(f"  RMSE      - Range: [{np.min(total_rmse_scores):.4f}, {np.max(total_rmse_scores):.4f}]")
            

        else:
            print("❌ No models were successfully built for any cluster")
            print("Check your data quality, cluster sizes, and parameter settings")

        # Save models if requested
        if save_models and cluster_models:
            try:
                self._save_models(cluster_models, models_dir)
                print(f"\n✅ Models saved to: {models_dir}")
            except Exception as e:
                print(f"\n❌ Error saving models: {str(e)}")

        # Store results in class attributes
        self.best_models = cluster_models

        print(f"\n🎉 Model building completed!")
        return cluster_models
    
    def _save_models(self, cluster_models: Dict, models_dir: str):
        """Save trained models to disk."""
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
            
            # Save metadata
            metadata = {
                'best_model_name': cluster_info['best_model_name'],
                'best_model_metrics': cluster_info['best_model_metrics'],
                'feature_columns': cluster_info['feature_columns'],
                'target_column': cluster_info['target_column'],
                'cluster_size': cluster_info['cluster_size'],
                'best_params': cluster_info['best_params']
            }
            
            metadata_file = os.path.join(cluster_dir, 'metadata.pkl')
            joblib.dump(metadata, metadata_file)
        
        # Save overall summary
        summary_file = os.path.join(models_dir, 'models_summary.pkl')
        joblib.dump({
            'cluster_models': {cid: {k: v for k, v in info.items() if k != 'best_model'} 
                              for cid, info in cluster_models.items()}
        }, summary_file)