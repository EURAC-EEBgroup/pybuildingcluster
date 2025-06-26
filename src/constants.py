"""Constants for PyBuildingCluster."""

SUPPORTED_MODELS = {
    'linear': 'LinearRegression',
    'random_forest': 'RandomForestRegressor',
    'gradient_boosting': 'GradientBoostingRegressor'
}

CLUSTERING_METHODS = ['kmeans', 'dbscan', 'hierarchical']