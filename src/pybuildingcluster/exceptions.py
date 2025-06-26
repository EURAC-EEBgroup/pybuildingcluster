"""Custom exceptions for PyBuildingCluster."""

class PyBuildingClusterError(Exception):
    """Base exception for PyBuildingCluster."""
    pass

class DataValidationError(PyBuildingClusterError):
    """Raised when data validation fails."""
    pass

class ClusteringError(PyBuildingClusterError):
    """Raised when clustering fails."""
    pass

class ModelTrainingError(PyBuildingClusterError):
    """Raised when model training fails."""
    pass

class SensitivityAnalysisError(PyBuildingClusterError):
    """Raised when sensitivity analysis fails."""
    pass