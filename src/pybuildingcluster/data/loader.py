"""
Data Loading Module

This module provides comprehensive data loading and preprocessing functionality
for building energy performance data analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
import os
import json
import yaml
from io import StringIO, BytesIO


class DataLoader:
    """
    A comprehensive data loader for building energy performance analysis.
    
    This class provides methods for loading data from various formats,
    preprocessing, validation, and basic exploratory data analysis.
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        encoding : str, optional
            Default encoding for text files, by default 'utf-8'
        """
        self.encoding = encoding
        self.loaded_data = {}
        self.data_info = {}
        
    def load_csv(
        self, 
        filepath: Union[str, Path], 
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file with comprehensive error handling.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the CSV file
        **kwargs
            Additional arguments passed to pd.read_csv
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
            
        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be parsed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Default CSV reading parameters
        csv_defaults = {
            'encoding': self.encoding,
            'low_memory': False,
            'parse_dates': True,
            'infer_datetime_format': True
        }
        
        # Merge with user parameters
        csv_params = {**csv_defaults, **kwargs}
        
        try:
            # Try to read with default parameters
            data = pd.read_csv(filepath, **csv_params)
            
            # Store metadata
            self.data_info[str(filepath)] = {
                'filepath': str(filepath),
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'loading_params': csv_params
            }
            
            print(f"✅ Successfully loaded CSV: {filepath}")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {len(data.columns)}")
            print(f"   File size: {self.data_info[str(filepath)]['file_size_mb']:.2f} MB")
            
            return data
            
        except UnicodeDecodeError as e:
            # Try different encodings
            encodings_to_try = ['latin-1', 'iso-8859-1', 'cp1252']
            print(f"⚠️  Unicode error with {self.encoding}, trying alternative encodings...")
            
            for encoding in encodings_to_try:
                try:
                    csv_params['encoding'] = encoding
                    data = pd.read_csv(filepath, **csv_params)
                    print(f"✅ Successfully loaded with encoding: {encoding}")
                    return data
                except Exception:
                    continue
            
            raise ValueError(f"Could not decode file {filepath} with any encoding")
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {filepath}")
            
        except pd.errors.ParserError as e:
            # Try with different separators
            separators_to_try = [';', '\t', '|']
            print(f"⚠️  Parser error, trying alternative separators...")
            
            for sep in separators_to_try:
                try:
                    csv_params['sep'] = sep
                    data = pd.read_csv(filepath, **csv_params)
                    print(f"✅ Successfully loaded with separator: '{sep}'")
                    return data
                except Exception:
                    continue
            
            raise ValueError(f"Could not parse CSV file {filepath}: {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Error loading CSV file {filepath}: {str(e)}")
    
    def load_excel(
        self, 
        filepath: Union[str, Path], 
        sheet_name: Union[str, int, None] = 0,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from Excel file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the Excel file
        sheet_name : Union[str, int, None], optional
            Sheet name or index to load, by default 0
        **kwargs
            Additional arguments passed to pd.read_excel
            
        Returns
        -------
        Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            Loaded DataFrame or dict of DataFrames
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        
        try:
            # Default Excel reading parameters
            excel_defaults = {
                'sheet_name': sheet_name,
                'parse_dates': True
            }
            
            # Merge with user parameters
            excel_params = {**excel_defaults, **kwargs}
            
            data = pd.read_excel(filepath, **excel_params)
            
            # Handle multiple sheets
            if isinstance(data, dict):
                print(f"✅ Successfully loaded Excel file with {len(data)} sheets: {filepath}")
                for sheet, df in data.items():
                    print(f"   Sheet '{sheet}': {df.shape}")
                
                # Store metadata for each sheet
                for sheet, df in data.items():
                    key = f"{filepath}#{sheet}"
                    self.data_info[key] = {
                        'filepath': str(filepath),
                        'sheet_name': sheet,
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict()
                    }
            else:
                print(f"✅ Successfully loaded Excel sheet: {filepath}")
                print(f"   Shape: {data.shape}")
                
                # Store metadata
                self.data_info[str(filepath)] = {
                    'filepath': str(filepath),
                    'sheet_name': sheet_name,
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict()
                }
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading Excel file {filepath}: {str(e)}")
    
    def load_json(
        self, 
        filepath: Union[str, Path], 
        orient: str = 'records',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the JSON file
        orient : str, optional
            JSON orientation, by default 'records'
        **kwargs
            Additional arguments passed to pd.read_json
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        try:
            json_params = {'orient': orient, **kwargs}
            data = pd.read_json(filepath, **json_params)
            
            # Store metadata
            self.data_info[str(filepath)] = {
                'filepath': str(filepath),
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'orient': orient
            }
            
            print(f"✅ Successfully loaded JSON: {filepath}")
            print(f"   Shape: {data.shape}")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error loading JSON file {filepath}: {str(e)}")
    
    def auto_detect_and_load(
        self, 
        filepath: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Automatically detect file format and load data.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the data file
        **kwargs
            Additional arguments passed to the appropriate loader
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix in ['.csv', '.txt']:
            return self.load_csv(filepath, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        elif suffix == '.json':
            return self.load_json(filepath, **kwargs)
        else:
            # Try to detect format by content
            try:
                # Try CSV first
                return self.load_csv(filepath, **kwargs)
            except Exception:
                try:
                    # Try JSON
                    return self.load_json(filepath, **kwargs)
                except Exception:
                    raise ValueError(f"Unsupported file format: {suffix}")
    
    def validate_data(
        self, 
        data: pd.DataFrame, 
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Validate loaded data and return validation report.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to validate
        required_columns : Optional[List[str]], optional
            List of required column names, by default None
        min_rows : int, optional
            Minimum number of rows required, by default 1
        check_duplicates : bool, optional
            Whether to check for duplicate rows, by default True
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic shape validation
        if data.empty:
            validation_report['is_valid'] = False
            validation_report['errors'].append("DataFrame is empty")
            return validation_report
        
        if len(data) < min_rows:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"DataFrame has {len(data)} rows, minimum required: {min_rows}")
        
        # Column validation
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_report['is_valid'] = False
                validation_report['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_stats = data.isnull().sum()
        missing_cols = missing_stats[missing_stats > 0]
        
        if len(missing_cols) > 0:
            validation_report['warnings'].append(f"Columns with missing values: {missing_cols.to_dict()}")
        
        # Check for duplicates
        if check_duplicates:
            n_duplicates = data.duplicated().sum()
            if n_duplicates > 0:
                validation_report['warnings'].append(f"Found {n_duplicates} duplicate rows")
        
        # Data type analysis
        dtype_summary = data.dtypes.value_counts().to_dict()
        validation_report['info']['dtype_distribution'] = dtype_summary
        
        # Memory usage
        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        validation_report['info']['memory_usage_mb'] = memory_usage_mb
        
        # Numeric columns statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            validation_report['info']['numeric_columns'] = len(numeric_cols)
            validation_report['info']['numeric_stats'] = {
                'has_negatives': (data[numeric_cols] < 0).any().any(),
                'has_zeros': (data[numeric_cols] == 0).any().any(),
                'has_infinites': np.isinf(data[numeric_cols]).any().any()
            }
        
        # Categorical columns analysis
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            validation_report['info']['categorical_columns'] = len(categorical_cols)
            validation_report['info']['unique_values_per_categorical'] = {
                col: data[col].nunique() for col in categorical_cols
            }
        
        return validation_report
    
    def explore_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform exploratory data analysis on the loaded data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to explore
            
        Returns
        -------
        Dict[str, Any]
            Exploratory analysis results
        """
        exploration_report = {
            'basic_info': {},
            'statistical_summary': {},
            'missing_data': {},
            'data_quality': {},
            'recommendations': []
        }
        
        # Basic information
        exploration_report['basic_info'] = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Statistical summary for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            exploration_report['statistical_summary']['numeric'] = numeric_data.describe().to_dict()
            
            # Correlation analysis
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'col1': corr_matrix.columns[i],
                                'col2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })
                
                exploration_report['statistical_summary']['high_correlations'] = high_corr_pairs
        
        # Categorical data summary
        categorical_data = data.select_dtypes(include=['object', 'category'])
        if not categorical_data.empty:
            cat_summary = {}
            for col in categorical_data.columns:
                cat_summary[col] = {
                    'unique_count': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'frequency_of_most_frequent': data[col].value_counts().iloc[0] if not data[col].empty else 0
                }
            exploration_report['statistical_summary']['categorical'] = cat_summary
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        missing_percentage = (missing_data / len(data)) * 100
        
        exploration_report['missing_data'] = {
            'total_missing_values': missing_data.sum(),
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentage': missing_percentage[missing_percentage > 0].to_dict()
        }
        
        # Data quality assessment
        exploration_report['data_quality'] = {
            'duplicate_rows': data.duplicated().sum(),
            'empty_columns': [col for col in data.columns if data[col].isnull().all()],
            'single_value_columns': [col for col in data.columns if data[col].nunique() <= 1],
            'potential_id_columns': [col for col in data.columns if data[col].nunique() == len(data)]
        }
        
        # Generate recommendations
        recommendations = []
        
        # Missing data recommendations
        if exploration_report['missing_data']['total_missing_values'] > 0:
            high_missing_cols = [
                col for col, pct in exploration_report['missing_data']['missing_percentage'].items() 
                if pct > 50
            ]
            if high_missing_cols:
                recommendations.append(f"Consider removing columns with >50% missing data: {high_missing_cols}")
            
            moderate_missing_cols = [
                col for col, pct in exploration_report['missing_data']['missing_percentage'].items() 
                if 10 < pct <= 50
            ]
            if moderate_missing_cols:
                recommendations.append(f"Consider imputation for columns with 10-50% missing data: {moderate_missing_cols}")
        
        # Data quality recommendations
        if exploration_report['data_quality']['duplicate_rows'] > 0:
            recommendations.append(f"Remove {exploration_report['data_quality']['duplicate_rows']} duplicate rows")
        
        if exploration_report['data_quality']['empty_columns']:
            recommendations.append(f"Remove empty columns: {exploration_report['data_quality']['empty_columns']}")
        
        if exploration_report['data_quality']['single_value_columns']:
            recommendations.append(f"Consider removing single-value columns: {exploration_report['data_quality']['single_value_columns']}")
        
        # Correlation recommendations
        if 'high_correlations' in exploration_report['statistical_summary']:
            high_corr = exploration_report['statistical_summary']['high_correlations']
            if high_corr:
                recommendations.append(f"Found {len(high_corr)} highly correlated feature pairs - consider feature selection")
        
        exploration_report['recommendations'] = recommendations
        
        return exploration_report
    
    def remove_columns(
        self, 
        data: pd.DataFrame, 
        columns_to_remove: List[str]
    ) -> pd.DataFrame:
        """
        Remove specified columns from DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        columns_to_remove : List[str]
            List of column names to remove
            
        Returns
        -------
        pd.DataFrame
            DataFrame with specified columns removed
        """
        existing_columns = [col for col in columns_to_remove if col in data.columns]
        missing_columns = [col for col in columns_to_remove if col not in data.columns]
        
        if missing_columns:
            warnings.warn(f"Columns not found in data: {missing_columns}")
        
        if existing_columns:
            data = data.drop(columns=existing_columns)
            print(f"✅ Removed {len(existing_columns)} columns: {existing_columns}")
        
        return data
    
    def clean_data(
        self, 
        data: pd.DataFrame,
        remove_duplicates: bool = True,
        remove_empty_columns: bool = True,
        remove_single_value_columns: bool = True,
        missing_threshold: float = 0.5,
        fill_missing: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Clean data with various preprocessing steps.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame
        remove_duplicates : bool, optional
            Whether to remove duplicate rows, by default True
        remove_empty_columns : bool, optional
            Whether to remove completely empty columns, by default True
        remove_single_value_columns : bool, optional
            Whether to remove columns with only one unique value, by default True
        missing_threshold : float, optional
            Remove columns with missing values above this threshold, by default 0.5
        fill_missing : Optional[str], optional
            Strategy for filling missing values ('mean', 'median', 'mode', 'forward', 'backward'), by default None
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame
        """
        initial_shape = data.shape
        cleaned_data = data.copy()
        
        print(f"🧹 Starting data cleaning...")
        print(f"   Initial shape: {initial_shape}")
        
        # Remove duplicate rows
        if remove_duplicates:
            n_duplicates = cleaned_data.duplicated().sum()
            if n_duplicates > 0:
                cleaned_data = cleaned_data.drop_duplicates()
                print(f"   ✅ Removed {n_duplicates} duplicate rows")
        
        # Remove empty columns
        if remove_empty_columns:
            empty_cols = [col for col in cleaned_data.columns if cleaned_data[col].isnull().all()]
            if empty_cols:
                cleaned_data = cleaned_data.drop(columns=empty_cols)
                print(f"   ✅ Removed {len(empty_cols)} empty columns: {empty_cols}")
        
        # Remove single value columns
        if remove_single_value_columns:
            single_value_cols = [col for col in cleaned_data.columns if cleaned_data[col].nunique() <= 1]
            if single_value_cols:
                cleaned_data = cleaned_data.drop(columns=single_value_cols)
                print(f"   ✅ Removed {len(single_value_cols)} single-value columns: {single_value_cols}")
        
        # Remove columns with too many missing values
        if missing_threshold < 1.0:
            missing_pct = cleaned_data.isnull().sum() / len(cleaned_data)
            high_missing_cols = missing_pct[missing_pct > missing_threshold].index.tolist()
            if high_missing_cols:
                cleaned_data = cleaned_data.drop(columns=high_missing_cols)
                print(f"   ✅ Removed {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values")
        
        # Fill missing values
        if fill_missing:
            if fill_missing == 'mean':
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].mean())
                print(f"   ✅ Filled missing values in {len(numeric_cols)} numeric columns with mean")
                
            elif fill_missing == 'median':
                numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].median())
                print(f"   ✅ Filled missing values in {len(numeric_cols)} numeric columns with median")
                
            elif fill_missing == 'mode':
                for col in cleaned_data.columns:
                    if cleaned_data[col].isnull().any():
                        mode_val = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 0
                        cleaned_data[col] = cleaned_data[col].fillna(mode_val)
                print(f"   ✅ Filled missing values with mode")
                
            elif fill_missing == 'forward':
                cleaned_data = cleaned_data.fillna(method='ffill')
                print(f"   ✅ Forward filled missing values")
                
            elif fill_missing == 'backward':
                cleaned_data = cleaned_data.fillna(method='bfill')
                print(f"   ✅ Backward filled missing values")
        
        final_shape = cleaned_data.shape
        print(f"   🎉 Cleaning completed!")
        print(f"   Final shape: {final_shape}")
        print(f"   Removed {initial_shape[0] - final_shape[0]} rows and {initial_shape[1] - final_shape[1]} columns")
        
        return cleaned_data
    
    def save_data(
        self, 
        data: pd.DataFrame, 
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to file in specified format.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to save
        filepath : Union[str, Path]
            Output file path
        format : Optional[str], optional
            Output format ('csv', 'excel', 'json'), by default None (auto-detect from extension)
        **kwargs
            Additional arguments passed to the save function
        """
        filepath = Path(filepath)
        
        # Auto-detect format if not specified
        if format is None:
            suffix = filepath.suffix.lower()
            if suffix == '.csv':
                format = 'csv'
            elif suffix in ['.xlsx', '.xls']:
                format = 'excel'
            elif suffix == '.json':
                format = 'json'
            else:
                format = 'csv'  # Default to CSV
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'csv':
                csv_defaults = {'index': False, 'encoding': self.encoding}
                csv_params = {**csv_defaults, **kwargs}
                data.to_csv(filepath, **csv_params)
                
            elif format == 'excel':
                excel_defaults = {'index': False}
                excel_params = {**excel_defaults, **kwargs}
                data.to_excel(filepath, **excel_params)
                
            elif format == 'json':
                json_defaults = {'orient': 'records', 'indent': 2}
                json_params = {**json_defaults, **kwargs}
                data.to_json(filepath, **json_params)
            
            print(f"✅ Data saved successfully to: {filepath}")
            print(f"   Format: {format}")
            print(f"   Shape: {data.shape}")
            
        except Exception as e:
            raise ValueError(f"Error saving data to {filepath}: {str(e)}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all loaded datasets.
        
        Returns
        -------
        pd.DataFrame
            Summary table of loaded datasets
        """
        if not self.data_info:
            print("No data loaded yet")
            return pd.DataFrame()
        
        summary_data = []
        for key, info in self.data_info.items():
            summary_data.append({
                'dataset': key,
                'filepath': info.get('filepath', ''),
                'shape': f"{info['shape'][0]} × {info['shape'][1]}",
                'columns': info['shape'][1],
                'rows': info['shape'][0],
                'file_size_mb': info.get('file_size_mb', 0),
                'dtypes': len(set(info['dtypes'].values())),
                'sheet_name': info.get('sheet_name', '')
            })
        
        return pd.DataFrame(summary_data)