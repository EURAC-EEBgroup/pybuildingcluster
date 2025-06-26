"""
Command Line Interface for Geoclustering Sensitivity Analysis

This module provides a CLI for running clustering and sensitivity analysis
from the command line.
"""

import click
import os
import yaml
from pathlib import Path
from typing import Optional, List

from . import GeoClusteringAnalyzer
from .utils.config import ConfigManager


@click.command()
@click.option(
    '--data-path', 
    required=True,
    type=click.Path(exists=True),
    help='Path to the input CSV data file'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Path to configuration YAML file'
)
@click.option(
    '--columns-selected',
    multiple=True,
    help='Columns to use for clustering (can be specified multiple times)'
)
@click.option(
    '--cluster-method',
    type=click.Choice(['elbow', 'silhouette', 'custom']),
    default='elbow',
    help='Method for determining number of clusters'
)
@click.option(
    '--cluster-value',
    type=int,
    help='Number of clusters (required if cluster-method is custom)'
)
@click.option(
    '--columns-to-delete',
    multiple=True,
    help='Columns to delete from dataset (can be specified multiple times)'
)
@click.option(
    '--save-clusters/--no-save-clusters',
    default=True,
    help='Whether to save cluster datasets'
)
@click.option(
    '--clusters-output-dir',
    default='data/data_cluster',
    help='Directory to save cluster datasets'
)
@click.option(
    '--models-dir',
    default='models',
    help='Directory to save models'
)
@click.option(
    '--results-dir',
    default='results',
    help='Directory to save results'
)
@click.option(
    '--target-column',
    help='Target column for regression modeling'
)
@click.option(
    '--verbose/--quiet',
    default=False,
    help='Enable verbose output'
)
def main(
    data_path: str,
    config: Optional[str],
    columns_selected: tuple,
    cluster_method: str,
    cluster_value: Optional[int],
    columns_to_delete: tuple,
    save_clusters: bool,
    clusters_output_dir: str,
    models_dir: str,
    results_dir: str,
    target_column: Optional[str],
    verbose: bool
):
    """
    Run geoclustering sensitivity analysis from the command line.
    
    This tool performs clustering analysis on building energy data,
    builds regression models, and conducts sensitivity analysis.
    
    Example usage:
    
    geoclustering-analysis --data-path data.csv --columns-selected col1 --columns-selected col2 --target-column target
    """
    
    # Set up logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting geoclustering sensitivity analysis...")
    
    # Load configuration if provided
    if config:
        config_manager = ConfigManager()
        config_data = config_manager.load_config(config)
        
        # Override CLI arguments with config file values if not provided via CLI
        if not columns_selected and 'columns_selected' in config_data:
            columns_selected = config_data['columns_selected']
        if not target_column and 'target_column' in config_data:
            target_column = config_data['target_column']
        # Add other config overrides as needed
    
    # Validate inputs
    if not columns_selected:
        raise click.ClickException("Must specify at least one column for clustering")
    
    if cluster_method == 'custom' and cluster_value is None:
        raise click.ClickException("Must specify cluster-value when using custom cluster method")
    
    # Create output directories
    os.makedirs(clusters_output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = GeoClusteringAnalyzer(
            data_path=data_path,
            columns_selected=list(columns_selected),
            cluster_method_custom=(cluster_method == 'custom'),
            cluster_value=cluster_value,
            cluster_method_stat=cluster_method if cluster_method != 'custom' else 'elbow',
            columns_to_delete=list(columns_to_delete),
            save_clusters=save_clusters,
            clusters_output_dir=clusters_output_dir,
            models_dir=models_dir,
            results_dir=results_dir
        )
        
        if verbose:
            click.echo("Loading and preprocessing data...")
        
        # Load data
        data = analyzer.load_data()
        click.echo(f"Loaded data with shape: {data.shape}")
        
        if verbose:
            click.echo("Performing clustering analysis...")
        
        # Perform clustering
        clusters = analyzer.fit_clustering()
        click.echo(f"Created {len(clusters['labels'].unique())} clusters")
        
        # Build regression models if target column specified
        if target_column:
            if verbose:
                click.echo("Building regression models...")
            
            models = analyzer.build_regression_models(target_column)
            click.echo(f"Built {len(models)} regression models")
            
            # Run basic sensitivity analysis
            if verbose:
                click.echo("Running sensitivity analysis...")
            
            # Define basic parameter variations for sensitivity analysis
            sensitivity_params = {
                col: {'min': data[col].min(), 'max': data[col].max(), 'steps': 10}
                for col in columns_selected
            }
            
            results = analyzer.run_sensitivity_analysis(sensitivity_params)
            click.echo("Sensitivity analysis completed")
            
            # Save results summary
            results_file = Path(results_dir) / "sensitivity_summary.txt"
            with open(results_file, 'w') as f:
                f.write("Geoclustering Sensitivity Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Data file: {data_path}\n")
                f.write(f"Number of samples: {data.shape[0]}\n")
                f.write(f"Number of features: {data.shape[1]}\n")
                f.write(f"Clustering columns: {list(columns_selected)}\n")
                f.write(f"Number of clusters: {len(clusters['labels'].unique())}\n")
                f.write(f"Target column: {target_column}\n")
                f.write(f"Models built: {len(models)}\n")
            
            click.echo(f"Results summary saved to: {results_file}")
        
        click.echo("Analysis completed successfully!")
        
    except Exception as e:
        raise click.ClickException(f"Analysis failed: {str(e)}")


@click.command()
@click.option(
    '--output-dir',
    default='.',
    help='Directory to create the example configuration file'
)
def create_config(output_dir: str):
    """Create an example configuration file."""
    
    config_template = {
        'data_path': 'path/to/your/data.csv',
        'columns_selected': ['column1', 'column2', 'column3'],
        'target_column': 'target_variable',
        'cluster_method_stat': 'elbow',
        'cluster_method_custom': False,
        'cluster_value': None,
        'columns_to_delete': ['unwanted_column1', 'unwanted_column2'],
        'save_clusters': True,
        'clusters_output_dir': 'data/data_cluster',
        'models_dir': 'models',
        'results_dir': 'results',
        'sensitivity_parameters': {
            'column1': {'min': 0, 'max': 100, 'steps': 10},
            'column2': {'min': -10, 'max': 10, 'steps': 20}
        }
    }
    
    config_path = Path(output_dir) / 'geoclustering_config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)
    
    click.echo(f"Example configuration file created at: {config_path}")
    click.echo("Edit this file with your specific parameters and use with --config option")


# Create a CLI group with multiple commands
@click.group()
def cli():
    """Geoclustering Sensitivity Analysis CLI Tools"""
    pass


cli.add_command(main, name='analyze')
cli.add_command(create_config, name='create-config')


if __name__ == '__main__':
    cli()