

"""
Example of pybuildingcluster application
============================
"""

import pandas as pd
import pybuildingcluster as pbui
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Use environment variables
def setup_environment():
    """Setup environment with fallback if .env doesn't exist"""
    
    # Try to load .env
    env_loaded = load_dotenv()
    
    if not env_loaded:
        print("⚠️ File .env not found, creating the default configuration...")
        
        
        # Create file .env if 
        env_content = f"""# Configurazione Paths
CLUSTERING_CSV_PATH=.../data/clustering.csv
DATA_DIR=.../pybuildingcluster/data
RESULTS_DIR=.../pybuildingcluster/results
MODELS_DIR=.../pybuildingcluster/models
"""
        
        env_file = Path('.env')
        env_file.write_text(env_content.strip())
        print(f"✅ File .env created: {env_file.absolute()}")
        
        # Reload after creating the file
        load_dotenv()
    else:
        print("✅ File .env successfully loaded")

# Setup environment
setup_environment()

#%%
# ======= Utils =======
def feature_columns_regression(building_data):
    """Define feature columns for clustering and modeling."""
    feature_remove_regression = [
        "QHnd", "EPl", "EPt", "EPc", "EPv", "EPw", "EPh", 
        "QHimp", "theoric_nominal_power", "energy_vectors_used"
    ]
    feature_columns_df = building_data.columns
    feature_columns_regression = [
        item for item in feature_columns_df 
        if item not in feature_remove_regression
    ]
    return feature_columns_regression

def validate_data_path(file_path):
    """Valida che il file esista"""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File non trovato: {file_path}")
        
        # Suggerisci alternative
        possible_paths = [
            Path.cwd() / "clustering.csv",
            Path.cwd() / "data" / "clustering.csv",
            Path.cwd() / "src" / "pybuildingcluster" / "data" / "clustering.csv"
        ]
        
        print("🔍 Percorsi alternativi da verificare:")
        for alt_path in possible_paths:
            exists = alt_path.exists()
            print(f"  {'✅' if exists else '❌'} {alt_path}")
            if exists:
                return str(alt_path)
        
        raise FileNotFoundError(f"File clustering.csv non trovato in nessuna location")
    
    return file_path

# ======= Data Loading =======
def load_building_data():
    """Carica i dati degli edifici con gestione errori"""
    
    # Get path from .env
    csv_path = os.getenv('CLUSTERING_CSV_PATH')
    
    if not csv_path:
        print("⚠️ CLUSTERING_CSV_PATH not found in .env")
        # Fallback to hardcoded path
        csv_path = "/Users/dantonucci/Documents/gitLab/pybuildingcluster/src/pybuildingcluster/data/clustering.csv"
    
    print(f"📂 Loading data from: {csv_path}")
    
    # Validate that the file exists
    csv_path = validate_data_path(csv_path)
    
    try:
        # Carica il dataset
        df = pd.read_csv(
            csv_path, 
            sep=",", 
            decimal=".", 
            low_memory=False, 
            header=0, 
            index_col=0
        )
        
        print(f"✅ Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Data cleaning
        print("🧹 Data cleaning in progress...")
        
        # Remove energy_vectors_used column if exists
        if 'energy_vectors_used' in df.columns:
            del df['energy_vectors_used']
            print("   • energy_vectors_used column removed")
        
        # Remove rows with problematic characters
        initial_rows = len(df)
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
        df = df.reset_index(drop=True)
        
        cleaned_rows = len(df)
        removed_rows = initial_rows - cleaned_rows
        
        if removed_rows > 0:
            print(f"   • {removed_rows} rows removed with problematic characters")
        
        print(f"✅ Dataset cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# ======= Main Data Processing =======
print("🚀 Starting building cluster analysis...")
print("=" * 50)

# Load data
df = load_building_data()

# Subset for clustering 
building_data = df.copy()

# Feature columns for clustering
feature_columns = ['QHnd', 'degree_days']
print(f"📊 Feature for clustering: {feature_columns}")

# Feature columns for regression
feature_columns_regression_ = feature_columns_regression(building_data)
print(f"📈 Feature for regression: {len(feature_columns_regression_)} columns")
print(f"   First 10: {feature_columns_regression_[:10]}")

# Dataset information
print(f"\n📋 Dataset information:")
print(f"   • Forma: {building_data.shape}")
print(f"   • Missing values: {building_data.isnull().sum().sum()}")
print(f"   • Data types:")
for dtype in building_data.dtypes.value_counts().items():
    print(f"     - {dtype[0]}: {dtype[1]} columns")

# Verify output directory
results_dir = Path(os.getenv('RESULTS_DIR', './results'))
results_dir.mkdir(exist_ok=True)
print(f"📁 Directory results: {results_dir.absolute()}")

print("\n✅ Setup completed! Ready for cluster analysis.")


#%%
# ======= Data Exploration =======
def explore_dataset(df):
    """Quick dataset exploration"""
    
    print("\n🔍 ANALYSIS DATASET:")
    print("=" * 40)
    
    # Basic statistics
    print(f"📊 Basic statistics:")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Columns: {len(df.columns):,}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠️ Columns with missing values:")
        for col, count in missing_cols.head(10).items():
            pct = count / len(df) * 100
            print(f"   • {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"✅ No missing values")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_cols = ['EPh', 'QHnd', 'degree_days', 'net_area']
    available_key_cols = [col for col in key_cols if col in numeric_cols]
    
    if available_key_cols:
        print(f"\n📈 Statistics for key columns:")
        print(df[available_key_cols].describe().round(2))
    
    return df

# Explore dataset
building_data = explore_dataset(building_data)
# Feature columns for clustering
feature_columns = ['QHnd', 'degree_days']
# Feature columns for regression
feature_columns_regression_ = feature_columns_regression(building_data)

#%%
# ======= Clustering =======
# Data clustering
df_cluster = building_data[feature_columns]

# Get optimal number of clusters
optimal_k_elbow  = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="elbow", k_range=(2, 15), plot=True)
optimal_k_silhouette = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="silhouette", k_range=(2, 15), plot=True)
optimal_k_calinski_harabasz = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="calinski_harabasz", k_range=(2, 15), plot=True)
print(f"Optimal number of clusters: {optimal_k_elbow}")
print(f"Optimal number of clusters: {optimal_k_silhouette}")
print(f"Optimal number of clusters: {optimal_k_calinski_harabasz}")

clustering_analyzer = pbui.ClusteringAnalyzer(random_state=42)
clusters = clustering_analyzer.fit_predict(
    data=building_data,
    feature_columns=feature_columns,
    n_clusters=optimal_k_silhouette,
    algorithm="kmeans",
    save_clusters=True,
    output_dir="/Users/dantonucci/Documents/gitLab/pybuildingcluster/examples/example_results"
)

results = pbui.ClusteringAnalyzer(random_state=42).fit_predict(
            data=df,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
            output_dir="/Users/dantonucci/Documents/gitLab/pybuildingcluster/examples/example_results"
        )

#  Evaluate metrics
stats = pbui.ClusteringAnalyzer(random_state=42).get_cluster_statistics(
    results['data_with_clusters'], 
    feature_columns
)
print(stats)

# ======= Regression Models =======
regression_builder = pbui.RegressionModelBuilder(random_state=42, problem_type="regression")

models = regression_builder.build_models(
    data=building_data,
    clusters=clusters,
    target_column='QHnd',
    feature_columns=feature_columns_regression_,
    models_to_train=['random_forest'],
    hyperparameter_tuning="none",
    save_models=False
)

# ======= Sensitivity Analysis =======
# From cluster 1 get limits of average opaque surface transmittance and average glazed surface transmittance
cluster_1 = clusters['data_with_clusters'][clusters['data_with_clusters']['cluster'] == 1]
cluster_1_limits = {
    'average_opaque_surface_transmittance': {'min': float(cluster_1['average_opaque_surface_transmittance'].min()), 'max': float(cluster_1['average_opaque_surface_transmittance'].max())},
    'average_glazed_surface_transmittance': {'min': float(cluster_1['average_glazed_surface_transmittance'].min()), 'max': float(cluster_1['average_glazed_surface_transmittance'].max())}
}

sensitivity_analyzer = pbui.SensitivityAnalyzer(random_state=42)
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
    feature_columns=feature_columns_regression_
)

# ======= Scenario Analysis =======
# Get scenarios generated by generate_scenario.py

import ast
from io import StringIO
csv_data = "/Users/dantonucci/Documents/gitLab/pybuildingcluster/examples/example_results/list_dict_scenarios.csv"
df = pd.read_csv(csv_data)

def convert_csv_to_scenarios_safe(df):
    """Converte il DataFrame in lista di scenari usando ast.literal_eval"""
    scenarios = []
    
    for _, row in df.iterrows():
        try:
            # Usa ast.literal_eval per convertire la stringa in dizionario
            parameters_dict = ast.literal_eval(row['parameters'])
            
            scenario = {
                'name': row['name'],
                'parameters': parameters_dict
            }
            scenarios.append(scenario)
            
        except Exception as e:
            print(f"⚠️ Errore nella conversione di {row['name']}: {e}")
    
    return scenarios

scenarios = convert_csv_to_scenarios_safe(df)

scenario_results = sensitivity_analyzer.compare_scenarios(
    cluster_df=data_with_clusters,
    scenarios=scenarios,
    target='QHnd',
    feature_columns=feature_columns_regression_,
    modello=models[1]['best_model']
)
# EVALUATE SCENARIOS AND CREATE REPORT 
sensitivity_analyzer._plot_scenario_results(scenario_results, 'QHnd')
sensitivity_analyzer.create_scenario_report_html(
    scenario_results, 
    scenarios, 
    'QHnd', 
    feature_columns_regression_,
    output_path = "/Users/dantonucci/Documents/gitLab/pybuildingcluster/examples/example_results/scenario_analysis_report.html")

