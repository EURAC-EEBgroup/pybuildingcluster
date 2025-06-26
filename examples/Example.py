
import pandas as pd
import pybuildingcluster as pbui
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili d'ambiente
def setup_environment():
    """Setup environment con fallback se .env non esiste"""
    
    # Prova a caricare .env
    env_loaded = load_dotenv()
    
    if not env_loaded:
        print("⚠️ File .env non trovato, creando configurazione di default...")
        
        # Crea file .env se non esiste
        env_content = f"""# Configurazione Paths
CLUSTERING_CSV_PATH=/Users/dantonucci/Documents/gitLab/pybuildingcluster/src/pybuildingcluster/data/clustering.csv
DATA_DIR=/Users/dantonucci/Documents/gitLab/pybuildingcluster/src/pybuildingcluster/data
RESULTS_DIR=/Users/dantonucci/Documents/gitLab/pybuildingcluster/results
MODELS_DIR=/Users/dantonucci/Documents/gitLab/pybuildingcluster/models
"""
        
        env_file = Path('.env')
        env_file.write_text(env_content.strip())
        print(f"✅ File .env creato: {env_file.absolute()}")
        
        # Ricarica dopo aver creato il file
        load_dotenv()
    else:
        print("✅ File .env caricato con successo")

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
    
    # Ottieni path dal .env
    csv_path = os.getenv('CLUSTERING_CSV_PATH')
    
    if not csv_path:
        print("⚠️ CLUSTERING_CSV_PATH non trovato nel .env")
        # Fallback al path hardcoded
        csv_path = "/Users/dantonucci/Documents/gitLab/pybuildingcluster/src/pybuildingcluster/data/clustering.csv"
    
    print(f"📂 Caricamento dati da: {csv_path}")
    
    # Valida che il file esista
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
        print("🧹 Pulizia dati in corso...")
        
        # Rimuovi colonna energy_vectors_used se esiste
        if 'energy_vectors_used' in df.columns:
            del df['energy_vectors_used']
            print("   • Rimossa colonna 'energy_vectors_used'")
        
        # Rimuovi righe con caratteri problematici
        initial_rows = len(df)
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
        df = df[~df.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
        df = df.reset_index(drop=True)
        
        cleaned_rows = len(df)
        removed_rows = initial_rows - cleaned_rows
        
        if removed_rows > 0:
            print(f"   • Rimosse {removed_rows} righe con caratteri problematici")
        
        print(f"✅ Dataset pulito: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        return df
        
    except Exception as e:
        print(f"❌ Errore nel caricamento dati: {e}")
        raise

# ======= Main Data Processing =======
print("🚀 Avvio analisi cluster edifici...")
print("=" * 50)

# Carica i dati
df = load_building_data()

# Subset for clustering 
building_data = df.copy()

# Feature columns for clustering
feature_columns = ['QHnd', 'degree_days']
print(f"📊 Feature per clustering: {feature_columns}")

# Feature columns for regression
feature_columns_regression_ = feature_columns_regression(building_data)
print(f"📈 Feature per regressione: {len(feature_columns_regression_)} colonne")
print(f"   Prime 10: {feature_columns_regression_[:10]}")

# Informazioni dataset
print(f"\n📋 Informazioni Dataset:")
print(f"   • Forma: {building_data.shape}")
print(f"   • Valori mancanti: {building_data.isnull().sum().sum()}")
print(f"   • Tipi di dati:")
for dtype in building_data.dtypes.value_counts().items():
    print(f"     - {dtype[0]}: {dtype[1]} colonne")

# Verifica directory per output
results_dir = Path(os.getenv('RESULTS_DIR', './results'))
results_dir.mkdir(exist_ok=True)
print(f"📁 Directory risultati: {results_dir.absolute()}")

print("\n✅ Setup completato! Pronto per l'analisi cluster.")


#%%
# ======= Data Exploration =======
def explore_dataset(df):
    """Esplorazione rapida del dataset"""
    
    print("\n🔍 ESPLORAZIONE DATASET:")
    print("=" * 40)
    
    # Statistiche base
    print(f"📊 Statistiche base:")
    print(f"   • Righe: {len(df):,}")
    print(f"   • Colonne: {len(df.columns):,}")
    print(f"   • Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Valori mancanti
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠️ Colonne con valori mancanti:")
        for col, count in missing_cols.head(10).items():
            pct = count / len(df) * 100
            print(f"   • {col}: {count:,} ({pct:.1f}%)")
    else:
        print(f"✅ Nessun valore mancante")
    
    # Colonne numeriche key
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_cols = ['EPh', 'QHnd', 'degree_days', 'net_area']
    available_key_cols = [col for col in key_cols if col in numeric_cols]
    
    if available_key_cols:
        print(f"\n📈 Statistiche colonne chiave:")
        print(df[available_key_cols].describe().round(2))
    
    return df

# Esplora il dataset
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

