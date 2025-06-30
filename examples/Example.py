"""
Example of pyBuildingCluster
============================

This example demonstrates how to use pyBuildingCluster to perform clustering, modeling, sensitivity analysis and scenario-based evaluations.
"""

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
CLUSTERING_CSV_PATH=../synthetic_epc_cleaned.csv
DATA_DIR=../data
RESULTS_DIR=../results
MODELS_DIR=../models
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
            Path.cwd() / "synthetic_epc_cleanedcsv",
            Path.cwd() / "data" / "synthetic_epc_cleanedcsv",
            Path.cwd() / "src" / "pybuildingcluster" / "data" / "synthetic_epc_cleanedcsv"
        ]
        
        print("🔍 Percorsi alternativi da verificare:")
        for alt_path in possible_paths:
            exists = alt_path.exists()
            print(f"  {'✅' if exists else '❌'} {alt_path}")
            if exists:
                return str(alt_path)
        
        raise FileNotFoundError(f"File synthetic_epc_cleanedcsv non trovato in nessuna location")
    
    return file_path

# ======= Data Loading =======
def load_building_data():
    """Carica i dati degli edifici con gestione errori"""
    
    # Ottieni path dal .env
    csv_path = os.getenv('CLUSTERING_CSV_PATH')
    
    if not csv_path:
        print("⚠️ CLUSTERING_CSV_PATH non trovato nel .env")
        # Fallback al path hardcoded
        csv_path = "../src/pybuildingcluster/data/synthetic_epc_cleanedcsv"
    
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
df_cluster = building_data[feature_columns]

# Get optimal number of clusters
optimal_k_elbow  = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="elbow", k_range=(2, 15), plot=True)
optimal_k_silhouette = pbui.ClusteringAnalyzer().determine_optimal_clusters(df_cluster, method="silhouette", k_range=(2, 15), plot=True)
print(f"Optimal number of clusters: {optimal_k_elbow}")
print(f"Optimal number of clusters: {optimal_k_silhouette}")


clustering_analyzer = pbui.ClusteringAnalyzer(random_state=42)
clusters = clustering_analyzer.fit_predict(
    data=building_data,
    feature_columns=feature_columns,
    method="silhouette",
    n_clusters=optimal_k_silhouette,
    algorithm="kmeans",
    save_clusters=True,
    output_dir="../examples/example_results"
)

results = pbui.ClusteringAnalyzer(random_state=42).fit_predict(
            data=df,
            feature_columns=feature_columns,
            method="silhouette",
            algorithm="kmeans",
            save_clusters=True,
            output_dir="../examples/example_results"
        )

#  Evaluate metrics
stats = pbui.ClusteringAnalyzer(random_state=42).get_cluster_statistics(
    results['data_with_clusters'], 
    feature_columns
)
print(stats)

# ======= Regression Models =======

models = pbui.RegressionModelBuilder(random_state=42, problem_type="regression").build_models(
    data=building_data,
    clusters=clusters,
    target_column='QHnd',
    feature_columns=feature_columns_regression_,
    models_to_train=['random_forest','xgboost','lightgbm'],
    hyperparameter_tuning="none",
    save_models=False,
    user_features=['average_opaque_surface_transmittance','average_glazed_surface_transmittance']
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
    feature_columns=models[1]['feature_columns']
)


# # ======= Scenario Analysis =======
# Get scenarios generated by generate_scenario.py

import numpy as np
import pandas as pd

# Extract limits for better readability
opaque_min = cluster_1_limits['average_opaque_surface_transmittance']['min']
opaque_max = cluster_1_limits['average_opaque_surface_transmittance']['max']
glazed_min = cluster_1_limits['average_glazed_surface_transmittance']['min']
glazed_max = cluster_1_limits['average_glazed_surface_transmittance']['max']

print(f"Range opaque transmittance: {opaque_min:.3f} - {opaque_max:.3f}")
print(f"Range glazed transmittance: {glazed_min:.3f} - {glazed_max:.3f}")

# Generate 10 strategic scenarios for energy efficiency
list_dict_scenarios = [
    # 1. OPTIMAL SCENARIO - Maximum efficiency (minimum values)
    {
        'name': 'High Efficiency Optimal', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min,
            'average_glazed_surface_transmittance': glazed_min
        }
    },
    
    # 2. WORST SCENARIO - Minimum efficiency (maximum values)
    {
        'name': 'Low Efficiency Worst', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_max,
            'average_glazed_surface_transmittance': glazed_max
        }
    },
    
    # 3. BALANCED SCENARIO - Intermediate values
    {
        'name': 'Balanced Performance', 
        'parameters': {
            'average_opaque_surface_transmittance': (opaque_min + opaque_max) / 2,
            'average_glazed_surface_transmittance': (glazed_min + glazed_max) / 2
        }
    },
    
    # 4. ENVELOPE OPTIMIZED - Efficient envelope, standard glazed
    {
        'name': 'Optimized Envelope', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.2,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.4
        }
    },
    
    # 5. GLAZED ADVANCED - Efficient glazed, standard envelope  
    {
        'name': 'Advanced Glazing', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.4,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.2
        }
    },
    
    # 6. CONSERVATIVE SCENARIO - Moderate improvement
    {
        'name': 'Conservative Upgrade', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.3,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.3
        }
    },
    
    # 7. AGGRESSIVE SCENARIO - High efficiency
    {
        'name': 'Aggressive Efficiency', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.1,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.15
        }
    },
    
    # 8. CURRENT MARKET STANDARD - Typical market performance
    {
        'name': 'Current Market Standard', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.6,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.5
        }
    },
    
    # 9. ECONOMIC RETROFIT SCENARIO - Cost-effective improvement
    {
        'name': 'Economic Retrofit', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.45,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.35
        }
    },
    
    # 10. HIGH PERFORMANCE SCENARIO - Quasi ottimale ma realistico
    {
        'name': 'High Performance Realistic', 
        'parameters': {
            'average_opaque_surface_transmittance': opaque_min + (opaque_max - opaque_min) * 0.15,
            'average_glazed_surface_transmittance': glazed_min + (glazed_max - glazed_min) * 0.25
        }
    }
]


scenario_results = sensitivity_analyzer.compare_scenarios(
    cluster_df=data_with_clusters,
    scenarios=list_dict_scenarios,
    target='QHnd',
    feature_columns=models[0]['feature_columns'],
    modello=models[0]['best_model']
)
# EVALUATE SCENARIOS AND CREATE REPORT 
sensitivity_analyzer._plot_scenario_results(scenario_results, 'QHnd')
sensitivity_analyzer.create_scenario_report_html(
    scenario_results, 
    list_dict_scenarios, 
    'QHnd', 
    models[0]['feature_columns'],
    output_path = "../examples/example_results/scenario_analysis_report.html")

