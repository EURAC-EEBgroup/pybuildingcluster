"""
Geoclustering Sensitivity Analysis

A Python library for sensitivity analysis of building clusters, evaluating refurbishment scenarios.

This library provides tools for:
- Clustering energy performance data using K-means and other algorithms
- Building regression models with Random Forest, XGBoost, and LightGBM
- Conducting sensitivity analysis using scenario analysis methods
- Optimizing cluster parameters using Optuna

Authors: Daniele Antonucci, Olga Somova
Organization: EURAC Research - Energy Efficient Buildings Group
License: MIT
"""


# Import core modules with error handling
_import_errors = []
try:
    from.core.clustering import ClusteringAnalyzer
except ImportError as e:
    print(f"Warning: Could not import ClusteringAnalyzer: {e}")
    ClusteringAnalyzer = None

try:
    from .core.regression import RegressionModelBuilder
except ImportError as e:
    print(f"Warning: Could not import RegressionModelBuilder: {e}")
    RegressionModelBuilder = None

try:
    from .core.sensitivity import SensitivityAnalyzer
except ImportError as e:
    print(f"Warning: Could not import SensitivityAnalyzer: {e}")
    SensitivityAnalyzer = None

try:
    from .core.optimization import ParameterOptimizer
except ImportError as e:
    print(f"Warning: Could not import ParameterOptimizer: {e}")
    ParameterOptimizer = None

try:
    from .core.models import PredictiveModel
except ImportError as e:
    print(f"Warning: Could not import PredictiveModel: {e}")
    PredictiveModel = None

try:
    from .data.loader import DataLoader
except ImportError as e:
    print(f"Warning: Could not import DataLoader: {e}")
    DataLoader = None

__version__ = "0.1.0"
__author__ = "Daniele Antonucci"
__email__ = "daniele.antonucci@eurac.edu"
__license__ = "MIT"

# Package metadata
__package_info__ = {
    "name": "pybuildingcluster",
    "version": __version__,
    "description": "Clustering and sensitivity analysis for building energy performance data",
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/EURAC-EEBgroup/pybuildingcluster",
    "keywords": ["clustering", "sensitivity analysis", "energy efficiency", "buildings", "machine learning"],
    "python_requires": ">=3.8"
}

def get_version():
    """Return the version string."""
    return __version__

def get_package_info():
    """Return package information dictionary."""
    return __package_info__.copy()

import pandas as pd
from pathlib import Path
from typing import Dict, List

class GeoClusteringAnalyzer:
    """
    Classe principale per l'analisi integrata di clustering, regressione e sensibilità.
    
    Workflow completo:
    1. Caricamento e pulizia dati
    2. Clustering ottimizzato
    3. Modelli di regressione per cluster
    4. Analisi di sensibilità e scenari
    5. Report HTML automatici
    
    Parameters
    ----------
    data_path : str
        Percorso al file CSV dei dati
    feature_columns_clustering : list
        Colonne per clustering (es: ['QHnd', 'degree_days'])
    feature_columns_regression : list
        Colonne per regressione (tutte meno target e variabili escluse)
    target_column : str, default='QHnd'
        Variabile target per regressione
    random_state : int, default=42
        Random state per riproducibilità
    output_dir : str, default='./results'
        Directory per salvare risultati
    """
    
    def __init__(
        self,
        data_path: str,
        feature_columns_clustering: List[str],
        feature_columns_regression: List[str] = None,
        target_column: str = 'QHnd',
        random_state: int = 42,
        output_dir: str = './results',
        user_features: List[str] = None,
    ):
        # Importa componenti pybuildingcluster
        # try:
        #     import pybuildingcluster as pbui
        #     self.pbui = pbui
        # except ImportError:
        #     raise ImportError("pybuildingcluster non trovato. Installare con: pip install pybuildingcluster")
        
        # Configurazione
        self.config = {
            'data_path': data_path,
            'feature_columns_clustering': feature_columns_clustering,
            'feature_columns_regression': feature_columns_regression,
            'target_column': target_column,
            'random_state': random_state,
            'output_dir': Path(output_dir),
            'user_features': user_features
        }
        
        # Crea directory output
        self.config['output_dir'].mkdir(exist_ok=True)
        
        # Inizializza componenti
        self.clustering_analyzer = ClusteringAnalyzer(random_state=random_state)
        self.regression_builder = RegressionModelBuilder(random_state=random_state, problem_type="regression")
        self.sensitivity_analyzer = SensitivityAnalyzer(random_state=random_state)
        
        # Storage risultati
        self.data = None
        self.clusters = None
        self.models = None
        self.scenarios = None
        self.results = {}
        
        print(f"🚀 GeoClusteringAnalyzer inizializzato")
        print(f"📂 Output directory: {self.config['output_dir'].absolute()}")
    
    def load_and_clean_data(self, columns_to_remove: List[str] = None) -> pd.DataFrame:
        """
        Carica e pulisce i dati.
        
        Parameters
        ----------
        columns_to_remove : list, optional
            Colonne da rimuovere dal dataset
            
        Returns
        -------
        pd.DataFrame
            Dataset pulito
        """
        print(f"📊 Caricamento dati da: {self.config['data_path']}")
        
        # Carica dati
        self.data = pd.read_csv(
            self.config['data_path'], 
            sep=",", 
            decimal=".", 
            low_memory=False, 
            header=0, 
            index_col=0
        )
        
        print(f"✅ Dataset caricato: {self.data.shape}")
        
        # Pulizia dati
        initial_rows = len(self.data)
        
        # Rimuovi colonne specificate
        if columns_to_remove:
            for col in columns_to_remove:
                if col in self.data.columns:
                    del self.data[col]
                    print(f"   • Rimossa colonna: {col}")
        
        # Rimuovi righe con caratteri problematici
        self.data = self.data[~self.data.apply(lambda row: row.astype(str).str.contains("\\n\\t\\t\\t\\t\\t\\t").any(), axis=1)]
        self.data = self.data[~self.data.apply(lambda row: row.astype(str).str.contains("\n").any(), axis=1)]
        self.data = self.data.reset_index(drop=True)
        
        # Auto-genera feature_columns_regression se non specificato
        if self.config['feature_columns_regression'] is None:
            exclude_cols = columns_to_remove + [self.config['target_column']]
            self.config['feature_columns_regression'] = [
                col for col in self.data.columns if col not in exclude_cols
            ]
        
        print(f"🧹 Dataset pulito: {self.data.shape} (rimosse {initial_rows - len(self.data)} righe)")
        print(f"📈 Feature clustering: {self.config['feature_columns_clustering']}")
        print(f"📊 Feature regressione: {len(self.config['feature_columns_regression'])} colonne")
        
        return self.data
    
    def perform_clustering(self, method: str = "silhouette", k_range: tuple = (2, 10)) -> Dict:
        """
        Esegue clustering ottimizzato.
        
        Parameters
        ----------
        method : str, default="silhouette"
            Metodo per determinare cluster ottimali ('elbow', 'silhouette', 'calinski_harabasz')
        k_range : tuple, default=(2, 10)
            Range di cluster da testare
            
        Returns
        -------
        dict
            Risultati clustering con dati e statistiche
        """
        if self.data is None:
            raise ValueError("Devi prima caricare i dati con load_and_clean_data()")
        
        print(f"🔍 Determinazione numero ottimale di cluster ({method})...")
        
        # Trova numero ottimale cluster
        df_cluster = self.data[self.config['feature_columns_clustering']]
        optimal_k = self.clustering_analyzer.determine_optimal_clusters(
            df_cluster, 
            method=method, 
            k_range=k_range, 
            plot=True
        )
        
        print(f"🎯 Numero ottimale cluster: {optimal_k}")
        
        # Esegue clustering
        print(f"⚙️ Esecuzione clustering...")
        self.clusters = self.clustering_analyzer.fit_predict(
            data=self.data,
            feature_columns=self.config['feature_columns_clustering'],
            n_clusters=optimal_k,
            algorithm="kmeans",
            save_clusters=True,
            output_dir=str(self.config['output_dir'])
        )
        
        # Statistiche cluster
        stats = self.clustering_analyzer.get_cluster_statistics(
            self.clusters['data_with_clusters'], 
            self.config['feature_columns_clustering']
        )
        
        self.results['clustering'] = {
            'optimal_k': optimal_k,
            'method': method,
            'statistics': stats,
            'data_with_clusters': self.clusters['data_with_clusters']
        }
        
        print(f"✅ Clustering completato: {optimal_k} cluster")
        print(f"📊 Statistiche salvate")
        
        return self.clusters
    
    def build_models(self, models_to_train: List[str] = None, hyperparameter_tuning: str = "none") -> Dict:
        """
        Costruisce modelli di regressione per ogni cluster.
        
        Parameters
        ----------
        models_to_train : list, optional
            Modelli da addestrare (default: ['random_forest'])
        hyperparameter_tuning : str, default="none"
            Tipo di tuning iperparametri
            
        Returns
        -------
        dict
            Modelli addestrati per ogni cluster
        """
        if self.clusters is None:
            raise ValueError("Devi prima eseguire clustering con perform_clustering()")
        
        if models_to_train is None:
            models_to_train = ['random_forest']
        
        print(f"🤖 Addestramento modelli: {models_to_train}")
        
        self.models = self.regression_builder.build_models(
            data=self.data,
            clusters=self.clusters,
            target_column=self.config['target_column'],
            feature_columns=self.config['feature_columns_regression'],
            models_to_train=models_to_train,
            hyperparameter_tuning=hyperparameter_tuning,
            save_models=True,
            user_features=self.config['user_features']
        )
        
        self.results['models'] = {
            'models_dict': self.models,
            'target_column': self.config['target_column'],
            'models_trained': models_to_train
        }
        
        print(f"✅ Modelli addestrati per {len(self.models)} cluster")
        
        return self.models
    
    def create_scenarios_from_cluster(self, cluster_id: int, 
                                     sensitivity_vars: List[str] = None,
                                     n_scenarios: int = 10) -> List[Dict]:
        """
        Crea scenari automaticamente basati sui limiti di un cluster.
        
        Parameters
        ----------
        cluster_id : int
            ID del cluster per estrarre limiti
        sensitivity_vars : list, optional
            Variabili per scenari (default: ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'])
        n_scenarios : int, default=10
            Numero di scenari da generare
            
        Returns
        -------
        list
            Lista di scenari in formato {'name': str, 'parameters': dict}
        """
        if self.clusters is None:
            raise ValueError("Devi prima eseguire clustering")
        
        if sensitivity_vars is None:
            sensitivity_vars = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        
        print(f"🎭 Creazione {n_scenarios} scenari dal cluster {cluster_id}")
        
        # Estrai dati cluster
        cluster_data = self.clusters['data_with_clusters'][
            self.clusters['data_with_clusters']['cluster'] == cluster_id
        ]
        
        if len(cluster_data) == 0:
            raise ValueError(f"Cluster {cluster_id} non trovato")
        
        # Calcola limiti per ogni variabile
        limits = {}
        for var in sensitivity_vars:
            if var in cluster_data.columns:
                limits[var] = {
                    'min': float(cluster_data[var].min()),
                    'max': float(cluster_data[var].max())
                }
        
        print(f"📊 Limiti estratti: {limits}")
        
        # Genera scenari strategici
        scenarios = []
        scenario_templates = [
            ("High Efficiency Optimal", [0.0, 0.0]),  # Minimi (migliore efficienza)
            ("Low Efficiency Worst", [1.0, 1.0]),     # Massimi (peggiore efficienza)
            ("Balanced Performance", [0.5, 0.5]),      # Valori medi
            ("Optimized Envelope", [0.2, 0.4]),       # Focus involucro
            ("Advanced Glazing", [0.4, 0.2]),         # Focus vetrate
            ("Conservative Upgrade", [0.3, 0.3]),      # Upgrade moderato
            ("Aggressive Efficiency", [0.1, 0.15]),    # Efficienza alta
            ("Current Market Standard", [0.6, 0.5]),   # Standard mercato
            ("Economic Retrofit", [0.45, 0.35]),       # Retrofit economico
            ("High Performance Realistic", [0.15, 0.25]) # Alta performance realistica
        ]
        
        for name, percentiles in scenario_templates[:n_scenarios]:
            parameters = {}
            for i, var in enumerate(sensitivity_vars):
                if var in limits:
                    min_val = limits[var]['min']
                    max_val = limits[var]['max']
                    value = min_val + (max_val - min_val) * percentiles[i]
                    parameters[var] = value
            
            scenarios.append({
                'name': name,
                'parameters': parameters
            })
        
        self.scenarios = scenarios
        print(f"✅ {len(scenarios)} scenari creati")
        
        return scenarios
    
    def run_sensitivity_analysis(self, cluster_id: int = None, 
                                sensitivity_vars: List[str] = None,
                                scenarios: List[Dict] = None,
                                n_points: int = 20) -> Dict:
        """
        Esegue analisi di sensibilità completa.
        
        Parameters
        ----------
        cluster_id : int, optional
            ID cluster per analisi (None = tutti i dati)
        sensitivity_vars : list, optional
            Variabili per analisi sensibilità
        scenarios : list, optional
            Scenari personalizzati (se None, crea automaticamente)
        n_points : int, default=20
            Punti per analisi sensibilità
            
        Returns
        -------
        dict
            Risultati completi analisi sensibilità
        """
        if self.models is None:
            raise ValueError("Devi prima addestrare i modelli con build_models()")
        
        if sensitivity_vars is None:
            sensitivity_vars = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
        
        # Crea scenari se non forniti
        if scenarios is None:
            if cluster_id is not None:
                scenarios = self.create_scenarios_from_cluster(cluster_id, sensitivity_vars)
            else:
                # Usa primo cluster come default
                scenarios = self.create_scenarios_from_cluster(1, sensitivity_vars)
        
        print(f"🔬 Analisi sensibilità in corso...")
        
        data_with_clusters = self.clusters['data_with_clusters']
        
        # Analisi sensibilità one-at-a-time
        print(f"📈 Analisi sensibilità parametrica...")
        sensitivity_results = self.sensitivity_analyzer.sensitivity_analysis(
            cluster_df=data_with_clusters,
            sensitivity_vars=sensitivity_vars,
            target=self.config['target_column'],
            modello=self.models[1]['best_model'],  # Usa modello cluster 1
            n_points=n_points,
            normalize_=True,
            plot_3d=False,
            cluster_id=cluster_id,
            feature_columns=self.config['feature_columns_regression']
        )
        
        # Analisi scenari
        print(f"🎭 Analisi scenari...")
        scenario_results = self.sensitivity_analyzer.compare_scenarios(
            cluster_df=data_with_clusters,
            scenarios=scenarios,
            target=self.config['target_column'],
            feature_columns=self.config['feature_columns_regression'],
            modello=self.models[1]['best_model']
        )
        
        # Grafici risultati
        print(f"📊 Generazione grafici...")
        self.sensitivity_analyzer._plot_scenario_results(scenario_results, self.config['target_column'])
        
        # Report HTML
        html_path = self.config['output_dir'] / f"scenario_analysis_report_{self.config['target_column']}.html"
        print(f"📄 Creazione report HTML...")
        
        self.sensitivity_analyzer.create_scenario_report_html(
            scenario_results, 
            scenarios, 
            self.config['target_column'], 
            self.config['feature_columns_regression'],
            output_path=str(html_path)
        )
        
        # Salva risultati
        self.results['sensitivity'] = {
            'sensitivity_analysis': sensitivity_results,
            'scenario_analysis': scenario_results,
            'scenarios_used': scenarios,
            'html_report_path': str(html_path),
            'cluster_analyzed': cluster_id
        }
        
        print(f"✅ Analisi sensibilità completata!")
        print(f"📄 Report HTML: {html_path}")
        
        return self.results['sensitivity']
    
    def run_complete_analysis(self, 
                            cluster_id: int = None,
                            clustering_method: str = "silhouette",
                            models_to_train: List[str] = None,
                            sensitivity_vars: List[str] = None,
                            scenarios: List[Dict] = None,
                            columns_to_remove: List[str] = None) -> Dict:
        """
        Esegue analisi completa end-to-end.
        
        Parameters
        ----------
        cluster_id : int, optional
            ID cluster per analisi sensibilità
        clustering_method : str, default="silhouette"
            Metodo clustering
        models_to_train : list, optional
            Modelli da addestrare
        sensitivity_vars : list, optional
            Variabili per analisi sensibilità
        scenarios : list, optional
            Scenari personalizzati
        columns_to_remove : list, optional
            Colonne da rimuovere
            
        Returns
        -------
        dict
            Tutti i risultati dell'analisi
        """
        print(f"🚀 ANALISI COMPLETA GEO-CLUSTERING")
        print(f"=" * 60)
        
        # 1. Carica e pulisci dati
        print(f"\n📊 STEP 1: Caricamento dati")
        self.load_and_clean_data(columns_to_remove)
        
        # 2. Clustering
        print(f"\n🔍 STEP 2: Clustering")
        self.perform_clustering(method=clustering_method)
        
        # 3. Modelli regressione
        print(f"\n🤖 STEP 3: Modelli regressione")
        self.build_models(models_to_train)
        
        # 4. Analisi sensibilità
        print(f"\n🔬 STEP 4: Analisi sensibilità")
        self.run_sensitivity_analysis(
            cluster_id=cluster_id,
            sensitivity_vars=sensitivity_vars,
            scenarios=scenarios
        )
        
        # Summary finale
        print(f"\n" + "=" * 60)
        print(f"🎉 ANALISI COMPLETA TERMINATA!")
        print(f"=" * 60)
        print(f"📊 Dataset: {self.data.shape}")
        print(f"🔍 Cluster: {self.results['clustering']['optimal_k']}")
        print(f"🤖 Modelli: {len(self.models)} cluster")
        print(f"🎭 Scenari: {len(self.scenarios) if self.scenarios else 0}")
        print(f"📄 Report: {self.results['sensitivity']['html_report_path']}")
        print(f"📁 Output: {self.config['output_dir'].absolute()}")
        
        return self.results
    
    def get_summary(self) -> Dict:
        """Restituisce un riassunto dei risultati."""
        if not self.results:
            return {"error": "Nessuna analisi eseguita"}
        
        summary = {
            "dataset_shape": self.data.shape if self.data is not None else None,
            "optimal_clusters": self.results.get('clustering', {}).get('optimal_k'),
            "models_trained": len(self.models) if self.models else 0,
            "scenarios_analyzed": len(self.scenarios) if self.scenarios else 0,
            "target_column": self.config['target_column'],
            "feature_columns_clustering": self.config['feature_columns_clustering'],
            "output_directory": str(self.config['output_dir']),
            "html_report": self.results.get('sensitivity', {}).get('html_report_path')
        }
        
        return summary


def check_dependencies():
    """
    Check which dependencies are available and print status.
    
    Returns
    -------
    dict
        Dictionary with availability status of each component
    """
    status = {
        'ClusteringAnalyzer': ClusteringAnalyzer is not None,
        'RegressionModelBuilder': RegressionModelBuilder is not None,
        'SensitivityAnalyzer': SensitivityAnalyzer is not None,
        'ParameterOptimizer': ParameterOptimizer is not None,
        'DataLoader': DataLoader is not None,
        'PredictiveModel': PredictiveModel is not None
    }
    
    print("🔍 Dependency Status:")
    for component, available in status.items():
        symbol = "✅" if available else "❌"
        print(f"   {symbol} {component}")
    
    if _import_errors:
        print("\n⚠️  Import Errors:")
        for error in _import_errors:
            print(f"   - {error}")
    
    return status


def get_available_components():
    """
    Get list of available components.
    
    Returns
    -------
    list
        List of available component names
    """
    components = []
    if ClusteringAnalyzer is not None:
        components.append('ClusteringAnalyzer')
    if RegressionModelBuilder is not None:
        components.append('RegressionModelBuilder')
    if SensitivityAnalyzer is not None:
        components.append('SensitivityAnalyzer')
    if ParameterOptimizer is not None:
        components.append('ParameterOptimizer')
    if PredictiveModel is not None:
        components.append('PredictiveModel')
    
    return components


# Export main classes and functions that are available
__all__ = ['GeoClusteringAnalyzer', 'check_dependencies', 'get_available_components']

# Add available components to __all__
available_components = get_available_components()
__all__.extend(available_components)

