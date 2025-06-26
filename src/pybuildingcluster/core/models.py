import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

class PredictiveModel:
    def __init__(self, file_path=None, target_column=None, problem_type='classification'):
        
        """
        
        Initialization of the predictive model
        
        Args:
            file_path: Path of the dataset file
            target_column: Name of the target column
            problem_type: Type of problem ('classification' or 'regression')
        """
        self.file_path = file_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf') if problem_type == 'classification' else float('inf')
        self.best_score_r2 = float('-inf') if problem_type == 'classification' else float('inf')
        
    def load_data(self, file_path=None, target_column=None):
        """
        Load data from a CSV or Excel file
        
        Args:
            file_path: Path of the dataset file
            target_column: Name of the target column
        """
        if file_path:
            self.file_path = file_path
        if target_column:
            self.target_column = target_column
            
        if self.file_path:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
        else:
            # Use an example dataset if no file is provided
            from sklearn.datasets import load_iris, load_boston, fetch_california_housing
            if self.problem_type == 'classification':
                data = load_iris()
                self.df = pd.DataFrame(data=data.data, columns=data.feature_names)
                self.df['target'] = data.target
                self.target_column = 'target'
            else:  # regression
                data = fetch_california_housing()
                self.df = pd.DataFrame(data=data.data, columns=data.feature_names)
                self.df['target'] = data.target
                self.target_column = 'target'
                
        print(f"Dataset loaded: {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self.df
        
    def exploratory_analysis(self):
        """
        Perform a basic exploratory analysis of the data
        """
        if self.df is None:
            self.carica_dati()
            
        print("Informazioni sul dataset:")
        print(self.df.info())
        print("\nStatistiche descrittive:")
        print(self.df.describe())
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values per colonna:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values in the dataset.")
            
        # Visualizations
        if self.df.shape[1] <= 20:  # Limit correlation matrix for datasets with many features
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()
        
        # Target variable distributions
        if self.target_column in self.df.columns:
            plt.figure(figsize=(8, 6))
            if self.problem_type == 'classification':
                self.df[self.target_column].value_counts().plot(kind='bar')
                plt.title('Distribution of Classes')
            else:
                sns.histplot(self.df[self.target_column], kde=True)
                plt.title('Distribution of Target Variable')
            plt.show()
        
        return self.df.head()
    
    def preprocess_data_prediction(self, df):   
        """
        Preprocess data for prediction
        """
        if self.df is None:
            self.carica_dati()
            
        if self.target_column not in self.df.columns:
            raise ValueError(f"The target column '{self.target_column}' does not exist in the dataset")
            
        # Separate feature and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Define transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Apply preprocessing
        # X_processed = preprocessor.fit_transform(df) # da modficare con preprocessor.fit_transform(X)
        X_processed = X
                
        return X_processed
        

    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess data for training
        
        Args:
            test_size: Proportion of the dataset to use as test set
            random_state: Seed for reproducibility
        """
        if self.df is None:
            self.carica_dati()
            
        if self.target_column not in self.df.columns:
            raise ValueError(f"La colonna target '{self.target_column}' non esiste nel dataset")
            
        # Separate feature and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Split train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Apply preprocessing
        # self.X_train_processed = preprocessor.fit_transform(self.X_train)
        # self.X_test_processed = preprocessor.transform(self.X_test)

        self.X_train_processed = self.X_train
        self.X_test_processed = self.X_test
        
        print(f"Data preprocessed: {self.X_train_processed.shape[0]} training samples, {self.X_test_processed.shape[0]} test samples")
        print(f"Features after preprocessing: {self.X_train_processed.shape[1]}")
        
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train different models on the dataset
        """
        if self.X_train is None:
            self.preprocess_dati()
            
        # Define models based on the problem type
        if self.problem_type == 'classification':
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42),
                'LightGBM': lgb.LGBMClassifier(random_state=42)
            }
        else:  # regressione
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42)
            }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining model: {name}")
            model.fit(self.X_train_processed, self.y_train)
            
            # Cross-validation evaluation
            if self.problem_type == 'classification':
                cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5, scoring='accuracy')
                print(f"Accuracy in cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            else:
                try:
                    cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5, scoring='neg_root_mean_squared_error')
                    print(f"RMSE in cross-validation: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except:
                    pass
            
            # Test set predictions
            y_pred = model.predict(self.X_test_processed)
            
            # Performance evaluation
            if self.problem_type == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f"Accuracy sul test set: {accuracy:.4f}")
                
                # Update the best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_score_r2 = accuracy
                    self.best_model = name
            else:

                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                print(f"RMSE sul test set: {rmse:.4f}")
                print(f"MAE sul test set: {mae:.4f}")
                print(f"R² score: {r2:.4f}")
                
                # Update the best model (based on RMSE)
                if rmse < abs(self.best_score):
                    self.best_score = rmse
                    self.best_score_r2 = r2
                    self.best_model = name
            
            # Save the model in the dictionary
            self.models[name] = model
            
        print(f"\nBest model: {self.best_model}")
        if self.problem_type == 'classification':
            print(f"Best score (accuracy): {self.best_score:.4f}")
        else:
            print(f"Best score (RMSE): {self.best_score:.4f}")
            
        return self.models
    
    def optimize_hyperparameters(self, model_name=None):
        """
        Optimize hyperparameters for the specified model
        
        Args:
            model_name: Name of the model to optimize
        """
        if not model_name:
            model_name = self.best_model
            
        if not self.models:
            self.train_models()
            
        print(f"Hyperparameter optimization for model: {model_name}")
        
        # Define search parameters based on the model
        if model_name == 'RandomForest':
            if self.problem_type == 'classification':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_name == 'XGBoost':
            if self.problem_type == 'classification':
                model = xgb.XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        elif model_name == 'LightGBM':
            if self.problem_type == 'classification':
                model = lgb.LGBMClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 70],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                model = lgb.LGBMRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 70],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        else:
            raise ValueError(f"Modello '{model_name}' non supportato")
            
        # Define the scoring metric
        if self.problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'neg_root_mean_squared_error'
            
        # Execute GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_processed, self.y_train)
        
        # Extract results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update the model with the best parameters
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        
        # Evaluate the optimized model
        y_pred = best_model.predict(self.X_test_processed)
        
        if self.problem_type == 'classification':
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\nAccuracy of the optimized model on the test set: {accuracy:.4f}")
            
            # Detailed classification report
            print("\nClassification report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model_name + "_optimized"
        else:
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"\nRMSE of the optimized model on the test set: {rmse:.4f}")
            print(f"MAE of the optimized model on the test set: {mae:.4f}")
            print(f"R² score of the optimized model: {r2:.4f}")
            
            # Visualize predictions vs actual values
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            plt.xlabel('Actual values')
            plt.ylabel('Predictions')
            plt.title('Predictions vs Actual values')
            plt.show()
            
            if rmse < abs(self.best_score):
                self.best_score = rmse
                self.best_score_r2 = r2
                self.best_model = model_name + "_optimized"
                
        return best_model
    
    def feature_importance(self, model_name=None):
        """
        Visualize the feature importance for the specified model
        
        Args:
            model_name: Name of the model to analyze
        """
        if not model_name:
            model_name = self.best_model
            if "_optimized" in model_name:
                model_name = model_name.replace("_optimized", "")
                
        if not self.models:
            self.addestra_modelli()
            
        model = self.models[model_name]
        
        # Verify that the model has an attribute for feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # If we have a ColumnTransformer, we need to retrieve the feature names
            if isinstance(self.X_train, pd.DataFrame):
                feature_names = self.X_train.columns
            else:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
                
            # Create a DataFrame with the importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Visualize the top 20 features (or fewer if there are fewer features)
            n_features = min(20, len(feature_importance_df))
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(n_features))
            plt.title(f'Top {n_features} Feature Importance - {model_name}')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        else:
            print(f"The model {model_name} does not support feature importance visualization")
            return None
    
    def save_model(self, model, file_path=None, save_preprocessor=True):
        """
        Save the specified model and optionally the preprocessor to file
        
        Args:
            model_name: Name of the model to save
            file_path: Path where to save the model (without extension)
            save_preprocessor: Whether to save the preprocessor as well
        
        Returns:
            dict: Dictionary with the paths of the saved files
        """
        
        model_name=self.best_model
        # if not model_name:
        #     model_name = self.best_model
            
        # if not self.models:
        #     self.addestra_modelli()
            
        # if not file_path:
        #     # Crea una directory models se non esiste
        #     if not os.path.exists('models'):
        #         os.makedirs('models')
        #     file_path = f"models/{model_name}"
        
        # # Assicurati che la directory esista
        # output_dir = os.path.dirname(file_path)
        # if output_dir and not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
            
        # model = self.models[model_name]
        
        # Save the model
        model_path = f"{file_path}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model {model_name} saved as {model_path}")
        
        saved_files = {'model': model_path}
        
        # Save the preprocessor if requested
        if save_preprocessor and hasattr(self, 'preprocessor'):
            preprocessor_path = f"{file_path}_preprocessor.joblib"
            joblib.dump(self.preprocessor, preprocessor_path)
            print(f"Preprocessor saved as {preprocessor_path}")
            saved_files['preprocessor'] = preprocessor_path
        
        # Save the model metadata
        import json
        metadata = {
            'model_name': model_name,
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'feature_columns': list(self.X_train.columns),
            'performance': {
                'metric': 'accuracy' if self.problem_type == 'classification' else 'rmse',
                'value': float(self.best_score),
                'metric2': 'accuracy' if self.problem_type == 'classification' else 'r2',
                'value2': float(self.best_score_r2)
            },
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = f"{file_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        saved_files['metadata'] = metadata_path
        print(f"Model metadata saved as {metadata_path}")
    
        return saved_files

    def download_model(self, model_path, preprocessor_path=None):
        """
        Carica un modello salvato in precedenza
        
        Args:
            model_path: Percorso del file del modello
            preprocessor_path: Percorso del file del preprocessor
        
        Returns:
            model: Il modello caricato
        """
        import joblib
        
        # Load the model
        model = joblib.load(model_path)
        print(f"Model {model_name} loaded from {model_path}")
        
        # Load the preprocessor if provided
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded from {preprocessor_path}")
        
        # Add the model to the models dictionary
        model_name = os.path.basename(model_path).replace('_model.joblib', '')
        self.models[model_name] = model
        
        return model

    def predict(self, dati_input, best_model, apply_preprocessing=True, intervallo_confidenza=0.95):
        """
        Make predictions on new data with confidence interval
        
        Args:
            dati_input: DataFrame or array with data to make predictions
            best_model: Model to use for predictions
            apply_preprocessing: Whether to apply preprocessing to the data
            intervallo_confidenza: Livello di confidenza per l'intervallo (default: 0.95 per 95%)
            
        Returns:
            dict: Predictions of the model with confidence intervals
        """
        model = best_model
        
        # Prepare the input data
        if isinstance(dati_input, pd.DataFrame):
            # Verify that all necessary columns are present
            if hasattr(self, 'X_train') and isinstance(self.X_train, pd.DataFrame):
                missing_cols = set(self.X_train.columns) - set(dati_input.columns)
                if missing_cols:
                    raise ValueError(f"I dati di input mancano delle seguenti colonne: {missing_cols}")
            
            # Applica lo stesso preprocessing usato per i dati di training se richiesto
            if apply_preprocessing and hasattr(self, 'preprocessor'):
                print("Applicazione preprocessing ai dati...")
                dati_processati = self.preprocessor.transform(dati_input)
            else:
                # Se non è richiesto preprocessing o non abbiamo un preprocessor, 
                # assumiamo che i dati siano già formattati correttamente
                dati_processati = dati_input
        else:
            # Assume that the data is already preprocessed correctly
            dati_processati = dati_input
        
        # Make predictions
        predictions = model.predict(dati_processati)
        
        result = {'predictions': predictions}
        
        # For classification problems, also offer probabilities if the model supports it
        if self.problem_type == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(dati_processati)
            result['probabilities'] = probabilities
        
        # For regression problems, calculate confidence intervals
        if self.problem_type == 'regressione':
            # For models that support direct uncertainty calculation (e.g., some versions of Random Forest)
            if hasattr(model, 'estimators_'):
                # Make predictions with all ensemble estimators
                preds_per_estimator = np.array([tree.predict(dati_processati) for tree in model.estimators_])
                
                # Calculate mean and standard deviation of predictions
                mean_prediction = np.mean(preds_per_estimator, axis=0)
                std_prediction = np.std(preds_per_estimator, axis=0)
                
                # Calculate confidence interval using normal distribution
                import scipy.stats as st
                alpha = 1 - intervallo_confidenza
                z_score = st.norm.ppf(1 - alpha/2)  # Two-tailed z-score
                
                ci_lower = mean_prediction - z_score * std_prediction
                ci_upper = mean_prediction + z_score * std_prediction
                
                result['ci_lower'] = ci_lower
                result['ci_upper'] = ci_upper
                
            # For models XGBoost and LightGBM that support quantile regression
            elif isinstance(model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
                # For XGBoost/LightGBM we use an estimate based on prediction error on the training set
                if hasattr(self, 'y_train') and hasattr(self, 'X_train_processed'):
                    # Calculate error on the training set
                    y_train_pred = model.predict(self.X_train_processed)
                    errors = np.abs(self.y_train - y_train_pred)
                    
                    # Estimate error for a certain confidence level
                    error_percentile = np.percentile(errors, intervallo_confidenza * 100)
                    
                    # Apply this error to the new predictions
                    ci_lower = predictions - error_percentile
                    ci_upper = predictions + error_percentile
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
                else:
                    # Se non abbiamo dati di training, utilizziamo una stima standard
                    rmse = np.sqrt(np.mean((self.y_test - model.predict(self.X_test_processed))**2))
                    z_score = st.norm.ppf(1 - (1 - intervallo_confidenza)/2)
                    
                    ci_lower = predictions - z_score * rmse
                    ci_upper = predictions + z_score * rmse
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
            else:
                # For other models, we use an estimate based on RMSE
                if hasattr(self, 'y_test') and hasattr(self, 'X_test_processed'):
                    rmse = np.sqrt(mean_squared_error(self.y_test, model.predict(self.X_test_processed)))
                    z_score = st.norm.ppf(1 - (1 - intervallo_confidenza)/2)
                    
                    ci_lower = predictions - z_score * rmse
                    ci_upper = predictions + z_score * rmse
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
                
        return result

    def predict_from_file(self, file_path, model_name=None, target_column=None, intervallo_confidenza=0.95):
        """
        Load a new dataset from file and perform predictions with confidence interval
        
        Args:
            file_path: Path of the file with new data
            model_name: Name of the model to use
            target_column: Name of the target column in the new dataset (if present)
            intervallo_confidenza: Livello di confidenza per l'intervallo (default: 0.95 per 95%)
            
        Returns:
            DataFrame: DataFrame with the original data and predictions with confidence intervals
        """
        # Load the new dataset
        if file_path.endswith('.csv'):
            nuovo_df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            nuovo_df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato file non supportato. Utilizzare CSV o Excel.")
        
        print(f"New dataset loaded: {nuovo_df.shape[0]} rows and {nuovo_df.shape[1]} columns")
        
        # Separate target if present
        X_nuovo = nuovo_df
        y_nuovo = None
        
        if target_column and target_column in nuovo_df.columns:
            X_nuovo = nuovo_df.drop(columns=[target_column])
            y_nuovo = nuovo_df[target_column]
            print(f"Target column '{target_column}' found and separated")
        
        X_nuovo = self.preprocess_data_prediction(X_nuovo)
        
        # Make predictions with confidence interval
        predictions = self.predict(X_nuovo, model_name, intervallo_confidenza=intervallo_confidenza)
        
        # Create a new DataFrame with the original data and predictions
        result_df = nuovo_df.copy()
        
        # Add the main prediction
        result_df['predizione'] = predictions['predictions']
        
        # Add confidence interval for regression
        if self.problem_type == 'regressione' and 'ci_lower' in predictions and 'ci_upper' in predictions:
            result_df['predizione_min'] = predictions['ci_lower']
            result_df['predizione_max'] = predictions['ci_upper']
            
            # Calculate the width of the confidence interval
            result_df['ic_ampiezza'] = result_df['predizione_max'] - result_df['predizione_min']
            
            print(f"confidence interval at {intervallo_confidenza*100}% added to predictions")
        
        # Add columns for the probabilities of each class for classification
        if self.problem_type == 'classification' and 'probabilities' in predictions:
            for i, col in enumerate(sorted(np.unique(self.y_train))):
                result_df[f'prob_classe_{col}'] = predictions['probabilities'][:, i]
                
            # Add a measure of uncertainty based on the entropy of the probability distribution
            if 'probabilities' in predictions:
                from scipy.stats import entropy
                # Calculate the entropy of the probabilities (higher entropy = higher uncertainty)
                result_df['incertezza'] = [entropy(prob) for prob in predictions['probabilities']]
        
        # If we have the real target, calculate the metrics
        if y_nuovo is not None:
            if self.problem_type == 'classification':
                accuracy = accuracy_score(y_nuovo, result_df['predizione'])
                print(f"Accuracy sul nuovo dataset: {accuracy:.4f}")
                
                # Classification report
                print("\nClassification report:")
                print(classification_report(y_nuovo, result_df['predizione']))
                
                # Confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_nuovo, result_df['predizione'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
            else:
                rmse = np.sqrt(mean_squared_error(y_nuovo, result_df['predizione']))
                r2 = r2_score(y_nuovo, result_df['predizione'])
                print(f"RMSE sul nuovo dataset: {rmse:.4f}")
                print(f"R² score: {r2:.4f}")
                
                # Calculate how many times the real value is within the confidence interval
                if 'predizione_min' in result_df.columns and 'predizione_max' in result_df.columns:
                    dentro_ic = ((y_nuovo >= result_df['predizione_min']) & 
                                (y_nuovo <= result_df['predizione_max'])).mean()
                    print(f"Percentuale di valori reali all'interno dell'intervallo di confidenza: {dentro_ic*100:.2f}%")
                    
                    # Add a column that indicates if the real value is within the interval
                    result_df['dentro_ic'] = ((y_nuovo >= result_df['predizione_min']) & 
                                            (y_nuovo <= result_df['predizione_max']))
                
                # Visualize predictions vs real values with confidence interval
                plt.figure(figsize=(10, 6))
                plt.scatter(y_nuovo, result_df['predizione'], alpha=0.5, label='Predictions')
                
                # Add error bars for the confidence interval
                if 'predizione_min' in result_df.columns and 'predizione_max' in result_df.columns:
                    # Ordiniamo i punti per valore reale per una visualizzazione più pulita
                    y_sorted = np.sort(y_nuovo)
                    indices = np.argsort(y_nuovo)
                    
                    pred_sorted = result_df['predizione'].values[indices]
                    lower_sorted = result_df['predizione_min'].values[indices]
                    upper_sorted = result_df['predizione_max'].values[indices]
                    
                    # Plot degli intervalli di confidenza (mostriamo solo un sottoinsieme per chiarezza)
                    n_points = len(y_sorted)
                    step = max(1, n_points // 100)  # Mostra al massimo 100 intervalli
                    
                    for i in range(0, n_points, step):
                        plt.plot([y_sorted[i], y_sorted[i]], 
                                [lower_sorted[i], upper_sorted[i]], 
                                'r-', alpha=0.3)
                
                plt.plot([min(y_nuovo), max(y_nuovo)], [min(y_nuovo), max(y_nuovo)], 'k--', label='Ideal')
                plt.xlabel('Real values')
                plt.ylabel('Predictions')
                plt.title(f'Predictions vs Real values with {intervallo_confidenza*100}% confidence interval')
                plt.legend()
                plt.show()
                
                # Visualize also a distribution of the confidence interval widths
                if 'ic_ampiezza' in result_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(result_df['ic_ampiezza'], kde=True)
                    plt.title(f'Distribution of confidence interval widths {intervallo_confidenza*100}%')
                    plt.xlabel('Interval width')
                    plt.ylabel('Frequency')
                    plt.show()
        
        return result_df

    def export_predictions(self, predictions_df, output_path=None, format='csv'):
        """
        Export predictions to a CSV or Excel file
        
        Args:
            predictions_df: DataFrame with predictions
            output_path: Path where to save the file
            format: File format ('csv' or 'excel')
            
        Returns:
            str: Path of the saved file
        """
        if output_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.{format}"
        
        if format == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format == 'excel':
            predictions_df.to_excel(output_path, index=False)
        else:
            raise ValueError("Format not supported. Use 'csv' or 'excel'")
        
        print(f"Predictions exported as {output_path}")
        return output_path

    
        


    def sensitivity_analysis(self, cluster_df, sensitivity_vars, target, modello=None, 
                           n_punti=20, normalizza=True, plot_3d=False):
        """
        Analyze how the variation of selected parameters influences the target value within a cluster.
        
        Args:
            cluster_df: DataFrame containing the cluster data to analyze
            sensitivity_vars: List of parameters to vary in the sensitivity analysis
            target: Name of the target column to predict
            modello: Pre-trained model to use (if None, uses self.best_model)
            n_punti: Number of points to use in the variation interval for each parameter
            normalizza: If normalize the variation parameters (min-max)
            plot_3d: If generate 3D plot when varying 2 parameters
            
        Returns:
            DataFrame with the sensitivity analysis results
        """
        import itertools
        from tqdm.notebook import tqdm
        
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("No model available. Train a model first.")
        
        if len(sensitivity_vars) > 3:
            print("Warning: more than 3 parameters specified. "
                "The analysis may take a long time and the results may be difficult to visualize.")
        
        # Create a "base point" using the mean values of the cluster for all features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        # X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        # X_processed.columns = X_base.columns

        base_point = X_base.mean().to_dict()
        print(f"Base point (mean values of the cluster):")
        for k, v in base_point.items():
            if k in sensitivity_vars:
                print(f"  {k}: {v:.4f}")
        
        # Determine the range for the parameters to vary
        ranges = {}
        for param in sensitivity_vars:
            if param not in X_base.columns:
                raise ValueError(f"Parameter '{param}' not found in DataFrame")
            
            min_val = X_base[param].min()
            max_val = X_base[param].max()
            
            ranges[param] = np.linspace(min_val, max_val, n_punti)
            print(f"Range for {param}: {min_val:.4f} - {max_val:.4f}")
        
        # Prepare the data structure for the results
        if len(sensitivity_vars) == 1:
            # Monodimensional case
            param = sensitivity_vars[0]
            results = []
            
            for val in tqdm(ranges[param], desc=f"Variation of {param}"):
                # Create a test point based on the base point
                test_point = base_point.copy()
                test_point[param] = val
                
                # Create DataFrame for the point
                df_test = pd.DataFrame([test_point])
                
                # Preprocessing e predizione
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                pred_result = self.predict(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param: val,
                    'prediction': prediction,
                    'prediction_min': ci_lower,
                    'prediction_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # Visualization
            plt.figure(figsize=(10, 6))
            plt.plot(results_df[param], results_df['prediction'], 'b-', label='Prediction')
            
            if 'prediction_min' in results_df.columns and results_df['prediction_min'].notna().all():
                plt.fill_between(
                    results_df[param], 
                    results_df['prediction_min'], 
                    results_df['prediction_max'], 
                    alpha=0.2, 
                    color='blue', 
                    label='Confidence interval'
                )
            
            plt.xlabel(param)
            plt.ylabel(target)
            plt.title(f"Effect of variation of {param} on {target}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            
            # Calculate the sensitivity
            param_range = results_df[param].max() - results_df[param].min()
            pred_range = results_df['prediction'].max() - results_df['prediction'].min()
            
            if normalizza and param_range > 0 and pred_range > 0:
                # Normalized sensitivity (variation % of the output / variation % of the input)
                param_mid = results_df[param].median()
                pred_mid = results_df['prediction'].median()
                
                elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                print(f"Normalized sensitivity of {target} with respect to {param}: {elasticity:.4f}")
                print(f"Interpretation: A variation of 1% in {param} causes a variation of {elasticity:.4f}% in {target}")
        
        elif len(sensitivity_vars) == 2:
            # Bidimensional case
            param1, param2 = sensitivity_vars
            results = []
            
            for val1, val2 in tqdm(list(itertools.product(ranges[param1], ranges[param2])), 
                                desc=f"Variation of {param1} and {param2}"):
                # Create a test point
                test_point = base_point.copy()
                # test_point = X_processed.mean().to_dict()
                test_point[param1] = val1
                test_point[param2] = val2
                
                # Create DataFrame for the point
                df_test = pd.DataFrame([test_point])
                
                # Preprocessing and prediction
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                pred_result = self.predict(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param1: val1,
                    param2: val2,
                    'prediction': prediction,
                    'prediction_min': ci_lower,
                    'prediction_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # Visualization of the heatmap
            pivot_table = results_df.pivot_table(
                index=param1, 
                columns=param2, 
                values='prediction'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap='viridis', annot=False, cbar_kws={'label': target})
            plt.title(f"Effect of variation of {param1} and {param2} on {target}")
            plt.tight_layout()
            plt.show()
            
            # If requested, generate a 3D plot
            if plot_3d:
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                x = results_df[param1].values
                y = results_df[param2].values
                z = results_df['prediction'].values
                
                # Create a trisurf surface
                surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.7)
                
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_zlabel(target)   
                ax.set_title(f"3D response surface: Effect of {param1} and {param2} on {target}")
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=target)
                plt.tight_layout()
                plt.show()
                
            # Calculate partial derivatives (gradient)
            print("\nSensitivity analysis:")
            
            for param in [param1, param2]:
                other_param = param2 if param == param1 else param1
                other_param_med = results_df[other_param].median()
                
                # Filter the results with the other parameter close to the median
                filtered = results_df[np.isclose(results_df[other_param], other_param_med, rtol=0.1)]
                
                if len(filtered) >= 2:
                    param_range = filtered[param].max() - filtered[param].min()
                    pred_range = filtered['prediction'].max() - filtered['prediction'].min()
                    
                    if normalizza and param_range > 0 and pred_range > 0:
                        # Normalized sensitivity
                        param_mid = filtered[param].median()
                        pred_mid = filtered['prediction'].median()
                        
                        elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                        print(f"Normalized sensitivity of {target} with respect to {param} (with {other_param} ≈ {other_param_med:.4f}): {elasticity:.4f}")
        
        else:
            # Multidimensional case - here we perform a one-at-a-time analysis
            results = []
            
            for param in sensitivity_vars:
                print(f"\nSensitivity analysis for {param}:")
                
                for val in tqdm(ranges[param], desc=f"Variation of {param}"):
                    # Create a test point
                    # test_point = base_point.copy()
                    test_point = X_processed.mean().to_dict()
                    test_point[param] = val
                    
                    # Create DataFrame for the point
                    df_test = pd.DataFrame([test_point])
                    
                    # Preprocessing and prediction
                    X_processed = self.preprocess_data_prediction(df_test)
                    pred_result = self.predict(X_processed, modello)
                    
                    prediction = pred_result['predictions'][0]
                    ci_lower = pred_result.get('ci_lower', [None])[0]
                    ci_upper = pred_result.get('ci_upper', [None])[0]
                    
                    result_record = {'variated_parameter': param}
                    result_record.update({p: base_point[p] for p in sensitivity_vars})
                    result_record[param] = val
                    result_record['prediction'] = prediction
                    result_record['prediction_min'] = ci_lower
                    result_record['prediction_max'] = ci_upper
                    
                    results.append(result_record)
                
                # Visualize the graph for this parameter
                param_results = pd.DataFrame([r for r in results if r['variated_parameter'] == param])
                
                plt.figure(figsize=(10, 6))
                plt.plot(param_results[param], param_results['prediction'], 'b-', label='Prediction')
                
                if 'prediction_min' in param_results.columns and param_results['prediction_min'].notna().all():
                    plt.fill_between(
                        param_results[param], 
                        param_results['prediction_min'], 
                        param_results['prediction_max'], 
                        alpha=0.2, 
                        color='blue', 
                        label='Confidence interval'
                    )
                    
                plt.xlabel(param)
                plt.ylabel(target)
                plt.title(f"Effect of variation of {param} on {target}")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()
                
                # Calculate the sensitivity
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['prediction'].max() - param_results['prediction'].min()
                
                if normalizza and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['prediction'].median()
                    
                    elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                    print(f"Normalized sensitivity of {target} with respect to {param}: {elasticity:.4f}")
            
            # Create a summary table of sensitivity
            sensitivity_summary = []
            
            for param in sensitivity_vars:
                param_results = pd.DataFrame([r for r in results if r['variated_parameter'] == param])
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['prediction'].max() - param_results['prediction'].min()
                
                if normalizza and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['prediction'].median()
                    elasticity = abs((pred_range / pred_mid) / (param_range / param_mid))
                else:
                    elasticity = abs(pred_range / param_range) if param_range > 0 else 0
                    
                sensitivity_summary.append({
                    'parameter': param,
                    'elasticity': elasticity,
                    'absolute_variation': pred_range
                })
            
            sensitivity_df = pd.DataFrame(sensitivity_summary).sort_values('elasticity', ascending=False)
            print("\nSummary of sensitivity (ordered by importance):")
            print(sensitivity_df)
            
            # Bar plot of the sensitivity
            plt.figure(figsize=(10, 6))
            sns.barplot(x='parameter', y='elasticity', data=sensitivity_df)
            plt.title(f"Sensitivity of {target} with respect to the parameters")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        return results_df
    
    def optimize_cluster_parameters(self, cluster_df, sensitivity_vars, target, objective='min',
                             modello=None, vincoli=None, n_trials=100, verbose=True):
        """
        Optimizes the parameters within a cluster to maximize or minimize a target.
        
        Args:
            cluster_df: DataFrame containing the cluster data to be analyzed
            sensitivity_vars: List of parameters to optimize
            target: Name of the target column to optimize
            objective: 'min' to minimize the target, 'max' to maximize it
            modello: Model pre-trained to use (if None, uses self.best_model)
            vincoli: Dictionary with constraints for each parameter {'param': (min, max)}
            n_trials: Number of iterations for optimization
            verbose: Whether to show optimization progress
            
        Returns:
            dict: Optimized parameters and predicted target value
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("To use this function, install optuna: pip install optuna")
            return None
        
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("No model available. Train a model first.")
        
        # Create a base point using the mean values of the cluster for all features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        base_point = X_base.mean().to_dict()

        X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        X_processed.columns = X_base.columns
        
        # Determine the ranges for the parameters to optimize
        if vincoli is None:
            vincoli = {}
            for param in sensitivity_vars:
                if param not in X_base.columns:
                    raise ValueError(f"The parameter '{param}' is not present in the DataFrame")
                
                min_val = X_base[param].min()
                max_val = X_base[param].max()
                
                # Expand slightly the range
                range_size = max_val - min_val
                min_val = max(min_val - range_size * 0.1, 0 if X_base[param].min() >= 0 else min_val * 1.1)
                max_val = max_val + range_size * 0.1
                
                vincoli[param] = (min_val, max_val)
        
        # Define the objective function for Optuna
        def objective_func(trial):
            # Create a test point based on the base point
            test_point = base_point.copy()
            # test_point = X_processed.mean().to_dict()
            
            # Suggest values for each parameter
            for param in sensitivity_vars:
                min_val, max_val = vincoli[param]
                test_point[param] = trial.suggest_float(param, min_val, max_val)
            
            # Create DataFrame for the point
            df_test = pd.DataFrame([test_point])
            
            # Preprocessing e prediction
            # X_processed = self.preprocess_data_prediction(df_test)
            X_processed = df_test
            pred_result = self.predict(X_processed, modello)
            
            prediction = pred_result['predictions'][0]
            
            # If minimizing, return the prediction directly
            # If maximizing, return the negative of the prediction (Optuna minimizes)
            return prediction if objective == 'min' else -prediction
        
        # Configure and run the optimization
        sampler = TPESampler(seed=42)  # For reproducibility
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=verbose)
        
        # Get the best parameters
        best_params = study.best_params
        
        # Create a point with the best parameters
        optimized_point = base_point.copy()
        optimized_point.update(best_params)
        
        # Calculate the predicted target value
        optimized_df = pd.DataFrame([optimized_point])
        X_processed = self.preprocess_data_prediction(optimized_df)
        X_processed = optimized_df
        pred_result = self.predict(X_processed, modello)
        predicted_target = pred_result['predictions'][0]
        
        # Visualizza risultati
        if verbose:
            print(f"\nOptimization completed for {objective}imize {target}")
            print(f"Predicted {target}: {predicted_target:.4f}")
            print("\nOptimized parameters:")
            for param, value in best_params.items():
                orig_value = base_point[param]
                change_pct = ((value - orig_value) / orig_value) * 100 if orig_value != 0 else float('inf')
                print(f"  {param}: {value:.4f} (original: {orig_value:.4f}, variation: {change_pct:.2f}%)")
            
            # Plot the optimization history
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title(f"Optimization history for {objective}imize {target}")
            plt.tight_layout()
            plt.show()
            
            # Plot the parameter importance
            plt.figure(figsize=(10, 6))
            try:
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.title(f"Importance of parameters for {objective}imize {target}")
                plt.tight_layout()
                plt.show()
            except:
                print("Unable to calculate parameter importance")
        
        result = {
            'optimized_parameters': best_params,
            'predicted_target': predicted_target,
            'original_target': base_point.get(target, None),
            'complete_point': optimized_point,
            'study': study  # Return the study object for future analysis
        }
        
        return result

    def compare_scenarios(self, cluster_df, scenarios, target, modello=None):
        """
        Compares different parameter scenarios and their effect on the target.
        
        Args:
            cluster_df: DataFrame containing the cluster data to be analyzed
            scenari: List of dictionaries, each representing a scenario {'name': 'Scenario 1', 'parameters': {'param1': val1, ...}}
            target: Name of the target column to predict
            modello: Pre-trained model to use (if None, uses self.best_model)
            
        Returns:
            DataFrame: Comparison of scenarios with predicted target values
        """
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("No model available. Train a model first.")
        
        #   Creare un punto base usando i valori medi del cluster per tutte le features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        base_point = X_base.mean().to_dict()
        X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        X_processed.columns = X_base.columns

        # Prepare the base point as a scenario
        scenarios_con_base = [{'name': 'Base (mean cluster)', 'parameters': {}}] + scenarios
        
        # List for results
        results = []
        
        # Evaluate each scenario
        for scenario in scenarios_con_base:
            # Create a test point based on the base point
            test_point = base_point.copy()
            # test_point = X_processed.mean().to_dict()
            
            # Update with scenario parameters (for the base scenario, no changes are made)
            test_point.update(scenario['parameters'])
            
            # Create DataFrame for the point
            df_test = pd.DataFrame([test_point])
            
            # Preprocessing and prediction
            # X_processed = self.preprocess_data_prediction(df_test)
            X_processed = df_test
            pred_result = self.predict(X_processed, modello)
            
            prediction = pred_result['predictions'][0]
            ci_lower = pred_result.get('ci_lower', [None])[0]
            ci_upper = pred_result.get('ci_upper', [None])[0]
            
            # Result for this scenario
            result = {
                'scenario': scenario['name'],
                'prediction': prediction,
                'prediction_min': ci_lower,
                'prediction_max': ci_upper
            }
            
            # Add parameter values
            for param, value in test_point.items():
                if param in set().union(*[s['parameters'].keys() for s in scenarios if 'parameters' in s]):
                    result[f"param_{param}"] = value
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate percentage variations relative to the base scenario
        base_prediction = results_df.loc[results_df['scenario'] == 'Base (mean cluster)', 'prediction'].values[0]
        results_df['variation_pct'] = ((results_df['prediction'] - base_prediction) / base_prediction) * 100
        
        # Visualize the results
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='scenario', y='prediction', data=results_df)
        
        # Add error bars if available
        if 'prediction_min' in results_df.columns and results_df['prediction_min'].notna().all():
            for i, row in results_df.iterrows():
                ax.errorbar(
                    i, row['prediction'], 
                    yerr=[[row['prediction']-row['prediction_min']], [row['prediction_max']-row['prediction']]],
                    fmt='none', capsize=5, color='black', alpha=0.7
                )
        
        plt.title(f"Scenarios comparison - Effect on {target}")
        plt.ylabel(target)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values above the bars
        for i, v in enumerate(results_df['prediction']):
            var_pct = results_df['variation_pct'].iloc[i]
            var_text = f" ({var_pct:+.1f}%)" if i > 0 else ""
            ax.text(i, v + (results_df['prediction'].max() * 0.01), 
                    f"{v:.2f}{var_text}", 
                    ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Show also a table of results
        print("\nResults table:")
        display_df = results_df[['scenario', 'prediction', 'variation_pct']].copy()
        display_df['prediction'] = display_df['prediction'].round(4)
        display_df['variation_pct'] = display_df['variation_pct'].round(2)
        display_df = display_df.rename(columns={'variation_pct': 'variation (%)'})
        print(display_df)
        return results_df

    def pipeline(self, optimize=True, visualize_feature_importance=True, 
                           save_model=True, file_path_salvataggio=None,
                           predict_new_file=True, new_file_path=None, 
                           target_column_nuovo=None, export_prediction=False,
                           formato_export='csv', sensitivity_analysis_=False, sensitivity_vars=None,
                           target="QHnd",
                           optimize_cluster_parameters=False, objective='min',compare_scenarios=False,scenarios=None,
                           cluster_df=None,
                           ):
        """
        Execute the complete pipeline from start to finish, including model saving
        and prediction on new data
        
        Args:
            optimize: If optimize hyperparameters
            visualize_feature_importance: If visualize feature importance
            save_model: If save the best model
            file_path_salvataggio: Path where to save the model (without extension)
            predict_new_file: If perform predictions on a new data file
            new_file_path: Path of the file with new data for prediction
            target_column_nuovo: Name of the target column in the new dataset (if present)
            export_prediction: If export predictions to a file
            formato_export: Format of the file ('csv' or 'excel')
        
        Returns:
            tuple: (best model, best score, saved files, predictions)
        """
        self.load_data()
        self.exploratory_analysis()
        self.preprocess_data()
        self.train_models()
        
        
        if optimize:
            best_model = self.optimize_hyperparameters()
            
        if visualize_feature_importance:
            self.feature_importance()
        
        files_salvati = None
        if save_model:
            files_salvati = self.save_model(
                model=best_model, 
                file_path=file_path_salvataggio
            )
            print(f"\nModel saved: {files_salvati}")
        
        df_prediction = None
        if predict_new_file and new_file_path:
            print(f"\nPrediction on new file: {new_file_path}")
            df_prediction = self.predict_from_file(
                file_path=new_file_path,
                model_name=best_model,
                target_column=target_column_nuovo
            )
            
            if export_prediction and df_prediction is not None:
                export_path = self.export_predictions(
                    predictions_df=df_prediction,
                    format=formato_export
                )
                print(f"Predictions exported to: {export_path}")
            
        print("\nPipeline completed!")
        print(f"Best model: {self.best_model}")
        if self.problem_type == 'classification':
            print(f"Best score (accuracy): {self.best_score:.4f}")
        else:
            print(f"Best score (RMSE): {self.best_score:.4f}")
        
        if sensitivity_analysis_:
            results_sensitivity_analysis = self.sensitivity_analysis(
                cluster_df=cluster_df,
                sensitivity_vars=sensitivity_vars,
                target=target,
                plot_3d=True,
                modello=best_model
            )

        if optimize_cluster_parameters:
            self.optimize_cluster_parameters(
                cluster_df=cluster_df,
                sensitivity_vars=sensitivity_vars,
                target=target,
                objective='min',
                modello=best_model, 
                vincoli=None, n_trials=100, verbose=True
            )

        if compare_scenarios:
            df_compare_scenarios = self.compare_scenarios(
            cluster_df=cluster_df, 
            scenarios=scenarios,
            target=target, 
            modello=best_model
        )

        return self.best_model, self.best_score, files_salvati, df_prediction, results_sensitivity_analysis, df_compare_scenarios

def run_model(
    file_path_cluster, 
    target_, 
    problem_type_, 
    variables_for_sensitivity_analysis, 
    file_path_save_model,
    confronta_scenari_cluster,
    scenari,
    path_save_result,
    cluster_name):

    model_ = PredictiveModel(file_path=file_path_cluster, target_column=target_, problem_type=problem_type_)
    _,_,_,predictions_df,results_sensitivity_analysis,df_compare_scenarios = model_.pipeline(
        predict_new_file=False, 
        new_file_path=file_path_cluster, 
        target_column_nuovo=target_,
        file_path_salvataggio = file_path_save_model,
        sensitivity_analysis_=True,
        sensitivity_vars=variables_for_sensitivity_analysis,
        target=target_,
        compare_scenarios=confronta_scenari_cluster,
        scenarios=scenari,
        cluster_df = pd.read_csv(file_path_cluster, sep=',', decimal='.', low_memory=False, header=0),
    )
    # save results
    results_sensitivity_analysis.to_csv(f"{path_save_result}/results_sensitivity_analysis_{cluster_name}.csv", index=False)
    df_compare_scenarios.to_csv(f"{path_save_result}/df_compare_scenarios_{cluster_name}.csv", index=False)

    return predictions_df,results_sensitivity_analysis,df_compare_scenarios