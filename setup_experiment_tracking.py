"""
Complete MLflow Experiment Tracking Setup
"""
import os
import sys
import subprocess
from pathlib import Path
import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd
import joblib
import json

def fix_mlflow_compatibility():
    """Fix MLflow compatibility issues"""
    print("🔧 FIXING MLFLOW COMPATIBILITY")
    print("=" * 40)
    
    try:
        # Try to fix importlib-metadata issue
        subprocess.run([sys.executable, "-m", "pip", "install", "importlib-metadata<5.0"], 
                      capture_output=True, text=True)
        print("✅ Updated importlib-metadata")
        
        # Update setuptools
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"], 
                      capture_output=True, text=True)
        print("✅ Updated setuptools")
        
        return True
    except Exception as e:
        print(f"⚠️ Compatibility fix had issues: {e}")
        return False

def setup_mlflow_tracking():
    """Set up comprehensive MLflow tracking"""
    print("\n🧪 SETTING UP MLFLOW EXPERIMENT TRACKING")
    print("=" * 50)
    
    # Create MLflow directory structure
    mlflow_dir = Path("mlflow_tracking")
    mlflow_dir.mkdir(exist_ok=True)
    
    # Set tracking URI
    tracking_uri = f"file:///{mlflow_dir.absolute()}"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"📁 MLflow tracking URI: {tracking_uri}")
    
    # Create experiments
    experiments = [
        ("Titanic_Classification", "Titanic passenger survival prediction"),
        ("Housing_Regression", "Boston housing price prediction"),
        ("Model_Comparison", "Cross-dataset model comparison"),
        ("Feature_Engineering_Experiments", "Feature engineering impact analysis")
    ]
    
    for exp_name, description in experiments:
        try:
            experiment_id = mlflow.create_experiment(
                name=exp_name,
                artifact_location=str(mlflow_dir / "artifacts" / exp_name)
            )
            print(f"✅ Created experiment: {exp_name} (ID: {experiment_id})")
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                print(f"✅ Experiment already exists: {exp_name}")
            else:
                print(f"⚠️ Error creating experiment {exp_name}: {e}")
    
    return tracking_uri

def create_comprehensive_experiment_tracker():
    """Create a comprehensive experiment tracking system"""
    print("\n📊 CREATING COMPREHENSIVE EXPERIMENT TRACKER")
    print("=" * 55)
    
    tracker_code = '''"""
Comprehensive MLflow Experiment Tracker
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json
from datetime import datetime

class ComprehensiveExperimentTracker:
    """Advanced experiment tracking with MLflow"""
    
    def __init__(self, tracking_uri="file:///./mlflow_tracking"):
        """Initialize the experiment tracker"""
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        print(f"🧪 MLflow tracking URI: {tracking_uri}")
    
    def run_titanic_experiments(self):
        """Run comprehensive Titanic experiments"""
        print("🚢 Running Titanic Classification Experiments...")
        
        # Load data
        data = pd.read_csv('data/features/titanic_features.csv')
        X = data.drop(['Survived'], axis=1)
        y = data['Survived']
        
        # Set experiment
        mlflow.set_experiment("Titanic_Classification")
        
        # Define models to test
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Hyperparameter grids
        param_grids = {
            'RandomForest': [
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
                {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2, 5, 10]}
            ],
            'LogisticRegression': [
                {'C': [0.1, 1.0, 10.0]},
                {'C': [1.0], 'solver': ['liblinear', 'lbfgs']}
            ],
            'SVM': [
                {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf']},
                {'C': [1.0], 'kernel': ['linear', 'rbf']}
            ]
        }
        
        results = []
        
        for model_name, base_model in models.items():
            for i, param_grid in enumerate(param_grids[model_name]):
                for params in self._generate_param_combinations(param_grid):
                    
                    run_name = f"{model_name}_run_{i}_{len(results)}"
                    
                    with mlflow.start_run(run_name=run_name):
                        # Set model parameters
                        model = base_model.__class__(**params, random_state=42)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                        
                        # Calculate metrics
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1_score': f1_score(y_test, y_pred)
                        }
                        
                        if y_proba is not None:
                            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                        metrics['cv_mean'] = cv_scores.mean()
                        metrics['cv_std'] = cv_scores.std()
                        
                        # Log everything
                        mlflow.log_params(params)
                        mlflow.log_params({
                            'model_type': model_name,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'features_count': X.shape[1]
                        })
                        mlflow.log_metrics(metrics)
                        
                        # Save model
                        mlflow.sklearn.log_model(model, f"model_{model_name.lower()}")
                        
                        # Store results
                        result = {'model': model_name, 'run_name': run_name, **params, **metrics}
                        results.append(result)
                        
                        print(f"✅ {run_name}: Accuracy = {metrics['accuracy']:.4f}")
        
        return results
    
    def run_housing_experiments(self):
        """Run comprehensive Housing experiments"""
        print("🏠 Running Housing Regression Experiments...")
        
        # Load data
        data = pd.read_csv('data/features/housing_features.csv')
        X = data.drop(['MEDV'], axis=1)
        y = data['MEDV']
        
        # Set experiment
        mlflow.set_experiment("Housing_Regression")
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR()
        }
        
        # Hyperparameter grids
        param_grids = {
            'RandomForest': [
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
                {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2, 5]}
            ],
            'LinearRegression': [{}],  # No hyperparameters
            'SVR': [
                {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf']},
                {'C': [1.0], 'kernel': ['linear', 'rbf']}
            ]
        }
        
        results = []
        
        for model_name, base_model in models.items():
            for i, param_grid in enumerate(param_grids[model_name]):
                for params in self._generate_param_combinations(param_grid):
                    
                    run_name = f"{model_name}_run_{i}_{len(results)}"
                    
                    with mlflow.start_run(run_name=run_name):
                        # Set model parameters
                        if model_name == 'LinearRegression':
                            model = base_model
                        else:
                            model = base_model.__class__(**params, random_state=42)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        metrics = {
                            'mse': mean_squared_error(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2_score': r2_score(y_test, y_pred)
                        }
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        metrics['cv_r2_mean'] = cv_scores.mean()
                        metrics['cv_r2_std'] = cv_scores.std()
                        
                        # Log everything
                        mlflow.log_params(params)
                        mlflow.log_params({
                            'model_type': model_name,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'features_count': X.shape[1]
                        })
                        mlflow.log_metrics(metrics)
                        
                        # Save model
                        mlflow.sklearn.log_model(model, f"model_{model_name.lower()}")
                        
                        # Store results
                        result = {'model': model_name, 'run_name': run_name, **params, **metrics}
                        results.append(result)
                        
                        print(f"✅ {run_name}: R² = {metrics['r2_score']:.4f}")
        
        return results
    
    def _generate_param_combinations(self, param_grid):
        """Generate all parameter combinations"""
        if not param_grid:
            return [{}]
        
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def compare_experiments(self):
        """Compare experiments across datasets"""
        print("📊 Comparing Experiments...")
        
        mlflow.set_experiment("Model_Comparison")
        
        with mlflow.start_run(run_name="Cross_Dataset_Comparison"):
            # Load experiment results
            titanic_results = self._get_experiment_results("Titanic_Classification")
            housing_results = self._get_experiment_results("Housing_Regression")
            
            # Log comparison metrics
            if titanic_results:
                best_titanic = max(titanic_results, key=lambda x: x.get('accuracy', 0))
                mlflow.log_metric("best_titanic_accuracy", best_titanic.get('accuracy', 0))
                mlflow.log_param("best_titanic_model", best_titanic.get('model', 'Unknown'))
            
            if housing_results:
                best_housing = max(housing_results, key=lambda x: x.get('r2_score', 0))
                mlflow.log_metric("best_housing_r2", best_housing.get('r2_score', 0))
                mlflow.log_param("best_housing_model", best_housing.get('model', 'Unknown'))
            
            print("✅ Cross-dataset comparison completed")
    
    def _get_experiment_results(self, experiment_name):
        """Get results from an experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                return runs.to_dict('records')
        except:
            pass
        return []
    
    def generate_experiment_report(self):
        """Generate comprehensive experiment report"""
        print("📄 Generating Experiment Report...")
        
        report_lines = []
        report_lines.append("# MLflow Experiment Tracking Report\\n")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        for exp in experiments:
            if exp.name != "Default":
                report_lines.append(f"## {exp.name}\\n")
                
                # Get runs for this experiment
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                
                if not runs.empty:
                    report_lines.append(f"Total runs: {len(runs)}\\n")
                    
                    # Best run
                    if 'metrics.accuracy' in runs.columns:
                        best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
                        report_lines.append(f"Best accuracy: {best_run['metrics.accuracy']:.4f}\\n")
                    elif 'metrics.r2_score' in runs.columns:
                        best_run = runs.loc[runs['metrics.r2_score'].idxmax()]
                        report_lines.append(f"Best R²: {best_run['metrics.r2_score']:.4f}\\n")
                
                report_lines.append("\\n")
        
        # Save report
        with open("mlflow_experiment_report.md", "w") as f:
            f.writelines(report_lines)
        
        print("✅ Experiment report saved to mlflow_experiment_report.md")

def main():
    """Main function to run comprehensive experiments"""
    print("🧪 COMPREHENSIVE MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
    
    tracker = ComprehensiveExperimentTracker()
    
    # Run experiments
    titanic_results = tracker.run_titanic_experiments()
    housing_results = tracker.run_housing_experiments()
    
    # Compare experiments
    tracker.compare_experiments()
    
    # Generate report
    tracker.generate_experiment_report()
    
    print("\\n🎉 All experiments completed!")
    print("📊 Check mlflow_experiment_report.md for detailed results")

if __name__ == "__main__":
    main()
'''
    
    with open("comprehensive_experiment_tracker.py", "w", encoding='utf-8') as f:
        f.write(tracker_code)
    
    print("✅ Comprehensive experiment tracker created!")

def create_mlflow_ui_alternative():
    """Create alternative MLflow UI"""
    print("\n🌐 CREATING MLFLOW UI ALTERNATIVE")
    print("=" * 40)
    
    ui_code = '''"""
Alternative MLflow UI using Streamlit
"""
import streamlit as st
import mlflow
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("🧪 MLflow Experiment Dashboard")
    st.sidebar.title("Navigation")
    
    # Set tracking URI
    tracking_uri = "file:///./mlflow_tracking"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get experiments
    try:
        experiments = mlflow.search_experiments()
        exp_names = [exp.name for exp in experiments if exp.name != "Default"]
        
        if exp_names:
            selected_exp = st.sidebar.selectbox("Select Experiment", exp_names)
            
            # Get experiment details
            experiment = mlflow.get_experiment_by_name(selected_exp)
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if not runs.empty:
                st.header(f"Experiment: {selected_exp}")
                st.write(f"Total runs: {len(runs)}")
                
                # Show metrics
                metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
                if metric_cols:
                    st.subheader("Metrics Overview")
                    
                    for metric_col in metric_cols:
                        metric_name = metric_col.replace('metrics.', '')
                        fig = px.histogram(runs, x=metric_col, title=f"{metric_name} Distribution")
                        st.plotly_chart(fig)
                
                # Show runs table
                st.subheader("All Runs")
                display_cols = ['run_id', 'status', 'start_time'] + metric_cols[:5]
                display_cols = [col for col in display_cols if col in runs.columns]
                st.dataframe(runs[display_cols])
                
            else:
                st.write("No runs found for this experiment")
        else:
            st.write("No experiments found")
            
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        st.write("Make sure you have run some experiments first!")

if __name__ == "__main__":
    main()
'''
    
    with open("streamlit_mlflow_ui.py", "w", encoding='utf-8') as f:
        f.write(ui_code)
    
    print("✅ Alternative MLflow UI created!")
    print("   Run with: streamlit run streamlit_mlflow_ui.py")

def main():
    """Main setup function"""
    print("🧪 COMPLETE MLFLOW EXPERIMENT TRACKING SETUP")
    print("=" * 70)
    
    # Fix compatibility
    fix_mlflow_compatibility()
    
    # Setup tracking
    tracking_uri = setup_mlflow_tracking()
    
    # Create comprehensive tracker
    create_comprehensive_experiment_tracker()
    
    # Create UI alternative
    create_mlflow_ui_alternative()
    
    print("\n" + "=" * 70)
    print("🎉 MLFLOW EXPERIMENT TRACKING COMPLETED!")
    print("=" * 70)
    print("\n📋 What you can now do:")
    print("  • python comprehensive_experiment_tracker.py - Run all experiments")
    print("  • streamlit run streamlit_mlflow_ui.py - Alternative UI")
    print("  • mlflow ui --backend-store-uri file:///./mlflow_tracking - Try MLflow UI")

if __name__ == "__main__":
    main()