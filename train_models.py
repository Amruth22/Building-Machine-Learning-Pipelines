"""
Simple model training script to extend the pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path

def train_titanic_models():
    """Train models for Titanic dataset"""
    print("🚢 Training Titanic Models...")
    
    # Load the engineered features
    data = pd.read_csv('data/features/titanic_features.csv')
    
    # Prepare features and target
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set MLflow experiment
    mlflow.set_experiment("Titanic_Model_Training")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"Titanic_{model_name.replace(' ', '_')}"):
            print(f"\n🔄 Training {model_name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("features_count", X_train.shape[1])
            
            # Log model
            mlflow.sklearn.log_model(model, f"titanic_{model_name.lower().replace(' ', '_')}")
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"✅ {model_name} - Accuracy: {accuracy:.4f}")
            print(f"📊 Classification Report:")
            print(classification_report(y_test, y_pred))
    
    return results

def train_housing_models():
    """Train models for Housing dataset"""
    print("🏠 Training Housing Models...")
    
    # Load the engineered features
    data = pd.read_csv('data/features/housing_features.csv')
    
    # Prepare features and target
    X = data.drop(['MEDV'], axis=1)
    y = data['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set MLflow experiment
    mlflow.set_experiment("Housing_Model_Training")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"Housing_{model_name.replace(' ', '_')}"):
            print(f"\n🔄 Training {model_name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("features_count", X_train.shape[1])
            
            # Log model
            mlflow.sklearn.log_model(model, f"housing_{model_name.lower().replace(' ', '_')}")
            
            # Store results
            results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'model': model,
                'predictions': y_pred
            }
            
            print(f"✅ {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return results

def main():
    """Main function to train all models"""
    print("🤖 Starting Model Training...")
    print("=" * 50)
    
    # Set MLflow tracking URI (Windows compatible)
    import os
    mlruns_path = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    
    try:
        # Train Titanic models
        titanic_results = train_titanic_models()
        
        print("\n" + "=" * 50)
        
        # Train Housing models
        housing_results = train_housing_models()
        
        print("\n" + "=" * 50)
        print("🎉 MODEL TRAINING COMPLETED!")
        print("=" * 50)
        
        print("\n📊 TITANIC RESULTS:")
        for model_name, results in titanic_results.items():
            print(f"  {model_name}: {results['accuracy']:.4f} accuracy")
        
        print("\n🏠 HOUSING RESULTS:")
        for model_name, results in housing_results.items():
            print(f"  {model_name}: {results['rmse']:.4f} RMSE, {results['r2']:.4f} R²")
        
        print(f"\n🔍 View results in MLflow UI: http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)