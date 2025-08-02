"""
Simplified model training script that works reliably on Windows
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
from pathlib import Path
import os

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
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    # Create models directory
    Path("trained_models").mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_filename = f"trained_models/titanic_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_filename)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'model': model,
            'predictions': y_pred,
            'model_file': model_filename
        }
        
        print(f"✅ {model_name} - Accuracy: {accuracy:.4f}")
        print(f"💾 Model saved to: {model_filename}")
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
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    # Create models directory
    Path("trained_models").mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        model_filename = f"trained_models/housing_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_filename)
        
        # Store results
        results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model': model,
            'predictions': y_pred,
            'model_file': model_filename
        }
        
        print(f"✅ {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"💾 Model saved to: {model_filename}")
    
    return results

def save_results_summary(titanic_results, housing_results):
    """Save training results summary"""
    summary = []
    
    summary.append("# Model Training Results Summary\n")
    summary.append(f"Generated on: {pd.Timestamp.now()}\n\n")
    
    summary.append("## Titanic Classification Results\n")
    for model_name, results in titanic_results.items():
        summary.append(f"### {model_name}\n")
        summary.append(f"- **Accuracy**: {results['accuracy']:.4f}\n")
        summary.append(f"- **Model File**: {results['model_file']}\n\n")
    
    summary.append("## Housing Regression Results\n")
    for model_name, results in housing_results.items():
        summary.append(f"### {model_name}\n")
        summary.append(f"- **RMSE**: {results['rmse']:.4f}\n")
        summary.append(f"- **R² Score**: {results['r2']:.4f}\n")
        summary.append(f"- **Model File**: {results['model_file']}\n\n")
    
    # Save to file
    with open("trained_models/training_summary.md", "w") as f:
        f.writelines(summary)
    
    print("📄 Training summary saved to: trained_models/training_summary.md")

def test_trained_models():
    """Test the trained models with sample predictions"""
    print("\n🧪 Testing Trained Models...")
    
    try:
        # Test Titanic model
        titanic_rf = joblib.load("trained_models/titanic_random_forest.joblib")
        titanic_data = pd.read_csv('data/features/titanic_features.csv')
        X_titanic = titanic_data.drop(['Survived'], axis=1)
        
        # Make a sample prediction
        sample_prediction = titanic_rf.predict(X_titanic.iloc[:1])
        sample_proba = titanic_rf.predict_proba(X_titanic.iloc[:1])
        
        print(f"✅ Titanic model test:")
        print(f"   Sample prediction: {'Survived' if sample_prediction[0] == 1 else 'Did not survive'}")
        print(f"   Survival probability: {sample_proba[0][1]:.3f}")
        
    except Exception as e:
        print(f"⚠️ Titanic model test failed: {e}")
    
    try:
        # Test Housing model
        housing_rf = joblib.load("trained_models/housing_random_forest.joblib")
        housing_data = pd.read_csv('data/features/housing_features.csv')
        X_housing = housing_data.drop(['MEDV'], axis=1)
        
        # Make a sample prediction
        sample_prediction = housing_rf.predict(X_housing.iloc[:1])
        actual_price = housing_data['MEDV'].iloc[0]
        
        print(f"✅ Housing model test:")
        print(f"   Predicted price: ${sample_prediction[0]:.1f}k")
        print(f"   Actual price: ${actual_price:.1f}k")
        print(f"   Difference: ${abs(sample_prediction[0] - actual_price):.1f}k")
        
    except Exception as e:
        print(f"⚠️ Housing model test failed: {e}")

def main():
    """Main function to train all models"""
    print("🤖 Starting Simplified Model Training...")
    print("=" * 50)
    
    try:
        # Train Titanic models
        titanic_results = train_titanic_models()
        
        print("\n" + "=" * 50)
        
        # Train Housing models
        housing_results = train_housing_models()
        
        print("\n" + "=" * 50)
        
        # Save results summary
        save_results_summary(titanic_results, housing_results)
        
        # Test models
        test_trained_models()
        
        print("\n" + "=" * 50)
        print("🎉 MODEL TRAINING COMPLETED!")
        print("=" * 50)
        
        print("\n📊 TITANIC RESULTS:")
        for model_name, results in titanic_results.items():
            print(f"  {model_name}: {results['accuracy']:.4f} accuracy")
        
        print("\n🏠 HOUSING RESULTS:")
        for model_name, results in housing_results.items():
            print(f"  {model_name}: {results['rmse']:.4f} RMSE, {results['r2']:.4f} R²")
        
        print(f"\n📁 Models saved in: trained_models/")
        print(f"📄 Summary report: trained_models/training_summary.md")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)