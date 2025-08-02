"""
Advanced model exploration and prediction tool
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from pathlib import Path

class ModelExplorer:
    """Advanced model exploration class"""
    
    def __init__(self):
        self.models = {}
        self.data = {}
        self.load_models_and_data()
    
    def load_models_and_data(self):
        """Load trained models and data"""
        try:
            # Load models
            self.models['titanic_rf'] = joblib.load('trained_models/titanic_random_forest.joblib')
            self.models['titanic_lr'] = joblib.load('trained_models/titanic_logistic_regression.joblib')
            self.models['housing_rf'] = joblib.load('trained_models/housing_random_forest.joblib')
            self.models['housing_lr'] = joblib.load('trained_models/housing_linear_regression.joblib')
            
            # Load data
            self.data['titanic'] = pd.read_csv('data/features/titanic_features.csv')
            self.data['housing'] = pd.read_csv('data/features/housing_features.csv')
            
            print("✅ Models and data loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models/data: {e}")
    
    def analyze_titanic_predictions(self):
        """Analyze Titanic model predictions"""
        print("🚢 TITANIC MODEL ANALYSIS")
        print("=" * 40)
        
        data = self.data['titanic']
        X = data.drop(['Survived'], axis=1)
        y = data['Survived']
        
        # Get predictions from both models
        rf_pred = self.models['titanic_rf'].predict(X)
        lr_pred = self.models['titanic_lr'].predict(X)
        rf_proba = self.models['titanic_rf'].predict_proba(X)[:, 1]
        lr_proba = self.models['titanic_lr'].predict_proba(X)[:, 1]
        
        # Create analysis DataFrame
        analysis = pd.DataFrame({
            'Actual': y,
            'RF_Prediction': rf_pred,
            'LR_Prediction': lr_pred,
            'RF_Probability': rf_proba,
            'LR_Probability': lr_proba,
            'Sex': data['Sex'] if 'Sex' in data.columns else 0,
            'Pclass': data['Pclass'] if 'Pclass' in data.columns else 1,
            'Age': data['Age'] if 'Age' in data.columns else data.filter(regex='Age').iloc[:, 0] if len(data.filter(regex='Age').columns) > 0 else 25
        })
        
        # Survival rates by different factors
        print("📊 Survival Analysis:")
        print(f"Overall survival rate: {y.mean():.1%}")
        
        if 'Sex' in data.columns:
            print("By Gender:")
            gender_survival = analysis.groupby('Sex')['Actual'].mean()
            for gender, rate in gender_survival.items():
                gender_name = 'Female' if gender == 1 else 'Male'
                print(f"  {gender_name}: {rate:.1%}")
        
        if 'Pclass' in data.columns:
            print("By Class:")
            class_survival = analysis.groupby('Pclass')['Actual'].mean()
            for pclass, rate in class_survival.items():
                print(f"  Class {pclass}: {rate:.1%}")
        
        # Model agreement analysis
        agreement = (analysis['RF_Prediction'] == analysis['LR_Prediction']).mean()
        print(f"\n🤝 Model Agreement: {agreement:.1%}")
        
        # Find interesting cases
        high_confidence_wrong = analysis[
            (analysis['Actual'] != analysis['RF_Prediction']) & 
            (analysis['RF_Probability'] > 0.8)
        ]
        
        if len(high_confidence_wrong) > 0:
            print(f"\n🔍 High-confidence wrong predictions: {len(high_confidence_wrong)}")
        
        return analysis
    
    def analyze_housing_predictions(self):
        """Analyze Housing model predictions"""
        print("\n🏠 HOUSING MODEL ANALYSIS")
        print("=" * 40)
        
        data = self.data['housing']
        X = data.drop(['MEDV'], axis=1)
        y = data['MEDV']
        
        # Get predictions from both models
        rf_pred = self.models['housing_rf'].predict(X)
        lr_pred = self.models['housing_lr'].predict(X)
        
        # Create analysis DataFrame
        analysis = pd.DataFrame({
            'Actual': y,
            'RF_Prediction': rf_pred,
            'LR_Prediction': lr_pred,
            'RF_Error': abs(y - rf_pred),
            'LR_Error': abs(y - lr_pred),
            'CRIM': data['CRIM'] if 'CRIM' in data.columns else 0,
            'RM': data['RM'] if 'RM' in data.columns else 6
        })
        
        # Price analysis
        print("📊 Price Analysis:")
        print(f"Average actual price: ${y.mean():.1f}k")
        print(f"Price range: ${y.min():.1f}k - ${y.max():.1f}k")
        print(f"Random Forest MAE: ${analysis['RF_Error'].mean():.2f}k")
        print(f"Linear Regression MAE: ${analysis['LR_Error'].mean():.2f}k")
        
        # Find best and worst predictions
        best_rf = analysis.loc[analysis['RF_Error'].idxmin()]
        worst_rf = analysis.loc[analysis['RF_Error'].idxmax()]
        
        print(f"\n🎯 Best RF prediction:")
        print(f"  Actual: ${best_rf['Actual']:.1f}k, Predicted: ${best_rf['RF_Prediction']:.1f}k")
        
        print(f"😬 Worst RF prediction:")
        print(f"  Actual: ${worst_rf['Actual']:.1f}k, Predicted: ${worst_rf['RF_Prediction']:.1f}k")
        
        return analysis
    
    def create_advanced_plots(self):
        """Create advanced visualization plots"""
        print("\n📈 Creating Advanced Visualizations...")
        
        Path("advanced_plots").mkdir(exist_ok=True)
        
        try:
            # Titanic analysis plots
            titanic_analysis = self.analyze_titanic_predictions()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Probability distribution
            ax1.hist(titanic_analysis['RF_Probability'], bins=20, alpha=0.7, label='Random Forest')
            ax1.hist(titanic_analysis['LR_Probability'], bins=20, alpha=0.7, label='Logistic Regression')
            ax1.set_title('Survival Probability Distribution')
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            
            # Model comparison scatter
            ax2.scatter(titanic_analysis['RF_Probability'], titanic_analysis['LR_Probability'], alpha=0.6)
            ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
            ax2.set_title('Model Probability Comparison')
            ax2.set_xlabel('Random Forest Probability')
            ax2.set_ylabel('Logistic Regression Probability')
            ax2.legend()
            
            # Confusion matrix for RF
            cm = confusion_matrix(titanic_analysis['Actual'], titanic_analysis['RF_Prediction'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='Blues')
            ax3.set_title('Random Forest Confusion Matrix')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            
            # Feature importance (if available)
            if hasattr(self.models['titanic_rf'], 'feature_importances_'):
                feature_names = self.data['titanic'].drop(['Survived'], axis=1).columns
                importances = self.models['titanic_rf'].feature_importances_
                top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
                
                features, importance_values = zip(*top_features)
                ax4.barh(range(len(features)), importance_values)
                ax4.set_yticks(range(len(features)))
                ax4.set_yticklabels(features)
                ax4.set_title('Top 10 Feature Importances')
                ax4.set_xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig('advanced_plots/titanic_advanced_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("✅ Titanic advanced plots saved!")
            
        except Exception as e:
            print(f"⚠️ Error creating Titanic plots: {e}")
        
        try:
            # Housing analysis plots
            housing_analysis = self.analyze_housing_predictions()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Actual vs Predicted scatter
            ax1.scatter(housing_analysis['Actual'], housing_analysis['RF_Prediction'], alpha=0.6, label='Random Forest')
            ax1.plot([housing_analysis['Actual'].min(), housing_analysis['Actual'].max()], 
                    [housing_analysis['Actual'].min(), housing_analysis['Actual'].max()], 'r--', label='Perfect Prediction')
            ax1.set_title('Actual vs Predicted Prices (RF)')
            ax1.set_xlabel('Actual Price ($1000s)')
            ax1.set_ylabel('Predicted Price ($1000s)')
            ax1.legend()
            
            # Residuals plot
            residuals = housing_analysis['Actual'] - housing_analysis['RF_Prediction']
            ax2.scatter(housing_analysis['RF_Prediction'], residuals, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Residuals Plot (RF)')
            ax2.set_xlabel('Predicted Price ($1000s)')
            ax2.set_ylabel('Residuals')
            
            # Error distribution
            ax3.hist(housing_analysis['RF_Error'], bins=20, alpha=0.7, color='orange')
            ax3.set_title('Prediction Error Distribution')
            ax3.set_xlabel('Absolute Error ($1000s)')
            ax3.set_ylabel('Frequency')
            
            # Model comparison
            ax4.scatter(housing_analysis['RF_Prediction'], housing_analysis['LR_Prediction'], alpha=0.6)
            min_pred = min(housing_analysis['RF_Prediction'].min(), housing_analysis['LR_Prediction'].min())
            max_pred = max(housing_analysis['RF_Prediction'].max(), housing_analysis['LR_Prediction'].max())
            ax4.plot([min_pred, max_pred], [min_pred, max_pred], 'r--', label='Perfect Agreement')
            ax4.set_title('Model Prediction Comparison')
            ax4.set_xlabel('Random Forest Prediction')
            ax4.set_ylabel('Linear Regression Prediction')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig('advanced_plots/housing_advanced_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("✅ Housing advanced plots saved!")
            
        except Exception as e:
            print(f"⚠️ Error creating Housing plots: {e}")
    
    def make_custom_predictions(self):
        """Make predictions on custom input"""
        print("\n🔮 CUSTOM PREDICTION EXAMPLES")
        print("=" * 40)
        
        # Titanic prediction example
        print("🚢 Titanic Prediction Example:")
        print("Predicting survival for a hypothetical passenger...")
        
        # Use the first row as template and modify
        titanic_data = self.data['titanic']
        sample_passenger = titanic_data.drop(['Survived'], axis=1).iloc[0:1].copy()
        
        # Make prediction
        rf_prob = self.models['titanic_rf'].predict_proba(sample_passenger)[0][1]
        lr_prob = self.models['titanic_lr'].predict_proba(sample_passenger)[0][1]
        
        print(f"Random Forest survival probability: {rf_prob:.3f}")
        print(f"Logistic Regression survival probability: {lr_prob:.3f}")
        print(f"Average prediction: {(rf_prob + lr_prob)/2:.3f}")
        
        # Housing prediction example
        print("\n🏠 Housing Prediction Example:")
        print("Predicting price for a hypothetical house...")
        
        housing_data = self.data['housing']
        sample_house = housing_data.drop(['MEDV'], axis=1).iloc[0:1].copy()
        
        rf_price = self.models['housing_rf'].predict(sample_house)[0]
        lr_price = self.models['housing_lr'].predict(sample_house)[0]
        
        print(f"Random Forest price prediction: ${rf_price:.1f}k")
        print(f"Linear Regression price prediction: ${lr_price:.1f}k")
        print(f"Average prediction: ${(rf_price + lr_price)/2:.1f}k")

def main():
    """Main exploration function"""
    print("🔍 ADVANCED MODEL EXPLORATION")
    print("=" * 50)
    
    explorer = ModelExplorer()
    
    # Analyze predictions
    explorer.analyze_titanic_predictions()
    explorer.analyze_housing_predictions()
    
    # Create advanced visualizations
    explorer.create_advanced_plots()
    
    # Make custom predictions
    explorer.make_custom_predictions()
    
    print("\n" + "=" * 50)
    print("🎉 ADVANCED ANALYSIS COMPLETED!")
    print("📁 Check advanced_plots/ directory for detailed visualizations")

if __name__ == "__main__":
    main()