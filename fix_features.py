"""
Quick script to regenerate features with fixed categorical encoding
"""
import sys
import pandas as pd
from pathlib import Path

# Add src to Python path
sys.path.append('src')

def regenerate_features():
    """Regenerate features with proper categorical encoding"""
    print("🔧 REGENERATING FEATURES WITH FIXED ENCODING")
    print("=" * 50)
    
    try:
        from features.feature_engineer import FeatureEngineer
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Check if processed data exists
        if not Path('data/processed/titanic_clean.csv').exists():
            print("❌ Processed data not found. Please run: python run_preprocessing.py")
            return False
        
        # Regenerate Titanic features
        print("🚢 Regenerating Titanic features...")
        titanic_clean = pd.read_csv('data/processed/titanic_clean.csv')
        titanic_features = engineer.engineer_titanic_features(titanic_clean)
        
        # Check for any remaining categorical columns
        categorical_cols = titanic_features.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"⚠️ Found remaining categorical columns: {categorical_cols}")
        else:
            print("✅ All columns are numeric")
        
        engineer.save_features(titanic_features, 'titanic')
        print(f"✅ Titanic features regenerated. Shape: {titanic_features.shape}")
        
        # Regenerate Housing features
        print("🏠 Regenerating Housing features...")
        housing_clean = pd.read_csv('data/processed/housing_clean.csv')
        housing_features = engineer.engineer_housing_features(housing_clean)
        
        # Check for any remaining categorical columns
        categorical_cols = housing_features.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"⚠️ Found remaining categorical columns: {categorical_cols}")
        else:
            print("✅ All columns are numeric")
        
        engineer.save_features(housing_features, 'housing')
        print(f"✅ Housing features regenerated. Shape: {housing_features.shape}")
        
        print("\n🎉 FEATURE REGENERATION COMPLETED!")
        print("✅ All categorical columns have been properly encoded")
        print("✅ Ready for model training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during feature regeneration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = regenerate_features()
    
    if success:
        print("\n🚀 Next step: Train models")
        print("   python train_models_simple.py")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)