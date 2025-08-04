"""
Simple script to run data preprocessing and feature engineering
"""
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append('src')

def run_preprocessing():
    """Run data preprocessing step"""
    print("🔄 STEP 2: DATA PREPROCESSING")
    print("=" * 40)
    
    try:
        from data.data_loader import DataLoader
        from data.preprocessor import DataPreprocessor
        
        # Initialize components
        loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        # Process Titanic dataset
        print("🚢 Processing Titanic dataset...")
        titanic_data = loader.load_titanic()
        if not titanic_data.empty:
            titanic_clean = preprocessor.preprocess_titanic(titanic_data)
            preprocessor.save_preprocessed_data(titanic_clean, 'titanic')
            print(f"✅ Titanic preprocessing completed. Shape: {titanic_clean.shape}")
        else:
            print("❌ Titanic data not found")
            return False
        
        # Process Housing dataset
        print("🏠 Processing Housing dataset...")
        housing_data = loader.load_housing()
        if not housing_data.empty:
            housing_clean = preprocessor.preprocess_housing(housing_data)
            preprocessor.save_preprocessed_data(housing_clean, 'housing')
            print(f"✅ Housing preprocessing completed. Shape: {housing_clean.shape}")
        else:
            print("❌ Housing data not found")
            return False
        
        print("✅ Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_feature_engineering():
    """Run feature engineering step"""
    print("\n🔧 STEP 3: FEATURE ENGINEERING")
    print("=" * 40)
    
    try:
        import pandas as pd
        from features.feature_engineer import FeatureEngineer
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Engineer Titanic features
        print("🚢 Engineering Titanic features...")
        titanic_clean = pd.read_csv('data/processed/titanic_clean.csv')
        titanic_features = engineer.engineer_titanic_features(titanic_clean)
        engineer.save_features(titanic_features, 'titanic')
        print(f"✅ Titanic feature engineering completed. Shape: {titanic_features.shape}")
        
        # Engineer Housing features
        print("🏠 Engineering Housing features...")
        housing_clean = pd.read_csv('data/processed/housing_clean.csv')
        housing_features = engineer.engineer_housing_features(housing_clean)
        engineer.save_features(housing_features, 'housing')
        print(f"✅ Housing feature engineering completed. Shape: {housing_features.shape}")
        
        print("✅ Feature engineering completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run preprocessing pipeline"""
    print("🚀 RUNNING DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Check if raw data exists
    if not Path('data/raw/titanic.csv').exists():
        print("❌ Raw data not found. Please run: python scripts/download_data.py")
        return 1
    
    # Step 1: Data Preprocessing
    success1 = run_preprocessing()
    
    if not success1:
        print("❌ Preprocessing failed. Cannot continue.")
        return 1
    
    # Step 2: Feature Engineering
    success2 = run_feature_engineering()
    
    if not success2:
        print("❌ Feature engineering failed.")
        return 1
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 PREPROCESSING PIPELINE COMPLETED!")
    print("=" * 50)
    
    print("\n📁 Files created:")
    files_to_check = [
        'data/processed/titanic_clean.csv',
        'data/processed/housing_clean.csv',
        'data/features/titanic_features.csv',
        'data/features/housing_features.csv'
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
    
    print("\n🚀 Next step: Run model training")
    print("   python train_models_simple.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)