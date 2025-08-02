"""
Complete DVC Data Versioning Setup
"""
import subprocess
import os
from pathlib import Path
import shutil

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"⚠️ {description} had issues: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def setup_dvc_data_versioning():
    """Set up complete DVC data versioning"""
    print("🔄 SETTING UP DVC DATA VERSIONING")
    print("=" * 50)
    
    # Check if DVC is initialized
    if not Path(".dvc").exists():
        print("🔧 Initializing DVC...")
        run_command("dvc init", "DVC initialization")
    else:
        print("✅ DVC already initialized")
    
    # Add datasets to DVC tracking
    datasets = [
        "data/raw/titanic.csv",
        "data/raw/housing.csv"
    ]
    
    for dataset in datasets:
        if Path(dataset).exists():
            print(f"\n📊 Adding {dataset} to DVC tracking...")
            
            # Remove from git if tracked
            run_command(f"git rm --cached {dataset}", f"Remove {dataset} from git")
            
            # Add to DVC
            success = run_command(f"dvc add {dataset}", f"Add {dataset} to DVC")
            
            if success:
                # Add .dvc file to git
                dvc_file = f"{dataset}.dvc"
                run_command(f"git add {dvc_file}", f"Add {dvc_file} to git")
                
                print(f"✅ {dataset} is now tracked by DVC")
            else:
                print(f"⚠️ Failed to add {dataset} to DVC")
        else:
            print(f"⚠️ Dataset {dataset} not found")
    
    # Add processed data to DVC
    processed_dirs = [
        "data/processed",
        "data/features",
        "trained_models"
    ]
    
    for dir_path in processed_dirs:
        if Path(dir_path).exists() and any(Path(dir_path).iterdir()):
            print(f"\n📁 Adding {dir_path} directory to DVC...")
            run_command(f"dvc add {dir_path}", f"Add {dir_path} to DVC")
            
            dvc_file = f"{dir_path}.dvc"
            if Path(dvc_file).exists():
                run_command(f"git add {dvc_file}", f"Add {dvc_file} to git")
    
    # Update .gitignore for DVC
    gitignore_content = """
# DVC tracked files
/data/raw/titanic.csv
/data/raw/housing.csv
/data/processed
/data/features
/trained_models
"""
    
    with open(".gitignore", "a") as f:
        f.write(gitignore_content)
    
    run_command("git add .gitignore", "Update .gitignore")
    
    print("\n✅ DVC data versioning setup completed!")
    return True

def create_dvc_pipeline():
    """Create a comprehensive DVC pipeline"""
    print("\n🔄 CREATING DVC PIPELINE")
    print("=" * 30)
    
    # Enhanced DVC pipeline
    dvc_pipeline = """stages:
  download_data:
    cmd: python scripts/download_data.py
    outs:
    - data/raw/titanic.csv
    - data/raw/housing.csv
    
  validate_data:
    cmd: python src/data/data_validator.py
    deps:
    - data/raw/titanic.csv
    - data/raw/housing.csv
    - src/data/data_validator.py
    outs:
    - data/validation/data_quality_report.json
    
  preprocess_titanic:
    cmd: python -c "
      import sys; sys.path.append('src');
      from data.data_loader import DataLoader;
      from data.preprocessor import DataPreprocessor;
      loader = DataLoader();
      preprocessor = DataPreprocessor();
      data = loader.load_titanic();
      clean_data = preprocessor.preprocess_titanic(data);
      preprocessor.save_preprocessed_data(clean_data, 'titanic')
      "
    deps:
    - data/raw/titanic.csv
    - src/data/data_loader.py
    - src/data/preprocessor.py
    outs:
    - data/processed/titanic_clean.csv
    
  preprocess_housing:
    cmd: python -c "
      import sys; sys.path.append('src');
      from data.data_loader import DataLoader;
      from data.preprocessor import DataPreprocessor;
      loader = DataLoader();
      preprocessor = DataPreprocessor();
      data = loader.load_housing();
      clean_data = preprocessor.preprocess_housing(data);
      preprocessor.save_preprocessed_data(clean_data, 'housing')
      "
    deps:
    - data/raw/housing.csv
    - src/data/data_loader.py
    - src/data/preprocessor.py
    outs:
    - data/processed/housing_clean.csv
    
  feature_engineering_titanic:
    cmd: python -c "
      import sys; sys.path.append('src');
      from data.data_loader import DataLoader;
      from data.preprocessor import DataPreprocessor;
      from features.feature_engineer import FeatureEngineer;
      import pandas as pd;
      data = pd.read_csv('data/processed/titanic_clean.csv');
      engineer = FeatureEngineer();
      features = engineer.engineer_titanic_features(data);
      engineer.save_features(features, 'titanic')
      "
    deps:
    - data/processed/titanic_clean.csv
    - src/features/feature_engineer.py
    outs:
    - data/features/titanic_features.csv
    
  feature_engineering_housing:
    cmd: python -c "
      import sys; sys.path.append('src');
      from data.data_loader import DataLoader;
      from data.preprocessor import DataPreprocessor;
      from features.feature_engineer import FeatureEngineer;
      import pandas as pd;
      data = pd.read_csv('data/processed/housing_clean.csv');
      engineer = FeatureEngineer();
      features = engineer.engineer_housing_features(data);
      engineer.save_features(features, 'housing')
      "
    deps:
    - data/processed/housing_clean.csv
    - src/features/feature_engineer.py
    outs:
    - data/features/housing_features.csv
    
  train_models:
    cmd: python train_models_simple.py
    deps:
    - data/features/titanic_features.csv
    - data/features/housing_features.csv
    - train_models_simple.py
    outs:
    - trained_models/
    metrics:
    - model_metrics.json
"""
    
    with open("dvc.yaml", "w") as f:
        f.write(dvc_pipeline)
    
    run_command("git add dvc.yaml", "Add DVC pipeline")
    print("✅ DVC pipeline created!")

def test_dvc_pipeline():
    """Test the DVC pipeline"""
    print("\n🧪 TESTING DVC PIPELINE")
    print("=" * 30)
    
    # Check DVC status
    run_command("dvc status", "Check DVC status")
    
    # Show DVC DAG
    run_command("dvc dag", "Show DVC pipeline DAG")
    
    print("✅ DVC pipeline testing completed!")

def main():
    """Main function"""
    print("🔄 COMPLETE DVC DATA VERSIONING SETUP")
    print("=" * 60)
    
    # Setup data versioning
    setup_dvc_data_versioning()
    
    # Create enhanced pipeline
    create_dvc_pipeline()
    
    # Test pipeline
    test_dvc_pipeline()
    
    print("\n" + "=" * 60)
    print("🎉 DVC DATA VERSIONING COMPLETED!")
    print("=" * 60)
    print("\n📋 What you can now do:")
    print("  • dvc status - Check pipeline status")
    print("  • dvc repro - Run the entire pipeline")
    print("  • dvc dag - View pipeline dependencies")
    print("  • dvc metrics show - View model metrics")
    print("  • dvc params diff - Compare parameters")

if __name__ == "__main__":
    main()