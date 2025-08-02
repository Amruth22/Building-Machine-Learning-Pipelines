"""
Simple dataset downloader for ML pipeline
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_titanic_dataset():
    """Download Titanic dataset"""
    print("📥 Downloading Titanic dataset...")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    titanic_path = data_dir / "titanic.csv"
    
    if titanic_path.exists():
        print(f"✅ Titanic dataset already exists at {titanic_path}")
        return True
    
    # Try to download from multiple sources
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    ]
    
    for url in urls:
        try:
            print(f"🔄 Trying to download from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save the data
            with open(titanic_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Verify the data
            data = pd.read_csv(titanic_path)
            required_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age']
            
            if all(col in data.columns for col in required_columns):
                print(f"✅ Titanic dataset downloaded successfully!")
                print(f"   Shape: {data.shape}")
                print(f"   Columns: {list(data.columns)}")
                return True
            else:
                print("⚠️ Downloaded data doesn't have required columns")
                
        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
            continue
    
    # If download failed, create synthetic data
    print("🔄 Creating synthetic Titanic dataset...")
    create_synthetic_titanic(titanic_path)
    return True

def create_synthetic_titanic(file_path):
    """Create synthetic Titanic dataset"""
    np.random.seed(42)
    n_samples = 891
    
    # Generate realistic Titanic-like data
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.binomial(1, 0.38, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f"Passenger, Mr. John {i}" for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_samples).clip(0.5, 80),
        'SibSp': np.random.poisson(0.5, n_samples).clip(0, 8),
        'Parch': np.random.poisson(0.4, n_samples).clip(0, 6),
        'Ticket': [f"TICKET{i}" for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 500),
        'Cabin': [f"C{i}" if np.random.random() > 0.77 else None for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }
    
    # Add realistic correlations
    for i in range(n_samples):
        # Higher class passengers more likely to survive
        if data['Pclass'][i] == 1:
            data['Survived'][i] = np.random.binomial(1, 0.63)
        elif data['Pclass'][i] == 2:
            data['Survived'][i] = np.random.binomial(1, 0.47)
        else:
            data['Survived'][i] = np.random.binomial(1, 0.24)
        
        # Women more likely to survive
        if data['Sex'][i] == 'female':
            data['Survived'][i] = np.random.binomial(1, 0.74)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic Titanic dataset created: {df.shape}")

def download_housing_dataset():
    """Download or create Housing dataset"""
    print("📥 Downloading Housing dataset...")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    housing_path = data_dir / "housing.csv"
    
    if housing_path.exists():
        print(f"✅ Housing dataset already exists at {housing_path}")
        return True
    
    try:
        # Try to use sklearn's Boston housing dataset
        from sklearn.datasets import load_boston
        print("🔄 Loading Boston Housing dataset from sklearn...")
        
        # Suppress the warning about deprecated dataset
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boston = load_boston()
        
        housing_df = pd.DataFrame(boston.data, columns=boston.feature_names)
        housing_df['MEDV'] = boston.target
        
        housing_df.to_csv(housing_path, index=False)
        print(f"✅ Housing dataset created successfully!")
        print(f"   Shape: {housing_df.shape}")
        print(f"   Columns: {list(housing_df.columns)}")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not load from sklearn: {e}")
        print("🔄 Creating synthetic housing dataset...")
        create_synthetic_housing(housing_path)
        return True

def create_synthetic_housing(file_path):
    """Create synthetic housing dataset"""
    np.random.seed(42)
    n_samples = 506
    
    # Generate synthetic housing data
    data = {}
    
    # Crime rate
    data['CRIM'] = np.random.exponential(3, n_samples).clip(0, 90)
    
    # Residential land zoned
    data['ZN'] = np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
    
    # Non-retail business acres
    data['INDUS'] = np.random.uniform(0, 30, n_samples)
    
    # Charles River dummy
    data['CHAS'] = np.random.binomial(1, 0.07, n_samples)
    
    # Nitric oxides concentration
    data['NOX'] = np.random.uniform(0.3, 0.9, n_samples)
    
    # Average number of rooms
    data['RM'] = np.random.normal(6.3, 0.7, n_samples).clip(3, 9)
    
    # Proportion built prior to 1940
    data['AGE'] = np.random.uniform(0, 100, n_samples)
    
    # Weighted distances to employment centers
    data['DIS'] = np.random.uniform(1, 12, n_samples)
    
    # Index of accessibility to radial highways
    data['RAD'] = np.random.choice(range(1, 25), n_samples)
    
    # Property tax rate
    data['TAX'] = np.random.uniform(150, 800, n_samples)
    
    # Pupil-teacher ratio
    data['PTRATIO'] = np.random.uniform(12, 22, n_samples)
    
    # Proportion of blacks
    data['B'] = np.random.uniform(300, 400, n_samples)
    
    # % lower status of population
    data['LSTAT'] = np.random.uniform(1, 40, n_samples)
    
    # Create target with realistic correlations
    base_price = 20
    price_factors = (
        (data['RM'] - 6) * 5 +      # More rooms = higher price
        -data['CRIM'] * 0.5 +       # Higher crime = lower price
        -data['LSTAT'] * 0.3 +      # Higher LSTAT = lower price
        data['CHAS'] * 3 +          # River proximity = higher price
        -data['NOX'] * 10 +         # Pollution = lower price
        np.random.normal(0, 3, n_samples)  # Random noise
    )
    
    data['MEDV'] = (base_price + price_factors).clip(5, 50)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic housing dataset created: {df.shape}")

def verify_datasets():
    """Verify that datasets were created successfully"""
    print("\n🔍 Verifying datasets...")
    
    datasets = {
        'data/raw/titanic.csv': ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'],
        'data/raw/housing.csv': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    }
    
    all_good = True
    
    for file_path, required_columns in datasets.items():
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"❌ {file_path} missing columns: {missing_columns}")
                    all_good = False
                else:
                    print(f"✅ {file_path} verified - Shape: {df.shape}")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                all_good = False
        else:
            print(f"❌ {file_path} not found")
            all_good = False
    
    return all_good

def main():
    """Main function to download all datasets"""
    print("📥 DATASET DOWNLOAD UTILITY")
    print("=" * 50)
    
    try:
        # Download datasets
        titanic_success = download_titanic_dataset()
        housing_success = download_housing_dataset()
        
        # Verify datasets
        if titanic_success and housing_success:
            verification_success = verify_datasets()
            
            if verification_success:
                print("\n🎉 All datasets ready!")
                print("=" * 50)
                print("You can now run:")
                print("  • jupyter notebook (open 01_data_exploration.ipynb)")
                print("  • python scripts/run_pipeline.py --dataset titanic")
                print("  • python train_models_simple.py")
                return 0
            else:
                print("\n⚠️ Some datasets have issues")
                return 1
        else:
            print("\n❌ Dataset download failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)