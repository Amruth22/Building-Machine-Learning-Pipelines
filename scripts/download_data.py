"""
Script to download sample datasets for ML pipeline
Downloads Titanic and Boston Housing datasets
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_titanic_dataset():
    """Download Titanic dataset"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    titanic_path = data_dir / "titanic.csv"
    
    if titanic_path.exists():
        logger.info(f"Titanic dataset already exists at {titanic_path}")
        return
    
    # Try multiple sources for Titanic dataset
    urls = [
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    ]
    
    for url in urls:
        try:
            logger.info(f"Downloading Titanic dataset from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Validate the downloaded data
            data = pd.read_csv(pd.io.common.StringIO(response.text))
            
            # Check if it looks like Titanic data
            required_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age']
            if all(col in data.columns for col in required_columns):
                data.to_csv(titanic_path, index=False)
                logger.info(f"Titanic dataset downloaded successfully to {titanic_path}")
                logger.info(f"Dataset shape: {data.shape}")
                return
            else:
                logger.warning(f"Downloaded data doesn't look like Titanic dataset")
                
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue
    
    # If all downloads failed, create a synthetic Titanic dataset
    logger.info("Creating synthetic Titanic dataset as fallback")
    create_synthetic_titanic(titanic_path)

def create_synthetic_titanic(file_path: Path):
    """Create a synthetic Titanic dataset for demonstration"""
    np.random.seed(42)
    n_samples = 891
    
    # Generate synthetic Titanic-like data
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.binomial(1, 0.38, n_samples),  # ~38% survival rate
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f"Passenger_{i}" for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_samples).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_samples).clip(0, 8),
        'Parch': np.random.poisson(0.4, n_samples).clip(0, 6),
        'Ticket': [f"TICKET_{i}" for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 500),
        'Cabin': [f"C{i}" if np.random.random() > 0.77 else None for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }
    
    # Add some realistic correlations
    # Higher class passengers more likely to survive
    for i in range(n_samples):
        if data['Pclass'][i] == 1:
            data['Survived'][i] = np.random.binomial(1, 0.63)
        elif data['Pclass'][i] == 2:
            data['Survived'][i] = np.random.binomial(1, 0.47)
        else:
            data['Survived'][i] = np.random.binomial(1, 0.24)
    
    # Women more likely to survive
    for i in range(n_samples):
        if data['Sex'][i] == 'female':
            data['Survived'][i] = np.random.binomial(1, 0.74)
        else:
            data['Survived'][i] = np.random.binomial(1, 0.19)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic Titanic dataset created at {file_path}")
    logger.info(f"Dataset shape: {df.shape}")

def download_housing_dataset():
    """Download or create Boston Housing dataset"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    housing_path = data_dir / "housing.csv"
    
    if housing_path.exists():
        logger.info(f"Housing dataset already exists at {housing_path}")
        return
    
    try:
        # Try to use sklearn's Boston housing dataset
        from sklearn.datasets import load_boston
        logger.info("Loading Boston Housing dataset from sklearn")
        
        boston = load_boston()
        housing_df = pd.DataFrame(boston.data, columns=boston.feature_names)
        housing_df['MEDV'] = boston.target
        
        housing_df.to_csv(housing_path, index=False)
        logger.info(f"Housing dataset created at {housing_path}")
        logger.info(f"Dataset shape: {housing_df.shape}")
        
    except Exception as e:
        logger.warning(f"Failed to load Boston housing from sklearn: {e}")
        logger.info("Creating synthetic housing dataset as fallback")
        create_synthetic_housing(housing_path)

def create_synthetic_housing(file_path: Path):
    """Create a synthetic housing dataset"""
    np.random.seed(42)
    n_samples = 506
    
    # Generate synthetic housing data with realistic correlations
    data = {}
    
    # Crime rate (per capita crime rate by town)
    data['CRIM'] = np.random.exponential(3, n_samples).clip(0, 90)
    
    # Residential land zoned for lots over 25,000 sq.ft.
    data['ZN'] = np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
    
    # Proportion of non-retail business acres per town
    data['INDUS'] = np.random.uniform(0, 30, n_samples)
    
    # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    data['CHAS'] = np.random.binomial(1, 0.07, n_samples)
    
    # Nitric oxides concentration (parts per 10 million)
    data['NOX'] = np.random.uniform(0.3, 0.9, n_samples)
    
    # Average number of rooms per dwelling
    data['RM'] = np.random.normal(6.3, 0.7, n_samples).clip(3, 9)
    
    # Proportion of owner-occupied units built prior to 1940
    data['AGE'] = np.random.uniform(0, 100, n_samples)
    
    # Weighted distances to employment centers
    data['DIS'] = np.random.uniform(1, 12, n_samples)
    
    # Index of accessibility to radial highways
    data['RAD'] = np.random.choice(range(1, 25), n_samples)
    
    # Property tax rate per $10,000
    data['TAX'] = np.random.uniform(150, 800, n_samples)
    
    # Pupil-teacher ratio by town
    data['PTRATIO'] = np.random.uniform(12, 22, n_samples)
    
    # Proportion of blacks by town
    data['B'] = np.random.uniform(300, 400, n_samples)
    
    # % lower status of the population
    data['LSTAT'] = np.random.uniform(1, 40, n_samples)
    
    # Create target variable with realistic correlations
    # More rooms, lower crime, lower LSTAT -> higher price
    base_price = 20
    price_factors = (
        (data['RM'] - 6) * 5 +  # More rooms increase price
        -data['CRIM'] * 0.5 +   # Higher crime decreases price
        -data['LSTAT'] * 0.3 +  # Higher LSTAT decreases price
        data['CHAS'] * 3 +      # River proximity increases price
        -data['NOX'] * 10 +     # Pollution decreases price
        np.random.normal(0, 3, n_samples)  # Random noise
    )
    
    data['MEDV'] = (base_price + price_factors).clip(5, 50)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    logger.info(f"Synthetic housing dataset created at {file_path}")
    logger.info(f"Dataset shape: {df.shape}")

def verify_datasets():
    """Verify that datasets were downloaded/created successfully"""
    data_dir = Path("data/raw")
    
    datasets = {
        'titanic.csv': ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age'],
        'housing.csv': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    }
    
    all_good = True
    
    for filename, required_columns in datasets.items():
        file_path = data_dir / filename
        
        if not file_path.exists():
            logger.error(f"Dataset {filename} not found!")
            all_good = False
            continue
        
        try:
            df = pd.read_csv(file_path)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Dataset {filename} missing columns: {missing_columns}")
                all_good = False
            else:
                logger.info(f"✅ Dataset {filename} verified successfully - Shape: {df.shape}")
                
        except Exception as e:
            logger.error(f"Error reading dataset {filename}: {e}")
            all_good = False
    
    if all_good:
        logger.info("🎉 All datasets downloaded and verified successfully!")
    else:
        logger.error("❌ Some datasets have issues. Please check the logs above.")
    
    return all_good

def main():
    """Main function to download all datasets"""
    logger.info("Starting dataset download process...")
    
    try:
        # Download datasets
        download_titanic_dataset()
        download_housing_dataset()
        
        # Verify datasets
        success = verify_datasets()
        
        if success:
            logger.info("Dataset download completed successfully!")
            return 0
        else:
            logger.error("Dataset download completed with errors!")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error during dataset download: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)