"""
Data loading utilities for ML pipeline
Handles loading of different datasets with validation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class for handling different datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_titanic(self) -> pd.DataFrame:
        """
        Load Titanic dataset
        
        Returns:
            pd.DataFrame: Titanic dataset
        """
        file_path = self.data_dir / "titanic.csv"
        
        if not file_path.exists():
            logger.warning(f"Titanic dataset not found at {file_path}")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded Titanic dataset: {data.shape}")
            self._validate_titanic_data(data)
            return data
        except Exception as e:
            logger.error(f"Error loading Titanic dataset: {e}")
            return pd.DataFrame()
    
    def load_housing(self) -> pd.DataFrame:
        """
        Load Boston Housing dataset
        
        Returns:
            pd.DataFrame: Housing dataset
        """
        file_path = self.data_dir / "housing.csv"
        
        if not file_path.exists():
            logger.warning(f"Housing dataset not found at {file_path}")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded Housing dataset: {data.shape}")
            self._validate_housing_data(data)
            return data
        except Exception as e:
            logger.error(f"Error loading Housing dataset: {e}")
            return pd.DataFrame()
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load dataset by name
        
        Args:
            dataset_name (str): Name of dataset ('titanic' or 'housing')
            
        Returns:
            pd.DataFrame: Requested dataset
        """
        if dataset_name.lower() == 'titanic':
            return self.load_titanic()
        elif dataset_name.lower() == 'housing':
            return self.load_housing()
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return pd.DataFrame()
    
    def _validate_titanic_data(self, data: pd.DataFrame) -> None:
        """
        Validate Titanic dataset structure
        
        Args:
            data (pd.DataFrame): Titanic dataset to validate
        """
        required_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("Dataset is empty")
        
        if data['Survived'].nunique() != 2:
            raise ValueError("Target variable 'Survived' should have exactly 2 unique values")
        
        logger.info("Titanic dataset validation passed")
    
    def _validate_housing_data(self, data: pd.DataFrame) -> None:
        """
        Validate Housing dataset structure
        
        Args:
            data (pd.DataFrame): Housing dataset to validate
        """
        required_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("Dataset is empty")
        
        if data['MEDV'].min() <= 0:
            raise ValueError("Target variable 'MEDV' should be positive")
        
        logger.info("Housing dataset validation passed")
    
    def get_data_info(self, dataset_name: str) -> dict:
        """
        Get information about a dataset
        
        Args:
            dataset_name (str): Name of dataset
            
        Returns:
            dict: Dataset information
        """
        data = self.load_dataset(dataset_name)
        
        if data.empty:
            return {}
        
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns)
        }

def download_sample_datasets():
    """Download sample datasets if they don't exist"""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Titanic dataset URL (from Kaggle or other source)
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    titanic_path = data_dir / "titanic.csv"
    
    if not titanic_path.exists():
        try:
            logger.info("Downloading Titanic dataset...")
            response = requests.get(titanic_url)
            response.raise_for_status()
            
            with open(titanic_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Titanic dataset downloaded to {titanic_path}")
        except Exception as e:
            logger.error(f"Failed to download Titanic dataset: {e}")
    
    # Boston Housing dataset (create sample if not available)
    housing_path = data_dir / "housing.csv"
    
    if not housing_path.exists():
        try:
            from sklearn.datasets import load_boston
            logger.info("Creating Boston Housing dataset...")
            
            # Note: load_boston is deprecated, but we'll use it for educational purposes
            # In production, use alternative datasets
            boston = load_boston()
            housing_df = pd.DataFrame(boston.data, columns=boston.feature_names)
            housing_df['MEDV'] = boston.target
            
            housing_df.to_csv(housing_path, index=False)
            logger.info(f"Housing dataset created at {housing_path}")
        except Exception as e:
            logger.error(f"Failed to create Housing dataset: {e}")
            # Create a simple synthetic dataset as fallback
            np.random.seed(42)
            n_samples = 506
            synthetic_data = {
                'CRIM': np.random.exponential(3, n_samples),
                'ZN': np.random.uniform(0, 100, n_samples),
                'INDUS': np.random.uniform(0, 30, n_samples),
                'CHAS': np.random.binomial(1, 0.1, n_samples),
                'NOX': np.random.uniform(0.3, 0.9, n_samples),
                'RM': np.random.normal(6, 1, n_samples),
                'AGE': np.random.uniform(0, 100, n_samples),
                'DIS': np.random.uniform(1, 15, n_samples),
                'RAD': np.random.randint(1, 25, n_samples),
                'TAX': np.random.uniform(150, 800, n_samples),
                'PTRATIO': np.random.uniform(12, 22, n_samples),
                'B': np.random.uniform(300, 400, n_samples),
                'LSTAT': np.random.uniform(1, 40, n_samples),
                'MEDV': np.random.uniform(5, 50, n_samples)
            }
            
            synthetic_df = pd.DataFrame(synthetic_data)
            synthetic_df.to_csv(housing_path, index=False)
            logger.info(f"Synthetic housing dataset created at {housing_path}")

if __name__ == "__main__":
    # Download datasets if running as script
    download_sample_datasets()
    
    # Test data loading
    loader = DataLoader()
    
    # Test Titanic
    titanic_data = loader.load_titanic()
    if not titanic_data.empty:
        print("Titanic dataset loaded successfully!")
        print(f"Shape: {titanic_data.shape}")
        print(f"Columns: {list(titanic_data.columns)}")
    
    # Test Housing
    housing_data = loader.load_housing()
    if not housing_data.empty:
        print("\nHousing dataset loaded successfully!")
        print(f"Shape: {housing_data.shape}")
        print(f"Columns: {list(housing_data.columns)}")