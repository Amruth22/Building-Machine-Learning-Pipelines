"""
Data preprocessing utilities for ML pipeline
Handles cleaning, transformation, and preparation of datasets
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing class for cleaning and transforming datasets"""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize data preprocessor
        
        Args:
            config_path (str): Path to data configuration file
        """
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration"""
        return {
            'preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'validation_size': 0.2
            },
            'datasets': {
                'titanic': {
                    'target_column': 'Survived',
                    'id_column': 'PassengerId',
                    'categorical_columns': ['Sex', 'Embarked', 'Pclass'],
                    'numerical_columns': ['Age', 'SibSp', 'Parch', 'Fare']
                },
                'housing': {
                    'target_column': 'MEDV',
                    'numerical_columns': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
                }
            }
        }
    
    def preprocess_titanic(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Titanic dataset
        
        Args:
            data (pd.DataFrame): Raw Titanic dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Preprocessing Titanic dataset...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values_titanic(df)
        
        # Create new features (basic feature engineering)
        df = self._create_basic_features_titanic(df)
        
        # Encode categorical variables
        df = self._encode_categorical_titanic(df)
        
        # Remove unnecessary columns
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Handle outliers
        df = self._handle_outliers(df, ['Fare', 'Age'])
        
        logger.info(f"Titanic preprocessing completed. Shape: {df.shape}")
        return df
    
    def preprocess_housing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Housing dataset
        
        Args:
            data (pd.DataFrame): Raw Housing dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Preprocessing Housing dataset...")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values (housing dataset typically has no missing values)
        df = self._handle_missing_values_housing(df)
        
        # Create new features (basic feature engineering)
        df = self._create_basic_features_housing(df)
        
        # Handle outliers
        numerical_cols = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        df = self._handle_outliers(df, numerical_cols)
        
        logger.info(f"Housing preprocessing completed. Shape: {df.shape}")
        return df
    
    def _handle_missing_values_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in Titanic dataset"""
        # Age: Fill with median
        if 'Age' in df.columns:
            age_median = df['Age'].median()
            df['Age'].fillna(age_median, inplace=True)
            logger.info(f"Filled {df['Age'].isnull().sum()} missing Age values with median: {age_median}")
        
        # Embarked: Fill with mode
        if 'Embarked' in df.columns:
            embarked_mode = df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S'
            df['Embarked'].fillna(embarked_mode, inplace=True)
            logger.info(f"Filled missing Embarked values with mode: {embarked_mode}")
        
        # Fare: Fill with median
        if 'Fare' in df.columns:
            fare_median = df['Fare'].median()
            df['Fare'].fillna(fare_median, inplace=True)
            logger.info(f"Filled missing Fare values with median: {fare_median}")
        
        # Cabin: Create binary feature for cabin availability
        if 'Cabin' in df.columns:
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            logger.info("Created HasCabin feature from Cabin column")
        
        return df
    
    def _handle_missing_values_housing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in Housing dataset"""
        # Housing dataset typically has no missing values, but let's be safe
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values in housing dataset: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing {col} values with median: {median_val}")
        
        return df
    
    def _create_basic_features_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for Titanic dataset"""
        # Family size
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            logger.info("Created FamilySize and IsAlone features")
        
        # Title extraction from Name
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
            
            # Group rare titles
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                'Capt': 'Rare', 'Sir': 'Rare'
            }
            df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
            logger.info("Created Title feature from Name")
        
        # Age groups
        if 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
            logger.info("Created AgeGroup feature")
        
        # Fare groups
        if 'Fare' in df.columns:
            df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            logger.info("Created FareGroup feature")
        
        return df
    
    def _create_basic_features_housing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for Housing dataset"""
        # Rooms per household (simplified)
        if 'RM' in df.columns:
            df['RoomsPerHousehold'] = df['RM'] / (df['RM'] + 1)  # Normalized rooms
            logger.info("Created RoomsPerHousehold feature")
        
        # Crime level groups
        if 'CRIM' in df.columns:
            df['CrimeLevel'] = pd.qcut(df['CRIM'], q=3, labels=['Low', 'Medium', 'High'])
            logger.info("Created CrimeLevel feature")
        
        # Age groups for buildings
        if 'AGE' in df.columns:
            df['AgeGroup'] = pd.cut(df['AGE'], bins=[0, 25, 50, 75, 100], 
                                   labels=['New', 'Moderate', 'Old', 'VeryOld'])
            logger.info("Created AgeGroup feature for buildings")
        
        # Distance groups
        if 'DIS' in df.columns:
            df['DistanceGroup'] = pd.qcut(df['DIS'], q=3, labels=['Close', 'Medium', 'Far'])
            logger.info("Created DistanceGroup feature")
        
        return df
    
    def _encode_categorical_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for Titanic dataset"""
        # Binary encoding for Sex
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
            logger.info("Encoded Sex column")
        
        # One-hot encoding for Embarked
        if 'Embarked' in df.columns:
            embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
            df = pd.concat([df, embarked_dummies], axis=1)
            df.drop('Embarked', axis=1, inplace=True)
            logger.info("One-hot encoded Embarked column")
        
        # One-hot encoding for Title
        if 'Title' in df.columns:
            title_dummies = pd.get_dummies(df['Title'], prefix='Title')
            df = pd.concat([df, title_dummies], axis=1)
            df.drop('Title', axis=1, inplace=True)
            logger.info("One-hot encoded Title column")
        
        # One-hot encoding for AgeGroup
        if 'AgeGroup' in df.columns:
            age_group_dummies = pd.get_dummies(df['AgeGroup'], prefix='AgeGroup')
            df = pd.concat([df, age_group_dummies], axis=1)
            df.drop('AgeGroup', axis=1, inplace=True)
            logger.info("One-hot encoded AgeGroup column")
        
        # One-hot encoding for FareGroup
        if 'FareGroup' in df.columns:
            fare_group_dummies = pd.get_dummies(df['FareGroup'], prefix='FareGroup')
            df = pd.concat([df, fare_group_dummies], axis=1)
            df.drop('FareGroup', axis=1, inplace=True)
            logger.info("One-hot encoded FareGroup column")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """
        Handle outliers in specified columns
        
        Args:
            df (pd.DataFrame): Dataset
            columns (List[str]): Columns to check for outliers
            method (str): Method to handle outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataset with outliers handled
        """
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
                if outliers_count > 0:
                    logger.info(f"Capped {outliers_count} outliers in {col} using IQR method")
            
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                z_scores = np.abs((df_clean[col] - mean) / std)
                
                outliers_count = (z_scores > 3).sum()
                # Cap outliers at 3 standard deviations
                df_clean.loc[z_scores > 3, col] = mean + 3 * std * np.sign(df_clean.loc[z_scores > 3, col] - mean)
                
                if outliers_count > 0:
                    logger.info(f"Capped {outliers_count} outliers in {col} using Z-score method")
        
        return df_clean
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test features
        """
        # Select numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            logger.warning("No numerical columns found for scaling")
            return X_train, X_test
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using standard scaling.")
            scaler = StandardScaler()
        
        # Fit scaler on training data and transform both sets
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        logger.info(f"Scaled {len(numerical_cols)} numerical columns using {method} scaling")
        return X_train_scaled, X_test_scaled
    
    def preprocess_dataset(self, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess dataset based on its type
        
        Args:
            dataset_name (str): Name of dataset ('titanic' or 'housing')
            data (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if dataset_name.lower() == 'titanic':
            return self.preprocess_titanic(data)
        elif dataset_name.lower() == 'housing':
            return self.preprocess_housing(data)
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return data
    
    def save_preprocessed_data(self, data: pd.DataFrame, dataset_name: str, 
                              output_dir: str = "data/processed") -> None:
        """
        Save preprocessed data to file
        
        Args:
            data (pd.DataFrame): Preprocessed dataset
            dataset_name (str): Name of dataset
            output_dir (str): Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_clean.csv"
        file_path = output_path / filename
        
        data.to_csv(file_path, index=False)
        logger.info(f"Preprocessed {dataset_name} data saved to {file_path}")

if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Process Titanic dataset
    titanic_data = loader.load_titanic()
    if not titanic_data.empty:
        titanic_clean = preprocessor.preprocess_titanic(titanic_data)
        preprocessor.save_preprocessed_data(titanic_clean, 'titanic')
        print(f"Titanic preprocessing completed. Shape: {titanic_clean.shape}")
        print(f"Columns: {list(titanic_clean.columns)}")
    
    # Process Housing dataset
    housing_data = loader.load_housing()
    if not housing_data.empty:
        housing_clean = preprocessor.preprocess_housing(housing_data)
        preprocessor.save_preprocessed_data(housing_clean, 'housing')
        print(f"Housing preprocessing completed. Shape: {housing_clean.shape}")
        print(f"Columns: {list(housing_clean.columns)}")