"""
Feature engineering utilities for ML pipeline
Creates new features and selects important ones
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering class for creating and selecting features"""
    
    def __init__(self, config_path: str = "configs/feature_config.yaml"):
        """
        Initialize feature engineer
        
        Args:
            config_path (str): Path to feature configuration file
        """
        self.config = self._load_config(config_path)
        self.feature_selectors = {}
        self.feature_importance = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default feature engineering configuration"""
        return {
            'titanic': {
                'new_features': {
                    'FamilySize': 'SibSp + Parch + 1',
                    'IsAlone': 'FamilySize == 1',
                    'Title': 'extract from Name',
                    'AgeGroup': 'binned Age',
                    'FareGroup': 'binned Fare'
                },
                'feature_selection': {
                    'method': 'selectkbest',
                    'k': 10
                }
            },
            'housing': {
                'new_features': {
                    'RoomsPerHousehold': 'RM / (RM + 1)',
                    'CrimeLevel': 'binned CRIM',
                    'AgeGroup': 'binned AGE'
                },
                'feature_selection': {
                    'method': 'selectkbest',
                    'k': 12
                }
            }
        }
    
    def engineer_titanic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for Titanic dataset
        
        Args:
            data (pd.DataFrame): Preprocessed Titanic dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        logger.info("Engineering features for Titanic dataset...")
        
        df = data.copy()
        
        # Family-related features
        df = self._create_family_features(df)
        
        # Name-related features
        df = self._create_name_features(df)
        
        # Age-related features
        df = self._create_age_features(df)
        
        # Fare-related features
        df = self._create_fare_features(df)
        
        # Interaction features
        df = self._create_interaction_features_titanic(df)
        
        # Statistical features
        df = self._create_statistical_features(df)
        
        logger.info(f"Titanic feature engineering completed. Shape: {df.shape}")
        return df
    
    def engineer_housing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for Housing dataset
        
        Args:
            data (pd.DataFrame): Preprocessed Housing dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        logger.info("Engineering features for Housing dataset...")
        
        df = data.copy()
        
        # Location-related features
        df = self._create_location_features(df)
        
        # Property-related features
        df = self._create_property_features(df)
        
        # Economic features
        df = self._create_economic_features(df)
        
        # Interaction features
        df = self._create_interaction_features_housing(df)
        
        # Polynomial features (limited)
        df = self._create_polynomial_features(df, degree=2, max_features=5)
        
        logger.info(f"Housing feature engineering completed. Shape: {df.shape}")
        return df
    
    def _create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create family-related features for Titanic"""
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            # Family size
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            
            # Is alone
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            
            # Family size categories
            df['FamilySizeGroup'] = pd.cut(df['FamilySize'], 
                                         bins=[0, 1, 4, 20], 
                                         labels=['Alone', 'Small', 'Large'])
            
            # Has siblings/spouse
            df['HasSibSp'] = (df['SibSp'] > 0).astype(int)
            
            # Has parents/children
            df['HasParch'] = (df['Parch'] > 0).astype(int)
            
            logger.info("Created family-related features")
        
        return df
    
    def _create_name_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create name-related features for Titanic"""
        if 'Name' in df.columns:
            # Extract title
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
            
            # Group rare titles
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
                'Mlle': 'Miss', 'Countess': 'Royalty', 'Ms': 'Miss', 'Lady': 'Royalty',
                'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty', 'Mme': 'Mrs',
                'Capt': 'Officer', 'Sir': 'Royalty'
            }
            df['Title'] = df['Title'].map(title_mapping).fillna('Other')
            
            # Name length
            df['NameLength'] = df['Name'].str.len()
            
            # Number of words in name
            df['NameWords'] = df['Name'].str.split().str.len()
            
            # Has nickname (parentheses in name)
            df['HasNickname'] = df['Name'].str.contains('\\(').astype(int)
            
            logger.info("Created name-related features")
        
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features"""
        if 'Age' in df.columns:
            # Age groups
            df['AgeGroup'] = pd.cut(df['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
            
            # Is child
            df['IsChild'] = (df['Age'] < 18).astype(int)
            
            # Is elderly
            df['IsElderly'] = (df['Age'] >= 60).astype(int)
            
            # Age squared (non-linear relationship)
            df['AgeSquared'] = df['Age'] ** 2
            
            # Age bins (more granular)
            try:
                df['AgeBin'] = pd.qcut(df['Age'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except ValueError:
                df['AgeBin'] = pd.cut(df['Age'], bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            
            logger.info("Created age-related features")
        
        return df
    
    def _create_fare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fare-related features"""
        if 'Fare' in df.columns:
            # Fare groups
            df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            
            # Fare per person (if family size available)
            if 'FamilySize' in df.columns:
                df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            
            # Log fare (handle skewness)
            df['LogFare'] = np.log1p(df['Fare'])
            
            # Fare bins
            df['FareBin'] = pd.cut(df['Fare'], bins=10, labels=False)
            
            # High fare indicator
            fare_threshold = df['Fare'].quantile(0.8)
            df['HighFare'] = (df['Fare'] > fare_threshold).astype(int)
            
            logger.info("Created fare-related features")
        
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-related features for Housing"""
        # Crime level categories
        if 'CRIM' in df.columns:
            try:
                df['CrimeLevel'] = pd.qcut(df['CRIM'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            except ValueError:
                df['CrimeLevel'] = pd.cut(df['CRIM'], bins=3, labels=['Low', 'Medium', 'High'])
            df['HighCrime'] = (df['CRIM'] > df['CRIM'].quantile(0.75)).astype(int)
            df['LogCrime'] = np.log1p(df['CRIM'])
        
        # Distance categories
        if 'DIS' in df.columns:
            try:
                df['DistanceGroup'] = pd.qcut(df['DIS'], q=3, labels=['Close', 'Medium', 'Far'], duplicates='drop')
            except ValueError:
                df['DistanceGroup'] = pd.cut(df['DIS'], bins=3, labels=['Close', 'Medium', 'Far'])
            df['VeryClose'] = (df['DIS'] < df['DIS'].quantile(0.25)).astype(int)
        
        # Accessibility
        if 'RAD' in df.columns:
            df['HighAccessibility'] = (df['RAD'] >= 20).astype(int)
            df['AccessibilityGroup'] = pd.cut(df['RAD'], bins=3, labels=['Low', 'Medium', 'High'])
        
        # River proximity
        if 'CHAS' in df.columns:
            df['RiverProximity'] = df['CHAS']  # Already binary
        
        logger.info("Created location-related features")
        return df
    
    def _create_property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create property-related features for Housing"""
        # Room-related features
        if 'RM' in df.columns:
            df['RoomCategory'] = pd.cut(df['RM'], bins=[0, 5, 6, 7, 10], 
                                       labels=['Small', 'Medium', 'Large', 'VeryLarge'])
            df['HighRooms'] = (df['RM'] > 7).astype(int)
            df['RoomSquared'] = df['RM'] ** 2
        
        # Age-related features
        if 'AGE' in df.columns:
            df['AgeGroup'] = pd.cut(df['AGE'], bins=[0, 25, 50, 75, 100], 
                                   labels=['New', 'Moderate', 'Old', 'VeryOld'])
            df['NewBuilding'] = (df['AGE'] < 25).astype(int)
            df['OldBuilding'] = (df['AGE'] > 75).astype(int)
        
        # Zoning
        if 'ZN' in df.columns:
            df['HasZoning'] = (df['ZN'] > 0).astype(int)
            df['HighZoning'] = (df['ZN'] > 50).astype(int)
        
        logger.info("Created property-related features")
        return df
    
    def _create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic-related features for Housing"""
        # Tax-related features
        if 'TAX' in df.columns:
            try:
                df['TaxLevel'] = pd.qcut(df['TAX'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            except ValueError:
                df['TaxLevel'] = pd.cut(df['TAX'], bins=3, labels=['Low', 'Medium', 'High'])
            df['HighTax'] = (df['TAX'] > df['TAX'].quantile(0.75)).astype(int)
        
        # Pupil-teacher ratio
        if 'PTRATIO' in df.columns:
            df['PTRatioGroup'] = pd.cut(df['PTRATIO'], bins=3, labels=['Good', 'Average', 'Poor'])
            df['GoodSchools'] = (df['PTRATIO'] < 15).astype(int)
        
        # Lower status population
        if 'LSTAT' in df.columns:
            try:
                df['LowStatusLevel'] = pd.qcut(df['LSTAT'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            except ValueError:
                df['LowStatusLevel'] = pd.cut(df['LSTAT'], bins=3, labels=['Low', 'Medium', 'High'])
            df['HighLowStatus'] = (df['LSTAT'] > df['LSTAT'].quantile(0.75)).astype(int)
            df['LogLSTAT'] = np.log1p(df['LSTAT'])
        
        logger.info("Created economic-related features")
        
        # Handle any remaining categorical columns for housing
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            if col not in ['MEDV']:  # Don't encode target variable
                try:
                    col_dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, col_dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
                    logger.info(f"One-hot encoded remaining categorical column: {col}")
                except Exception as e:
                    logger.warning(f"Could not encode column {col}: {e}")
        
        return df
    
    def _create_interaction_features_titanic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for Titanic"""
        # Age and class interaction
        if 'Age' in df.columns and 'Pclass' in df.columns:
            df['Age_Pclass'] = df['Age'] * df['Pclass']
        
        # Fare and class interaction
        if 'Fare' in df.columns and 'Pclass' in df.columns:
            df['Fare_Pclass'] = df['Fare'] / df['Pclass']
        
        # Family size and class
        if 'FamilySize' in df.columns and 'Pclass' in df.columns:
            df['FamilySize_Pclass'] = df['FamilySize'] * df['Pclass']
        
        # Age and fare
        if 'Age' in df.columns and 'Fare' in df.columns:
            df['Age_Fare'] = df['Age'] * df['Fare']
        
        logger.info("Created interaction features for Titanic")
        return df
    
    def _create_interaction_features_housing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for Housing"""
        # Rooms and age interaction
        if 'RM' in df.columns and 'AGE' in df.columns:
            df['RM_AGE'] = df['RM'] * df['AGE']
        
        # Crime and distance
        if 'CRIM' in df.columns and 'DIS' in df.columns:
            df['CRIM_DIS'] = df['CRIM'] / (df['DIS'] + 1)
        
        # Tax and pupil-teacher ratio
        if 'TAX' in df.columns and 'PTRATIO' in df.columns:
            df['TAX_PTRATIO'] = df['TAX'] * df['PTRATIO']
        
        # NOX and age
        if 'NOX' in df.columns and 'AGE' in df.columns:
            df['NOX_AGE'] = df['NOX'] * df['AGE']
        
        logger.info("Created interaction features for Housing")
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) >= 2:
            # Mean of numerical features
            df['NumFeaturesMean'] = df[numerical_cols].mean(axis=1)
            
            # Standard deviation of numerical features
            df['NumFeaturesStd'] = df[numerical_cols].std(axis=1)
            
            # Number of zero values
            df['NumZeros'] = (df[numerical_cols] == 0).sum(axis=1)
            
            logger.info("Created statistical features")
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, degree: int = 2, max_features: int = 5) -> pd.DataFrame:
        """Create polynomial features for top numerical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select top numerical columns by variance
        if len(numerical_cols) > max_features:
            variances = df[numerical_cols].var().sort_values(ascending=False)
            top_cols = variances.head(max_features).index.tolist()
        else:
            top_cols = numerical_cols
        
        if len(top_cols) >= 2:
            # Create polynomial features for selected columns
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[top_cols])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(top_cols)
            
            # Add only interaction terms (not individual features)
            interaction_names = [name for name in feature_names if ' ' in name]
            interaction_indices = [i for i, name in enumerate(feature_names) if ' ' in name]
            
            if interaction_indices:
                poly_df = pd.DataFrame(poly_features[:, interaction_indices], 
                                     columns=interaction_names, 
                                     index=df.index)
                df = pd.concat([df, poly_df], axis=1)
                logger.info(f"Created {len(interaction_names)} polynomial interaction features")
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, dataset_name: str, 
                       method: str = 'selectkbest', k: int = 10) -> pd.DataFrame:
        """
        Select important features
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            dataset_name (str): Name of dataset
            method (str): Feature selection method
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        logger.info(f"Selecting features for {dataset_name} using {method}")
        
        if method == 'selectkbest':
            # Determine if classification or regression
            if dataset_name.lower() == 'titanic':
                score_func = f_classif
            else:
                score_func = f_regression
            
            selector = SelectKBest(score_func=score_func, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # Store feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            self.feature_importance[dataset_name] = feature_scores
            
        elif method == 'rfe':
            # Use Random Forest for RFE
            if dataset_name.lower() == 'titanic':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            selector = RFE(estimator=estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # Store feature rankings
            feature_rankings = dict(zip(X.columns, selector.ranking_))
            self.feature_importance[dataset_name] = feature_rankings
        
        else:
            logger.warning(f"Unknown feature selection method: {method}. Returning all features.")
            X_selected_df = X
            selected_features = X.columns.tolist()
        
        # Store selector for later use
        self.feature_selectors[dataset_name] = selector
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return X_selected_df
    
    def engineer_features(self, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on dataset type
        
        Args:
            dataset_name (str): Name of dataset
            data (pd.DataFrame): Preprocessed dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        if dataset_name.lower() == 'titanic':
            return self.engineer_titanic_features(data)
        elif dataset_name.lower() == 'housing':
            return self.engineer_housing_features(data)
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return data
    
    def save_features(self, data: pd.DataFrame, dataset_name: str, 
                     output_dir: str = "data/features") -> None:
        """
        Save engineered features to file
        
        Args:
            data (pd.DataFrame): Dataset with engineered features
            dataset_name (str): Name of dataset
            output_dir (str): Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_features.csv"
        file_path = output_path / filename
        
        data.to_csv(file_path, index=False)
        logger.info(f"Features for {dataset_name} saved to {file_path}")

if __name__ == "__main__":
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    
    # Process Titanic dataset
    titanic_data = loader.load_titanic()
    if not titanic_data.empty:
        titanic_clean = preprocessor.preprocess_titanic(titanic_data)
        titanic_features = engineer.engineer_titanic_features(titanic_clean)
        engineer.save_features(titanic_features, 'titanic')
        print(f"Titanic feature engineering completed. Shape: {titanic_features.shape}")
    
    # Process Housing dataset
    housing_data = loader.load_housing()
    if not housing_data.empty:
        housing_clean = preprocessor.preprocess_housing(housing_data)
        housing_features = engineer.engineer_housing_features(housing_clean)
        engineer.save_features(housing_features, 'housing')
        print(f"Housing feature engineering completed. Shape: {housing_features.shape}")