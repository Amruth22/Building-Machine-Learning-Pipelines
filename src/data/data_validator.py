"""
Data validation utilities for ML pipeline
Ensures data quality and consistency
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation class for quality checks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data validator
        
        Args:
            config (dict): Validation configuration
        """
        self.config = config or self._get_default_config()
        self.validation_results = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration"""
        return {
            'min_samples': 100,
            'max_missing_ratio': 0.5,
            'outlier_threshold': 3.0,
            'duplicate_threshold': 0.1,
            'required_columns': [],
            'categorical_max_cardinality': 50
        }
    
    def validate_dataset(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data validation
        
        Args:
            data (pd.DataFrame): Dataset to validate
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Validation results
        """
        logger.info(f"Starting validation for {dataset_name} dataset")
        
        results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'validation_passed': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Basic structure validation
        self._validate_structure(data, results)
        
        # Missing values validation
        self._validate_missing_values(data, results)
        
        # Data types validation
        self._validate_data_types(data, results)
        
        # Duplicates validation
        self._validate_duplicates(data, results)
        
        # Outliers detection
        self._detect_outliers(data, results)
        
        # Statistical validation
        self._validate_statistics(data, results)
        
        # Dataset-specific validation
        if dataset_name.lower() == 'titanic':
            self._validate_titanic_specific(data, results)
        elif dataset_name.lower() == 'housing':
            self._validate_housing_specific(data, results)
        
        # Generate summary
        self._generate_summary(results)
        
        self.validation_results[dataset_name] = results
        logger.info(f"Validation completed for {dataset_name}")
        
        return results
    
    def _validate_structure(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate basic data structure"""
        # Check if dataset is empty
        if data.empty:
            results['issues'].append("Dataset is empty")
            results['validation_passed'] = False
            return
        
        # Check minimum sample size
        if len(data) < self.config['min_samples']:
            results['warnings'].append(f"Dataset has only {len(data)} samples, minimum recommended: {self.config['min_samples']}")
        
        # Check for required columns
        if self.config['required_columns']:
            missing_cols = [col for col in self.config['required_columns'] if col not in data.columns]
            if missing_cols:
                results['issues'].append(f"Missing required columns: {missing_cols}")
                results['validation_passed'] = False
    
    def _validate_missing_values(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate missing values"""
        missing_stats = data.isnull().sum()
        missing_ratios = missing_stats / len(data)
        
        # Check overall missing value ratio
        total_missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if total_missing_ratio > self.config['max_missing_ratio']:
            results['issues'].append(f"High missing value ratio: {total_missing_ratio:.2%}")
            results['validation_passed'] = False
        
        # Check individual columns
        high_missing_cols = missing_ratios[missing_ratios > self.config['max_missing_ratio']].index.tolist()
        if high_missing_cols:
            results['warnings'].append(f"Columns with high missing values: {high_missing_cols}")
        
        results['summary']['missing_values'] = {
            'total_missing': int(missing_stats.sum()),
            'missing_ratio': float(total_missing_ratio),
            'columns_with_missing': missing_stats[missing_stats > 0].to_dict()
        }
    
    def _validate_data_types(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate data types"""
        dtype_info = {}
        
        for col in data.columns:
            dtype_info[col] = {
                'dtype': str(data[col].dtype),
                'unique_values': int(data[col].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(data[col]),
                'is_categorical': pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object'
            }
            
            # Check for high cardinality categorical variables
            if dtype_info[col]['is_categorical'] and dtype_info[col]['unique_values'] > self.config['categorical_max_cardinality']:
                results['warnings'].append(f"High cardinality categorical column: {col} ({dtype_info[col]['unique_values']} unique values)")
        
        results['summary']['data_types'] = dtype_info
    
    def _validate_duplicates(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate duplicate records"""
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / len(data)
        
        if duplicate_ratio > self.config['duplicate_threshold']:
            results['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.2%} ({duplicate_count} duplicates)")
        
        results['summary']['duplicates'] = {
            'duplicate_count': int(duplicate_count),
            'duplicate_ratio': float(duplicate_ratio)
        }
    
    def _detect_outliers(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Detect outliers in numerical columns"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            if data[col].nunique() > 2:  # Skip binary variables
                # Z-score method
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers_zscore = (z_scores > self.config['outlier_threshold']).sum()
                
                # IQR method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                
                outlier_info[col] = {
                    'outliers_zscore': int(outliers_zscore),
                    'outliers_iqr': int(outliers_iqr),
                    'outlier_ratio_zscore': float(outliers_zscore / len(data)),
                    'outlier_ratio_iqr': float(outliers_iqr / len(data))
                }
        
        results['summary']['outliers'] = outlier_info
    
    def _validate_statistics(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Validate basic statistics"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        stats_info = {}
        
        for col in numeric_cols:
            col_stats = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
                'skewness': float(data[col].skew()),
                'kurtosis': float(data[col].kurtosis())
            }
            
            # Check for extreme skewness
            if abs(col_stats['skewness']) > 2:
                results['warnings'].append(f"High skewness in column {col}: {col_stats['skewness']:.2f}")
            
            # Check for constant values
            if data[col].nunique() == 1:
                results['warnings'].append(f"Constant column detected: {col}")
            
            stats_info[col] = col_stats
        
        results['summary']['statistics'] = stats_info
    
    def _validate_titanic_specific(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Titanic-specific validation"""
        # Check target variable
        if 'Survived' in data.columns:
            if data['Survived'].nunique() != 2:
                results['issues'].append("Target variable 'Survived' should have exactly 2 unique values")
                results['validation_passed'] = False
            
            if not set(data['Survived'].unique()).issubset({0, 1}):
                results['issues'].append("Target variable 'Survived' should contain only 0 and 1")
                results['validation_passed'] = False
        
        # Check passenger class
        if 'Pclass' in data.columns:
            valid_classes = {1, 2, 3}
            if not set(data['Pclass'].unique()).issubset(valid_classes):
                results['warnings'].append(f"Unexpected passenger classes: {set(data['Pclass'].unique()) - valid_classes}")
        
        # Check sex values
        if 'Sex' in data.columns:
            valid_sex = {'male', 'female'}
            if not set(data['Sex'].unique()).issubset(valid_sex):
                results['warnings'].append(f"Unexpected sex values: {set(data['Sex'].unique()) - valid_sex}")
        
        # Check age range
        if 'Age' in data.columns:
            if data['Age'].min() < 0 or data['Age'].max() > 120:
                results['warnings'].append(f"Unusual age range: {data['Age'].min():.1f} to {data['Age'].max():.1f}")
    
    def _validate_housing_specific(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Housing-specific validation"""
        # Check target variable
        if 'MEDV' in data.columns:
            if data['MEDV'].min() <= 0:
                results['issues'].append("Target variable 'MEDV' should be positive")
                results['validation_passed'] = False
            
            if data['MEDV'].max() > 100:
                results['warnings'].append(f"Unusually high housing prices detected: max = {data['MEDV'].max()}")
        
        # Check crime rate
        if 'CRIM' in data.columns:
            if data['CRIM'].min() < 0:
                results['issues'].append("Crime rate 'CRIM' should be non-negative")
                results['validation_passed'] = False
        
        # Check number of rooms
        if 'RM' in data.columns:
            if data['RM'].min() < 1 or data['RM'].max() > 20:
                results['warnings'].append(f"Unusual room count range: {data['RM'].min():.1f} to {data['RM'].max():.1f}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generate validation summary"""
        summary = {
            'total_issues': len(results['issues']),
            'total_warnings': len(results['warnings']),
            'validation_status': 'PASSED' if results['validation_passed'] else 'FAILED',
            'data_quality_score': self._calculate_quality_score(results)
        }
        
        results['summary']['validation_summary'] = summary
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for issues and warnings
        score -= len(results['issues']) * 20  # Major issues
        score -= len(results['warnings']) * 5  # Minor warnings
        
        # Deduct points for missing values
        if 'missing_values' in results['summary']:
            missing_ratio = results['summary']['missing_values']['missing_ratio']
            score -= missing_ratio * 30
        
        # Deduct points for duplicates
        if 'duplicates' in results['summary']:
            duplicate_ratio = results['summary']['duplicates']['duplicate_ratio']
            score -= duplicate_ratio * 20
        
        return max(0.0, min(100.0, score))
    
    def save_validation_report(self, output_dir: str = "data/validation") -> None:
        """Save validation report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "data_quality_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_file}")
    
    def print_validation_summary(self, dataset_name: str) -> None:
        """Print validation summary"""
        if dataset_name not in self.validation_results:
            logger.error(f"No validation results found for {dataset_name}")
            return
        
        results = self.validation_results[dataset_name]
        summary = results['summary']['validation_summary']
        
        print(f"\n{'='*50}")
        print(f"DATA VALIDATION REPORT: {dataset_name.upper()}")
        print(f"{'='*50}")
        print(f"Dataset Shape: {results['shape']}")
        print(f"Validation Status: {summary['validation_status']}")
        print(f"Data Quality Score: {summary['data_quality_score']:.1f}/100")
        print(f"Issues Found: {summary['total_issues']}")
        print(f"Warnings: {summary['total_warnings']}")
        
        if results['issues']:
            print(f"\n🚨 ISSUES:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        if results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        print(f"\n{'='*50}")

if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Initialize components
    loader = DataLoader()
    validator = DataValidator()
    
    # Validate Titanic dataset
    titanic_data = loader.load_titanic()
    if not titanic_data.empty:
        titanic_results = validator.validate_dataset(titanic_data, 'titanic')
        validator.print_validation_summary('titanic')
    
    # Validate Housing dataset
    housing_data = loader.load_housing()
    if not housing_data.empty:
        housing_results = validator.validate_dataset(housing_data, 'housing')
        validator.print_validation_summary('housing')
    
    # Save validation report
    validator.save_validation_report()