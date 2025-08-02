"""
Main pipeline execution script
Runs the complete ML pipeline for specified dataset
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import yaml
import mlflow
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader
from data.data_validator import DataValidator
from data.preprocessor import DataPreprocessor
from features.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Main ML Pipeline orchestrator"""
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        """
        Initialize ML Pipeline
        
        Args:
            config_path (str): Path to pipeline configuration
        """
        self.config = self._load_config(config_path)
        self.setup_mlflow()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Pipeline state
        self.pipeline_results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load pipeline configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default pipeline configuration"""
        return {
            'mlflow': {
                'tracking_uri': 'mlruns',
                'artifact_location': 'mlruns'
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow_uri = self.config.get('mlflow', {}).get('tracking_uri', 'mlruns')
        mlflow.set_tracking_uri(f"file://{Path(mlflow_uri).absolute()}")
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    def run_data_pipeline(self, dataset_name: str) -> dict:
        """
        Run data processing pipeline
        
        Args:
            dataset_name (str): Name of dataset to process
            
        Returns:
            dict: Pipeline results
        """
        logger.info(f"Starting data pipeline for {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Stage 1: Data Loading
            logger.info("Stage 1: Loading data...")
            raw_data = self.data_loader.load_dataset(dataset_name)
            
            if raw_data.empty:
                raise ValueError(f"Failed to load {dataset_name} dataset")
            
            results['stages']['data_loading'] = {
                'status': 'success',
                'shape': raw_data.shape,
                'columns': list(raw_data.columns)
            }
            logger.info(f"✅ Data loaded successfully. Shape: {raw_data.shape}")
            
            # Stage 2: Data Validation
            logger.info("Stage 2: Validating data...")
            validation_results = self.data_validator.validate_dataset(raw_data, dataset_name)
            
            results['stages']['data_validation'] = {
                'status': 'success' if validation_results['validation_passed'] else 'warning',
                'quality_score': validation_results['summary']['validation_summary']['data_quality_score'],
                'issues': validation_results['issues'],
                'warnings': validation_results['warnings']
            }
            
            if not validation_results['validation_passed']:
                logger.warning(f"⚠️ Data validation issues found: {validation_results['issues']}")
            else:
                logger.info("✅ Data validation passed")
            
            # Stage 3: Data Preprocessing
            logger.info("Stage 3: Preprocessing data...")
            clean_data = self.preprocessor.preprocess_dataset(dataset_name, raw_data)
            
            # Save preprocessed data
            self.preprocessor.save_preprocessed_data(clean_data, dataset_name)
            
            results['stages']['data_preprocessing'] = {
                'status': 'success',
                'original_shape': raw_data.shape,
                'processed_shape': clean_data.shape,
                'columns_added': len(clean_data.columns) - len(raw_data.columns)
            }
            logger.info(f"✅ Data preprocessing completed. Shape: {clean_data.shape}")
            
            # Stage 4: Feature Engineering
            logger.info("Stage 4: Engineering features...")
            featured_data = self.feature_engineer.engineer_features(dataset_name, clean_data)
            
            # Save features
            self.feature_engineer.save_features(featured_data, dataset_name)
            
            results['stages']['feature_engineering'] = {
                'status': 'success',
                'input_shape': clean_data.shape,
                'output_shape': featured_data.shape,
                'features_added': len(featured_data.columns) - len(clean_data.columns)
            }
            logger.info(f"✅ Feature engineering completed. Shape: {featured_data.shape}")
            
            # Store final data
            results['final_data'] = featured_data
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"❌ Data pipeline failed: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results
    
    def run_model_pipeline(self, dataset_name: str, data: pd.DataFrame) -> dict:
        """
        Run model training pipeline
        
        Args:
            dataset_name (str): Name of dataset
            data (pd.DataFrame): Processed data
            
        Returns:
            dict: Model pipeline results
        """
        logger.info(f"Starting model pipeline for {dataset_name}")
        
        # This is a placeholder - we'll implement model training in the next files
        # For now, just return basic info
        results = {
            'dataset_name': dataset_name,
            'data_shape': data.shape,
            'status': 'placeholder',
            'message': 'Model training pipeline will be implemented in model classes'
        }
        
        logger.info("Model pipeline placeholder completed")
        return results
    
    def run_complete_pipeline(self, dataset_name: str) -> dict:
        """
        Run complete ML pipeline
        
        Args:
            dataset_name (str): Name of dataset to process
            
        Returns:
            dict: Complete pipeline results
        """
        logger.info(f"🚀 Starting complete ML pipeline for {dataset_name}")
        
        # Set MLflow experiment
        experiment_name = f"{dataset_name.title()}_Pipeline"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{dataset_name}_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            try:
                # Log pipeline parameters
                mlflow.log_param("dataset_name", dataset_name)
                mlflow.log_param("pipeline_start_time", datetime.now().isoformat())
                
                # Run data pipeline
                data_results = self.run_data_pipeline(dataset_name)
                
                # Log data pipeline metrics
                mlflow.log_metric("data_quality_score", 
                                data_results['stages']['data_validation']['quality_score'])
                mlflow.log_metric("final_features_count", data_results['final_data'].shape[1])
                mlflow.log_metric("final_samples_count", data_results['final_data'].shape[0])
                
                # Run model pipeline (placeholder for now)
                model_results = self.run_model_pipeline(dataset_name, data_results['final_data'])
                
                # Combine results
                complete_results = {
                    'pipeline_id': mlflow.active_run().info.run_id,
                    'dataset_name': dataset_name,
                    'status': 'success',
                    'data_pipeline': data_results,
                    'model_pipeline': model_results,
                    'completion_time': datetime.now().isoformat()
                }
                
                # Log completion
                mlflow.log_param("pipeline_status", "success")
                mlflow.log_param("pipeline_end_time", datetime.now().isoformat())
                
                logger.info(f"🎉 Complete pipeline finished successfully for {dataset_name}")
                
                return complete_results
                
            except Exception as e:
                logger.error(f"❌ Pipeline failed: {str(e)}")
                mlflow.log_param("pipeline_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def generate_pipeline_report(self, results: dict) -> None:
        """Generate pipeline execution report"""
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        dataset_name = results['dataset_name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"{dataset_name}_pipeline_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# ML Pipeline Report: {dataset_name.title()}\n\n")
            f.write(f"**Generated:** {results.get('completion_time', 'N/A')}\n")
            f.write(f"**Pipeline ID:** {results.get('pipeline_id', 'N/A')}\n")
            f.write(f"**Status:** {results.get('status', 'N/A')}\n\n")
            
            # Data pipeline results
            if 'data_pipeline' in results:
                data_results = results['data_pipeline']
                f.write("## Data Pipeline Results\n\n")
                
                for stage_name, stage_results in data_results['stages'].items():
                    f.write(f"### {stage_name.replace('_', ' ').title()}\n")
                    f.write(f"- **Status:** {stage_results['status']}\n")
                    
                    if 'shape' in stage_results:
                        f.write(f"- **Data Shape:** {stage_results['shape']}\n")
                    if 'quality_score' in stage_results:
                        f.write(f"- **Quality Score:** {stage_results['quality_score']:.1f}/100\n")
                    if 'issues' in stage_results and stage_results['issues']:
                        f.write(f"- **Issues:** {', '.join(stage_results['issues'])}\n")
                    if 'warnings' in stage_results and stage_results['warnings']:
                        f.write(f"- **Warnings:** {', '.join(stage_results['warnings'])}\n")
                    
                    f.write("\n")
            
            # Model pipeline results (placeholder)
            if 'model_pipeline' in results:
                model_results = results['model_pipeline']
                f.write("## Model Pipeline Results\n\n")
                f.write(f"- **Status:** {model_results['status']}\n")
                f.write(f"- **Message:** {model_results['message']}\n\n")
        
        logger.info(f"Pipeline report saved to {report_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['titanic', 'housing'],
                       help='Dataset to process')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--report', action='store_true',
                       help='Generate pipeline report')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize pipeline
        pipeline = MLPipeline(args.config)
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(args.dataset)
        
        # Generate report if requested
        if args.report:
            pipeline.generate_pipeline_report(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Status: {results['status']}")
        print(f"Pipeline ID: {results['pipeline_id']}")
        print(f"Final Data Shape: {results['data_pipeline']['final_data'].shape}")
        print(f"Completion Time: {results['completion_time']}")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n{'='*60}")
        print(f"❌ PIPELINE FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"{'='*60}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)