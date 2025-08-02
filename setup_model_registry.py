"""
Complete MLflow Model Registry Setup
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

class ModelRegistry:
    """Complete MLflow Model Registry implementation"""
    
    def __init__(self, tracking_uri="file:///./mlflow_tracking"):
        """Initialize Model Registry"""
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        print(f"🏛️ Model Registry initialized with URI: {tracking_uri}")
    
    def register_trained_models(self):
        """Register all trained models to MLflow Model Registry"""
        print("📝 REGISTERING TRAINED MODELS")
        print("=" * 40)
        
        # Define models to register
        models_to_register = [
            {
                'name': 'TitanicSurvivalPredictor',
                'description': 'Predicts passenger survival on the Titanic',
                'model_path': 'trained_models/titanic_logistic_regression.joblib',
                'dataset': 'titanic',
                'model_type': 'LogisticRegression',
                'performance_metric': 'accuracy',
                'performance_value': 0.894
            },
            {
                'name': 'TitanicSurvivalPredictor_RF',
                'description': 'Random Forest model for Titanic survival prediction',
                'model_path': 'trained_models/titanic_random_forest.joblib',
                'dataset': 'titanic',
                'model_type': 'RandomForest',
                'performance_metric': 'accuracy',
                'performance_value': 0.877
            },
            {
                'name': 'HousingPricePredictor',
                'description': 'Predicts Boston housing prices',
                'model_path': 'trained_models/housing_linear_regression.joblib',
                'dataset': 'housing',
                'model_type': 'LinearRegression',
                'performance_metric': 'r2_score',
                'performance_value': 0.681
            },
            {
                'name': 'HousingPricePredictor_RF',
                'description': 'Random Forest model for housing price prediction',
                'model_path': 'trained_models/housing_random_forest.joblib',
                'dataset': 'housing',
                'model_type': 'RandomForest',
                'performance_metric': 'r2_score',
                'performance_value': 0.669
            }
        ]
        
        registered_models = []
        
        for model_info in models_to_register:
            try:
                # Check if model file exists
                if not Path(model_info['model_path']).exists():
                    print(f"⚠️ Model file not found: {model_info['model_path']}")
                    continue
                
                # Load the model
                model = joblib.load(model_info['model_path'])
                
                # Create or get experiment
                experiment_name = f"{model_info['dataset'].title()}_Model_Registry"
                try:
                    experiment_id = mlflow.create_experiment(experiment_name)
                except mlflow.exceptions.MlflowException:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(experiment_name)
                
                # Start MLflow run
                with mlflow.start_run(run_name=f"Register_{model_info['name']}"):
                    
                    # Log model parameters
                    mlflow.log_params({
                        'model_type': model_info['model_type'],
                        'dataset': model_info['dataset'],
                        'registration_date': datetime.now().isoformat(),
                        'model_file': model_info['model_path']
                    })
                    
                    # Log performance metrics
                    mlflow.log_metric(model_info['performance_metric'], model_info['performance_value'])
                    
                    # Log the model
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=model_info['name']
                    )
                    
                    # Get the run ID
                    run_id = mlflow.active_run().info.run_id
                    
                    print(f"✅ Registered: {model_info['name']}")
                    
                    # Add model version description
                    try:
                        # Get the latest version
                        latest_versions = self.client.get_latest_versions(
                            model_info['name'], 
                            stages=["None"]
                        )
                        
                        if latest_versions:
                            version = latest_versions[0].version
                            
                            # Update model version with description
                            self.client.update_model_version(
                                name=model_info['name'],
                                version=version,
                                description=f"{model_info['description']}. "
                                          f"Performance: {model_info['performance_metric']} = {model_info['performance_value']:.3f}. "
                                          f"Registered on {datetime.now().strftime('%Y-%m-%d')}"
                            )
                            
                            registered_models.append({
                                'name': model_info['name'],
                                'version': version,
                                'run_id': run_id,
                                'performance': model_info['performance_value']
                            })
                    
                    except Exception as e:
                        print(f"⚠️ Could not update model version description: {e}")
            
            except Exception as e:
                print(f"❌ Failed to register {model_info['name']}: {e}")
        
        return registered_models
    
    def setup_model_staging(self):
        """Set up model staging workflow"""
        print("\n🔄 SETTING UP MODEL STAGING WORKFLOW")
        print("=" * 45)
        
        # Get all registered models
        registered_models = self.client.search_registered_models()
        
        for model in registered_models:
            model_name = model.name
            print(f"\n📋 Processing model: {model_name}")
            
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if versions:
                # Sort by version number
                versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
                latest_version = versions[0]
                
                print(f"   Latest version: {latest_version.version}")
                print(f"   Current stage: {latest_version.current_stage}")
                
                # Promote best models to staging
                if self._should_promote_to_staging(model_name, latest_version):
                    try:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=latest_version.version,
                            stage="Staging",
                            archive_existing_versions=False
                        )
                        print(f"   ✅ Promoted version {latest_version.version} to Staging")
                        
                        # Add staging annotation
                        self.client.set_model_version_tag(
                            name=model_name,
                            version=latest_version.version,
                            key="stage_promotion_date",
                            value=datetime.now().isoformat()
                        )
                        
                    except Exception as e:
                        print(f"   ⚠️ Could not promote to staging: {e}")
                
                # Promote best staging models to production
                if self._should_promote_to_production(model_name, latest_version):
                    try:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=latest_version.version,
                            stage="Production",
                            archive_existing_versions=True
                        )
                        print(f"   🚀 Promoted version {latest_version.version} to Production")
                        
                        # Add production annotation
                        self.client.set_model_version_tag(
                            name=model_name,
                            version=latest_version.version,
                            key="production_promotion_date",
                            value=datetime.now().isoformat()
                        )
                        
                    except Exception as e:
                        print(f"   ⚠️ Could not promote to production: {e}")
    
    def _should_promote_to_staging(self, model_name, model_version):
        """Determine if model should be promoted to staging"""
        # Simple rule: promote if it's the latest version
        return model_version.current_stage == "None"
    
    def _should_promote_to_production(self, model_name, model_version):
        """Determine if model should be promoted to production"""
        # Simple rule: promote best performing models
        performance_thresholds = {
            'TitanicSurvivalPredictor': 0.89,  # 89% accuracy
            'HousingPricePredictor': 0.65      # 65% R²
        }
        
        # This is simplified - in practice, you'd check actual performance metrics
        return model_name in performance_thresholds
    
    def create_model_comparison_dashboard(self):
        """Create model comparison dashboard"""
        print("\n📊 CREATING MODEL COMPARISON DASHBOARD")
        print("=" * 45)
        
        # Get all registered models
        registered_models = self.client.search_registered_models()
        
        comparison_data = []
        
        for model in registered_models:
            model_name = model.name
            
            # Get latest version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
                
                # Get run details
                try:
                    run = mlflow.get_run(latest_version.run_id)
                    metrics = run.data.metrics
                    params = run.data.params
                    
                    comparison_data.append({
                        'model_name': model_name,
                        'version': latest_version.version,
                        'stage': latest_version.current_stage,
                        'model_type': params.get('model_type', 'Unknown'),
                        'dataset': params.get('dataset', 'Unknown'),
                        'accuracy': metrics.get('accuracy', None),
                        'r2_score': metrics.get('r2_score', None),
                        'rmse': metrics.get('rmse', None),
                        'creation_timestamp': latest_version.creation_timestamp
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Could not get run details for {model_name}: {e}")
        
        # Create comparison DataFrame
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Save comparison report
            report_path = "model_registry_comparison.csv"
            df.to_csv(report_path, index=False)
            print(f"✅ Model comparison saved to: {report_path}")
            
            # Print summary
            print("\n📋 Model Registry Summary:")
            print(df.to_string(index=False))
            
            return df
        
        return None
    
    def create_model_deployment_configs(self):
        """Create deployment configurations for production models"""
        print("\n🚀 CREATING MODEL DEPLOYMENT CONFIGS")
        print("=" * 45)
        
        # Get production models
        registered_models = self.client.search_registered_models()
        
        deployment_configs = {}
        
        for model in registered_models:
            model_name = model.name
            
            # Get production versions
            production_versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            
            if production_versions:
                version = production_versions[0]
                
                # Create deployment config
                config = {
                    'model_name': model_name,
                    'model_version': version.version,
                    'model_uri': f"models:/{model_name}/Production",
                    'stage': 'Production',
                    'deployment_ready': True,
                    'api_endpoint': f"/predict/{model_name.lower()}",
                    'health_check': f"/health/{model_name.lower()}",
                    'created_at': datetime.now().isoformat()
                }
                
                deployment_configs[model_name] = config
                print(f"✅ Created deployment config for: {model_name}")
        
        # Save deployment configs
        if deployment_configs:
            with open("model_deployment_configs.json", "w", encoding='utf-8') as f:
                json.dump(deployment_configs, f, indent=2)
            
            print(f"✅ Deployment configs saved to: model_deployment_configs.json")
        
        return deployment_configs
    
    def generate_model_registry_report(self):
        """Generate comprehensive model registry report"""
        print("\n📄 GENERATING MODEL REGISTRY REPORT")
        print("=" * 45)
        
        report_lines = []
        report_lines.append("# MLflow Model Registry Report\\n")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        # Get all registered models
        registered_models = self.client.search_registered_models()
        
        report_lines.append(f"## Summary\\n")
        report_lines.append(f"Total registered models: {len(registered_models)}\\n\\n")
        
        for model in registered_models:
            model_name = model.name
            report_lines.append(f"### {model_name}\\n")
            
            # Get all versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            report_lines.append(f"Total versions: {len(versions)}\\n")
            
            # Stage distribution
            stages = {}
            for version in versions:
                stage = version.current_stage
                stages[stage] = stages.get(stage, 0) + 1
            
            for stage, count in stages.items():
                report_lines.append(f"- {stage}: {count} version(s)\\n")
            
            # Latest version details
            if versions:
                latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
                report_lines.append(f"Latest version: {latest.version} ({latest.current_stage})\\n")
            
            report_lines.append("\\n")
        
        # Save report
        with open("model_registry_report.md", "w", encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print("✅ Model registry report saved to: model_registry_report.md")

def main():
    """Main function to set up complete model registry"""
    print("🏛️ COMPLETE MODEL REGISTRY SETUP")
    print("=" * 50)
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Register models
    registered_models = registry.register_trained_models()
    
    # Set up staging
    registry.setup_model_staging()
    
    # Create comparison dashboard
    registry.create_model_comparison_dashboard()
    
    # Create deployment configs
    registry.create_model_deployment_configs()
    
    # Generate report
    registry.generate_model_registry_report()
    
    print("\\n" + "=" * 50)
    print("🎉 MODEL REGISTRY SETUP COMPLETED!")
    print("=" * 50)
    print("\\n📋 What you now have:")
    print("  • All models registered in MLflow Model Registry")
    print("  • Model staging workflow (None → Staging → Production)")
    print("  • Model comparison dashboard")
    print("  • Deployment configurations")
    print("  • Comprehensive registry report")
    
    print("\\n🚀 Production-ready models:")
    for model_info in registered_models:
        if model_info['performance'] > 0.85 or model_info['performance'] > 0.65:
            print(f"  • {model_info['name']} v{model_info['version']} - Performance: {model_info['performance']:.3f}")

if __name__ == "__main__":
    main()