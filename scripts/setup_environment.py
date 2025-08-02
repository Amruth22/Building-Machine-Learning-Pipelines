"""
Environment setup script for ML pipeline
Initializes MLflow, DVC, and creates necessary directories
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import mlflow
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directories for the ML pipeline"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "data/validation",
        "models/titanic",
        "models/housing",
        "experiments",
        "reports/evaluation_plots",
        "logs",
        "configs",
        "notebooks",
        "tests",
        "docker"
    ]
    
    logger.info("Creating directory structure...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    logger.info("Directory structure created successfully!")

def initialize_dvc():
    """Initialize DVC for data versioning"""
    try:
        logger.info("Initializing DVC...")
        
        # Check if DVC is already initialized
        if Path(".dvc").exists():
            logger.info("DVC already initialized")
            return True
        
        # Initialize DVC
        result = subprocess.run(["dvc", "init"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ DVC initialized successfully")
            
            # Create .dvcignore file
            dvcignore_content = """# DVC ignore file
*.log
*.tmp
__pycache__/
.pytest_cache/
.coverage
htmlcov/
"""
            with open(".dvcignore", "w") as f:
                f.write(dvcignore_content)
            
            logger.info("✅ .dvcignore file created")
            return True
        else:
            logger.error(f"Failed to initialize DVC: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("DVC not found. Please install DVC: pip install dvc")
        return False
    except Exception as e:
        logger.error(f"Error initializing DVC: {e}")
        return False

def initialize_mlflow():
    """Initialize MLflow tracking"""
    try:
        logger.info("Initializing MLflow...")
        
        # Set MLflow tracking URI to local directory
        mlflow_dir = Path("mlruns")
        mlflow_dir.mkdir(exist_ok=True)
        
        mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")
        
        # Create default experiment
        try:
            experiment_id = mlflow.create_experiment("Default")
            logger.info(f"✅ Created default MLflow experiment with ID: {experiment_id}")
        except mlflow.exceptions.MlflowException:
            logger.info("Default MLflow experiment already exists")
        
        # Create experiments for different datasets
        experiments = ["Titanic_Classification", "Housing_Regression"]
        
        for exp_name in experiments:
            try:
                experiment_id = mlflow.create_experiment(exp_name)
                logger.info(f"✅ Created MLflow experiment '{exp_name}' with ID: {experiment_id}")
            except mlflow.exceptions.MlflowException:
                logger.info(f"MLflow experiment '{exp_name}' already exists")
        
        logger.info("✅ MLflow initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing MLflow: {e}")
        return False

def create_config_files():
    """Create configuration files"""
    logger.info("Creating configuration files...")
    
    # Data configuration
    data_config = {
        'datasets': {
            'titanic': {
                'file_path': 'data/raw/titanic.csv',
                'target_column': 'Survived',
                'id_column': 'PassengerId',
                'categorical_columns': ['Sex', 'Embarked', 'Pclass'],
                'numerical_columns': ['Age', 'SibSp', 'Parch', 'Fare']
            },
            'housing': {
                'file_path': 'data/raw/housing.csv',
                'target_column': 'MEDV',
                'numerical_columns': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
            }
        },
        'preprocessing': {
            'test_size': 0.2,
            'random_state': 42,
            'validation_size': 0.2
        }
    }
    
    with open("configs/data_config.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    # Model configuration
    model_config = {
        'classification': {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000],
                'solver': ['liblinear', 'lbfgs']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'regression': {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'linear_regression': {},
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    with open("configs/model_config.yaml", "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    
    # Pipeline configuration
    pipeline_config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/pipeline.log'
        },
        'mlflow': {
            'tracking_uri': 'mlruns',
            'artifact_location': 'mlruns'
        },
        'evaluation': {
            'cv_folds': 5,
            'scoring_metrics': {
                'classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'regression': ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            }
        }
    }
    
    with open("configs/pipeline_config.yaml", "w") as f:
        yaml.dump(pipeline_config, f, default_flow_style=False)
    
    # Feature engineering configuration
    feature_config = {
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
                'RoomsPerHousehold': 'RM / (RM + 1)',  # Simplified
                'CrimeLevel': 'binned CRIM',
                'AgeGroup': 'binned AGE'
            },
            'feature_selection': {
                'method': 'selectkbest',
                'k': 12
            }
        }
    }
    
    with open("configs/feature_config.yaml", "w") as f:
        yaml.dump(feature_config, f, default_flow_style=False)
    
    logger.info("✅ Configuration files created successfully")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/
ml_pipeline_env/

# Jupyter Notebook
.ipynb_checkpoints

# MLflow
mlruns/
mlartifacts/

# DVC
/data/raw/titanic.csv
/data/raw/housing.csv
/data/processed/
/data/features/
/models/

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment variables
.env
.env.local

# Temporary files
*.tmp
*.temp
temp/

# Model artifacts (large files)
*.pkl
*.joblib
*.h5
*.pb

# Data files (managed by DVC)
data/raw/*.csv
data/processed/*.csv
data/features/*.csv
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    logger.info("✅ .gitignore file created")

def create_docker_files():
    """Create Docker configuration files"""
    logger.info("Creating Docker configuration files...")
    
    # Dockerfile
    dockerfile_content = """FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/features

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port for MLflow UI
EXPOSE 5000

# Default command
CMD ["python", "scripts/run_pipeline.py", "--help"]
"""
    
    with open("docker/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Docker Compose
    docker_compose_content = """version: '3.8'

services:
  ml-pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../mlruns:/app/mlruns
      - ../logs:/app/logs
    environment:
      - PYTHONPATH=/app/src
    ports:
      - "5000:5000"
    command: ["python", "scripts/run_pipeline.py", "--dataset", "titanic"]
  
  mlflow-ui:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../mlruns:/app/mlruns
    ports:
      - "5001:5000"
    command: ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
    
  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../:/app
    ports:
      - "8888:8888"
    command: ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
"""
    
    with open("docker/docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    logger.info("✅ Docker configuration files created")

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'mlflow', 'dvc', 'jupyter', 'pytest', 'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} is not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def main():
    """Main setup function"""
    logger.info("🚀 Starting ML Pipeline environment setup...")
    
    try:
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed. Please install required packages.")
            return 1
        
        # Create directory structure
        create_directory_structure()
        
        # Initialize DVC
        if not initialize_dvc():
            logger.warning("⚠️ DVC initialization failed, but continuing...")
        
        # Initialize MLflow
        if not initialize_mlflow():
            logger.error("❌ MLflow initialization failed")
            return 1
        
        # Create configuration files
        create_config_files()
        
        # Create .gitignore
        create_gitignore()
        
        # Create Docker files
        create_docker_files()
        
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Download datasets: python scripts/download_data.py")
        logger.info("2. Run pipeline: python scripts/run_pipeline.py --dataset titanic")
        logger.info("3. Start MLflow UI: mlflow ui")
        logger.info("4. Open Jupyter notebooks: jupyter notebook")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)