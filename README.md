# 🚀 Building Machine Learning Pipelines with MLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Beginner Friendly](https://img.shields.io/badge/Beginner-Friendly-brightgreen.svg)

**🔄 Build ML Pipelines • 📊 Version Data • 🧪 Track Experiments • 🏛️ Manage Models**

</div>

---

## 📖 Table of Contents

- [🌟 What is This Project?](#-what-is-this-project)
- [🎓 What You'll Learn](#-what-youll-learn)
- [🏗️ Project Architecture](#️-project-architecture)
- [📁 Project Structure Explained](#-project-structure-explained)
- [🛠️ Setup Instructions](#️-setup-instructions)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📊 Datasets We'll Use](#-datasets-well-use)
- [🔄 ML Pipeline Walkthrough](#-ml-pipeline-walkthrough)
- [📊 Data Versioning with DVC](#-data-versioning-with-dvc)
- [🧪 Experiment Tracking with MLflow](#-experiment-tracking-with-mlflow)
- [🏛️ Model Registry](#️-model-registry)
- [🐳 Docker Deployment](#-docker-deployment)
- [📚 Jupyter Tutorials](#-jupyter-tutorials)
- [🧪 Testing](#-testing)
- [💡 Tips for Beginners](#-tips-for-beginners)
- [🔧 Troubleshooting](#-troubleshooting)
- [🎯 Next Steps](#-next-steps)

---

## 🌟 What is This Project?

This is a **comprehensive, beginner-friendly project** that teaches you how to build **production-ready machine learning pipelines** with modern MLOps practices. You'll learn to manage data, track experiments, and deploy models like a professional ML engineer!

### 🎯 **Perfect for:**
- 👨‍💻 **Data Scientists** learning MLOps
- 🎓 **Students** studying machine learning
- 🚀 **Engineers** transitioning to ML
- 🔬 **Anyone** wanting to build professional ML systems

### 🆓 **Why This Project?**
- **Real Datasets** - Work with Titanic, Boston Housing, and more
- **Industry Tools** - MLflow, DVC, Docker (all free!)
- **End-to-End** - From raw data to deployed model
- **Production Ready** - Best practices and professional workflows
- **Step-by-Step** - Detailed tutorials for every concept

### 🔄 **What Makes This Special?**
Unlike simple ML tutorials, this project teaches you the **complete ML lifecycle** that companies actually use in production!

---

## 🎓 What You'll Learn

By completing this project, you'll master:

| 📚 **MLOps Concept** | 🎯 **What You'll Master** | 🛠️ **Tools Used** |
|---------------------|---------------------------|-------------------|
| **🔄 ML Pipelines** | Automated, reproducible workflows | Scikit-learn, Custom classes |
| **📊 Data Versioning** | Track data changes and lineage | DVC (Data Version Control) |
| **🧪 Experiment Tracking** | Log parameters, metrics, models | MLflow Tracking |
| **🏛️ Model Registry** | Manage model lifecycle | MLflow Model Registry |
| **🐳 Containerization** | Package ML apps for deployment | Docker |
| **📈 Model Evaluation** | Compare and validate models | Scikit-learn metrics |
| **🔧 Pipeline Orchestration** | Automate ML workflows | Custom pipeline classes |
| **📊 Data Quality** | Validate and monitor data | Pandas, custom validators |

---

## 🏗️ Project Architecture

Our ML system follows industry best practices:

```mermaid
graph TD
    A[Raw Data] --> B[Data Versioning - DVC]
    B --> C[Data Pipeline]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Experiment Tracking - MLflow]
    F --> G[Model Evaluation]
    G --> H[Model Registry]
    H --> I[Model Deployment - Docker]
    I --> J[Production Model]
    
    K[Data Validation] --> C
    L[Hyperparameter Tuning] --> E
    M[Model Monitoring] --> J
```

### 🔄 **Pipeline Flow:**
1. **📥 Data Ingestion** → Load and validate raw data
2. **📊 Data Versioning** → Track data with DVC
3. **🧹 Data Processing** → Clean and engineer features
4. **🤖 Model Training** → Train multiple algorithms
5. **🧪 Experiment Tracking** → Log everything with MLflow
6. **📈 Model Evaluation** → Compare model performance
7. **🏛️ Model Registry** → Store and version best models
8. **🐳 Deployment** → Package with Docker
9. **📊 Monitoring** → Track model performance

---

## 📁 Project Structure Explained

Let's understand every file and folder in our ML pipeline project:

```
📦 Building-Machine-Learning-Pipelines/
├── 📄 README.md                    # 👈 You are here! Complete documentation
├── 📄 requirements.txt             # 📦 Python dependencies
├── 📄 dvc.yaml                     # 🔄 DVC pipeline definition
├── 📄 params.yaml                  # ⚙️ Pipeline parameters
├── 📄 docker-compose.yml           # 🐳 Docker orchestration
├── 📄 .dvcignore                   # 🚫 DVC ignore rules
├── 📄 .gitignore                   # 🚫 Git ignore rules
│
├── 📁 data/                        # 📊 Data storage (DVC tracked)
│   ├── 📁 raw/                    # 🗃️ Original datasets
│   ├── 📁 processed/              # 🧹 Cleaned data
│   └── 📁 features/               # 🔧 Engineered features
│
├── 📁 notebooks/                   # 📚 Jupyter tutorials
│   ├── 📄 01_data_exploration.ipynb      # 🔍 Data analysis
│   ├── 📄 02_feature_engineering.ipynb   # 🔧 Feature creation
│   ├── 📄 03_model_training.ipynb        # 🤖 Model development
│   ├── 📄 04_experiment_tracking.ipynb   # 🧪 MLflow tutorial
│   └── 📄 05_model_deployment.ipynb      # 🚀 Deployment guide
│
├── 📁 src/                         # 💻 Source code
│   ├── 📄 __init__.py             # 🐍 Package marker
│   ├── 📁 data/                   # 📊 Data processing
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_loader.py      # 📥 Data loading utilities
│   │   ├── 📄 data_validator.py   # ✅ Data quality checks
│   │   └── 📄 preprocessor.py     # 🧹 Data preprocessing
│   │
│   ├── 📁 features/               # 🔧 Feature engineering
│   │   ├── 📄 __init__.py
│   │   ├── 📄 feature_engineer.py # 🔧 Feature creation
│   │   └── 📄 feature_selector.py # 🎯 Feature selection
│   │
│   ├── 📁 models/                 # 🤖 Model definitions
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_model.py       # 🏗️ Base model class
│   │   ├── 📄 classifiers.py      # 📊 Classification models
│   │   └── 📄 regressors.py       # 📈 Regression models
│   │
│   ├── 📁 evaluation/             # 📈 Model evaluation
│   │   ├── 📄 __init__.py
│   │   ├── 📄 metrics.py          # 📊 Evaluation metrics
│   │   └── 📄 validator.py        # ✅ Model validation
│   │
│   ├── 📁 pipeline/               # 🔄 Pipeline orchestration
│   │   ├── 📄 __init__.py
│   │   ├── 📄 training_pipeline.py # 🎓 Training workflow
│   │   └── 📄 prediction_pipeline.py # 🔮 Prediction workflow
│   │
│   └── 📁 utils/                  # 🛠️ Utility functions
│       ├── 📄 __init__.py
│       ├── 📄 config.py           # ⚙️ Configuration management
│       ├── 📄 logger.py           # 📝 Logging utilities
│       └── 📄 mlflow_utils.py     # 🧪 MLflow helpers
│
├── 📁 experiments/                 # 🧪 Experiment tracking
│   └── 📄 experiment_config.yaml  # ⚙️ Experiment settings
│
├── 📁 models/                      # 🏛️ Model registry
│   ├── 📁 titanic/                # 🚢 Titanic models
│   └── 📁 housing/                # 🏠 Housing models
│
├── 📁 docker/                      # 🐳 Docker configurations
│   ├── 📄 Dockerfile              # 🐳 Container definition
│   ├── 📄 docker-compose.yml      # 🎼 Multi-container setup
│   └── 📄 requirements.txt        # 📦 Container dependencies
│
├── 📁 tests/                       # 🧪 Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_data_processing.py # 📊 Data tests
│   ├── 📄 test_models.py          # 🤖 Model tests
│   └── 📄 test_pipeline.py        # 🔄 Pipeline tests
│
├── 📁 configs/                     # ⚙️ Configuration files
│   ├── 📄 data_config.yaml        # 📊 Data settings
│   ├── 📄 model_config.yaml       # 🤖 Model parameters
│   └── 📄 pipeline_config.yaml    # 🔄 Pipeline settings
│
└── 📁 scripts/                     # 🔧 Utility scripts
    ├── 📄 setup_environment.py    # 🛠️ Environment setup
    ├── 📄 download_data.py        # 📥 Data download
    └── 📄 run_pipeline.py         # 🚀 Pipeline execution
```

### 🔍 **Key Components Explained:**

#### 📊 **Data Layer (`data/` + `src/data/`)**
- **Raw data storage** with DVC versioning
- **Data validation** to ensure quality
- **Preprocessing pipelines** for cleaning
- **Feature engineering** for model input

#### 🤖 **Model Layer (`src/models/` + `models/`)**
- **Model definitions** for different algorithms
- **Training pipelines** with hyperparameter tuning
- **Model registry** for version management
- **Evaluation metrics** for comparison

#### 🧪 **Experiment Layer (`experiments/` + MLflow)**
- **Experiment tracking** with MLflow
- **Parameter logging** for reproducibility
- **Metric comparison** across runs
- **Artifact storage** for models and plots

#### 🐳 **Deployment Layer (`docker/`)**
- **Containerization** for consistent environments
- **API endpoints** for model serving
- **Local deployment** setup
- **Production-ready** configurations

---

## 🛠️ Setup Instructions

### 📋 **Prerequisites:**
- 🐍 **Python 3.8 or higher** ([Download here](https://python.org))
- 🐳 **Docker** ([Download here](https://docker.com))
- 💻 **Git** for version control
- 🌐 **Internet connection** for data download

### 🚀 **Step 1: Clone the Repository**
```bash
# Download the project
git clone https://github.com/Amruth22/Building-Machine-Learning-Pipelines.git

# Enter the project folder
cd Building-Machine-Learning-Pipelines
```

### 🏠 **Step 2: Create Virtual Environment**
```bash
# Create virtual environment (highly recommended!)
python -m venv ml_pipeline_env

# Activate it:
# On Windows:
ml_pipeline_env\Scripts\activate

# On Mac/Linux:
source ml_pipeline_env/bin/activate

# You should see (ml_pipeline_env) in your terminal!
```

### 📦 **Step 3: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# This installs:
# - scikit-learn (ML algorithms)
# - pandas, numpy (data processing)
# - mlflow (experiment tracking)
# - dvc (data versioning)
# - jupyter (notebooks)
# - docker (containerization)
# - pytest (testing)
```

### 🔧 **Step 4: Initialize MLOps Tools**
```bash
# Initialize DVC for data versioning
dvc init

# Initialize MLflow tracking
python scripts/setup_environment.py

# Download datasets
python scripts/download_data.py
```

### ✅ **Step 5: Verify Installation**
```bash
# Run tests to ensure everything works
pytest tests/ -v

# Start Jupyter for tutorials
jupyter notebook notebooks/

# Check MLflow UI
mlflow ui
```

---

## 🚀 Quick Start Guide

### 🎯 **Option 1: Run Complete Pipeline**
```bash
# Execute the full ML pipeline
python scripts/run_pipeline.py --dataset titanic

# This will:
# 1. Load and validate data
# 2. Engineer features
# 3. Train multiple models
# 4. Track experiments in MLflow
# 5. Register best model
# 6. Generate evaluation report
```

### 📚 **Option 2: Follow Jupyter Tutorials**
```bash
# Start Jupyter notebook server
jupyter notebook

# Open tutorials in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_feature_engineering.ipynb
# 3. notebooks/03_model_training.ipynb
# 4. notebooks/04_experiment_tracking.ipynb
# 5. notebooks/05_model_deployment.ipynb
```

### 🐳 **Option 3: Docker Deployment**
```bash
# Build and run with Docker
docker-compose up --build

# Access the deployed model API
curl http://localhost:8000/predict \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4, 5]}'
```

---

## 📊 Datasets We'll Use

### 🚢 **1. Titanic Dataset (Classification)**
**Problem**: Predict passenger survival  
**Features**: Age, Sex, Class, Fare, etc.  
**Target**: Survived (0/1)  
**Size**: 891 samples, 12 features

```python
# Example data point
{
  "PassengerId": 1,
  "Pclass": 3,
  "Name": "Braund, Mr. Owen Harris",
  "Sex": "male",
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Ticket": "A/5 21171",
  "Fare": 7.2500,
  "Cabin": null,
  "Embarked": "S",
  "Survived": 0  # Target
}
```

### 🏠 **2. Boston Housing Dataset (Regression)**
**Problem**: Predict house prices  
**Features**: Crime rate, Rooms, Age, etc.  
**Target**: Price in $1000s  
**Size**: 506 samples, 13 features

```python
# Example data point
{
  "CRIM": 0.00632,     # Crime rate
  "ZN": 18.0,          # Residential land zoned
  "INDUS": 2.31,       # Non-retail business acres
  "CHAS": 0,           # Charles River dummy
  "NOX": 0.538,        # Nitric oxides concentration
  "RM": 6.575,         # Average rooms per dwelling
  "AGE": 65.2,         # Age of owner-occupied units
  "DIS": 4.0900,       # Distance to employment centers
  "RAD": 1,            # Accessibility to highways
  "TAX": 296,          # Property tax rate
  "PTRATIO": 15.3,     # Pupil-teacher ratio
  "B": 396.90,         # Proportion of blacks
  "LSTAT": 4.98,       # Lower status population
  "MEDV": 24.0         # Target: Median home value
}
```

### 📈 **Why These Datasets?**
- **🎓 Educational**: Perfect for learning ML concepts
- **🔍 Well-studied**: Lots of resources and benchmarks
- **🌟 Diverse**: Classification and regression problems
- **📊 Real-world**: Actual data with real challenges
- **🔧 Feature-rich**: Great for feature engineering practice

---

## 🔄 ML Pipeline Walkthrough

Let's walk through our complete ML pipeline step by step:

### 📥 **Stage 1: Data Ingestion**
```python
# src/data/data_loader.py
class DataLoader:
    def load_titanic(self):
        """Load Titanic dataset with validation"""
        data = pd.read_csv('data/raw/titanic.csv')
        self.validate_data(data)
        return data
    
    def validate_data(self, data):
        """Ensure data quality"""
        assert not data.empty, "Dataset is empty!"
        assert 'Survived' in data.columns, "Target column missing!"
        logging.info(f"Loaded {len(data)} samples")
```

### 🧹 **Stage 2: Data Preprocessing**
```python
# src/data/preprocessor.py
class DataPreprocessor:
    def clean_data(self, data):
        """Clean and prepare data"""
        # Handle missing values
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        
        # Remove outliers
        data = self.remove_outliers(data)
        
        return data
```

### 🔧 **Stage 3: Feature Engineering**
```python
# src/features/feature_engineer.py
class FeatureEngineer:
    def create_features(self, data):
        """Create new features"""
        # Family size
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        
        # Title extraction
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Age groups
        data['AgeGroup'] = pd.cut(data['Age'], bins=5, labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        return data
```

### 🤖 **Stage 4: Model Training**
```python
# src/models/classifiers.py
class TitanicClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'logistic_regression': LogisticRegression(),
            'gradient_boosting': GradientBoostingClassifier()
        }
    
    def train_all_models(self, X_train, y_train):
        """Train multiple models and compare"""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = model
        return results
```

### 🧪 **Stage 5: Experiment Tracking**
```python
# src/utils/mlflow_utils.py
class MLflowTracker:
    def log_experiment(self, model_name, params, metrics, model):
        """Log experiment to MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Log artifacts
            mlflow.log_artifact("plots/confusion_matrix.png")
```

### 📈 **Stage 6: Model Evaluation**
```python
# src/evaluation/metrics.py
class ModelEvaluator:
    def evaluate_classification(self, y_true, y_pred):
        """Comprehensive classification evaluation"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
```

---

## 📊 Data Versioning with DVC

### 🎯 **What is DVC?**
**DVC (Data Version Control)** is like Git for data. It tracks changes in your datasets, making your ML experiments reproducible.

### 🔧 **DVC Setup:**
```bash
# Initialize DVC in your project
dvc init

# Add data to DVC tracking
dvc add data/raw/titanic.csv

# Commit DVC files to Git
git add data/raw/titanic.csv.dvc .dvcignore
git commit -m "Add Titanic dataset to DVC"
```

### 📋 **DVC Pipeline Definition (`dvc.yaml`):**
```yaml
stages:
  data_preprocessing:
    cmd: python src/data/preprocessor.py
    deps:
    - data/raw/titanic.csv
    - src/data/preprocessor.py
    outs:
    - data/processed/titanic_clean.csv
    
  feature_engineering:
    cmd: python src/features/feature_engineer.py
    deps:
    - data/processed/titanic_clean.csv
    - src/features/feature_engineer.py
    outs:
    - data/features/titanic_features.csv
    
  model_training:
    cmd: python src/models/train_model.py
    deps:
    - data/features/titanic_features.csv
    - src/models/train_model.py
    params:
    - model_config.yaml:random_forest
    outs:
    - models/titanic/random_forest.pkl
    metrics:
    - metrics/titanic_metrics.json
```

### 🚀 **Running DVC Pipeline:**
```bash
# Run the entire pipeline
dvc repro

# Run specific stage
dvc repro model_training

# Check pipeline status
dvc status

# Compare experiments
dvc metrics diff
```

### 📊 **Data Versioning Benefits:**
- **🔄 Reproducibility**: Exact same data for every experiment
- **📈 Lineage**: Track how data changes over time
- **🔍 Comparison**: Compare results across data versions
- **💾 Storage**: Efficient storage of large datasets
- **🤝 Collaboration**: Share data versions with team

---

## 🧪 Experiment Tracking with MLflow

### 🎯 **What is MLflow?**
**MLflow** is a platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.

### 🚀 **MLflow Setup:**
```bash
# Start MLflow tracking server
mlflow ui

# Access at: http://localhost:5000
```

### 📊 **Tracking Experiments:**
```python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("Titanic Survival Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log artifacts
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### 📈 **Comparing Experiments:**
```python
# Search experiments
experiments = mlflow.search_runs(experiment_ids=["1"])

# Compare metrics
best_run = experiments.loc[experiments['metrics.accuracy'].idxmax()]
print(f"Best accuracy: {best_run['metrics.accuracy']}")
print(f"Best parameters: {best_run['params.n_estimators']}")
```

### 🏆 **Benefits of Experiment Tracking:**
- **📊 Comparison**: Compare different models and parameters
- **🔄 Reproducibility**: Recreate any experiment exactly
- **📈 Visualization**: See metrics and plots over time
- **🤝 Collaboration**: Share experiments with team
- **🎯 Optimization**: Find best hyperparameters

---

## 🏛️ Model Registry

### 🎯 **What is Model Registry?**
The **Model Registry** is a centralized store for managing model versions, stages, and metadata.

### 📋 **Model Lifecycle Stages:**
1. **🔬 Staging** - Model under testing
2. **✅ Production** - Model serving users
3. **📦 Archived** - Old model versions

### 🚀 **Registering Models:**
```python
import mlflow.sklearn

# Register model from run
model_uri = f"runs:/{run_id}/random_forest_model"
mlflow.register_model(model_uri, "TitanicSurvivalModel")

# Or register directly
mlflow.sklearn.log_model(
    model, 
    "random_forest_model",
    registered_model_name="TitanicSurvivalModel"
)
```

### 🔄 **Managing Model Versions:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to production
client.transition_model_version_stage(
    name="TitanicSurvivalModel",
    version=1,
    stage="Production"
)

# Add model description
client.update_model_version(
    name="TitanicSurvivalModel",
    version=1,
    description="Random Forest model with 85% accuracy"
)

# Get production model
production_model = mlflow.sklearn.load_model(
    model_uri="models:/TitanicSurvivalModel/Production"
)
```

### 📊 **Model Registry Benefits:**
- **🏛️ Centralized**: All models in one place
- **📈 Versioning**: Track model evolution
- **🔄 Lifecycle**: Manage staging to production
- **📝 Metadata**: Store model information
- **🤝 Collaboration**: Team model management

---

## 🐳 Docker Deployment

### 🎯 **Why Docker?**
**Docker** ensures your ML model runs consistently across different environments.

### 📋 **Dockerfile:**
```dockerfile
# docker/Dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "src/api/app.py"]
```

### 🎼 **Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: 
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=TitanicSurvivalModel
      - MODEL_STAGE=Production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    
  mlflow:
    image: python:3.8-slim
    command: >
      bash -c "pip install mlflow && 
               mlflow server --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
```

### 🚀 **Deployment Commands:**
```bash
# Build and run containers
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs ml-api

# Stop containers
docker-compose down
```

### 🌐 **API Endpoints:**
```python
# src/api/app.py
from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load production model
model = mlflow.sklearn.load_model("models:/TitanicSurvivalModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    data = request.get_json()
    features = data['features']
    
    prediction = model.predict([features])
    probability = model.predict_proba([features])
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability[0][1])
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 🧪 **Testing Deployment:**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [3, 1, 22, 1, 0, 7.25, 0, 1, 0]
  }'

# Response:
# {
#   "prediction": 0,
#   "probability": 0.23
# }
```

---

## 📚 Jupyter Tutorials

Our project includes comprehensive Jupyter notebook tutorials:

### 📖 **Tutorial Sequence:**

#### 📊 **1. Data Exploration (`01_data_exploration.ipynb`)**
- Load and examine datasets
- Understand data distributions
- Identify missing values and outliers
- Create visualizations
- Generate data quality reports

#### 🔧 **2. Feature Engineering (`02_feature_engineering.ipynb`)**
- Handle missing values
- Create new features
- Encode categorical variables
- Scale numerical features
- Select important features

#### 🤖 **3. Model Training (`03_model_training.ipynb`)**
- Split data into train/test sets
- Train multiple algorithms
- Perform hyperparameter tuning
- Cross-validation
- Model comparison

#### 🧪 **4. Experiment Tracking (`04_experiment_tracking.ipynb`)**
- Set up MLflow experiments
- Log parameters and metrics
- Compare experiment runs
- Visualize results
- Select best models

#### 🚀 **5. Model Deployment (`05_model_deployment.ipynb`)**
- Register models in MLflow
- Create prediction pipelines
- Build Docker containers
- Test deployed models
- Monitor performance

### 🎯 **How to Use Tutorials:**
```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Open tutorials in order (01, 02, 03, 04, 05)
# Run cells step by step
# Experiment with different parameters
```

### 💡 **Tutorial Features:**
- **📝 Detailed explanations** for every step
- **💻 Runnable code** with real datasets
- **📊 Visualizations** to understand concepts
- **🧪 Exercises** to practice skills
- **🔗 Links** to additional resources

---

## 🧪 Testing

### 🎯 **Testing Strategy:**
We use **pytest** for comprehensive testing of our ML pipeline.

### 📋 **Test Categories:**

#### 🧪 **Unit Tests:**
```python
# tests/test_data_processing.py
def test_data_loader():
    """Test data loading functionality"""
    loader = DataLoader()
    data = loader.load_titanic()
    
    assert not data.empty
    assert 'Survived' in data.columns
    assert len(data) > 0

def test_preprocessor():
    """Test data preprocessing"""
    preprocessor = DataPreprocessor()
    raw_data = pd.read_csv('data/raw/titanic.csv')
    clean_data = preprocessor.clean_data(raw_data)
    
    # Check no missing values in critical columns
    assert clean_data['Age'].isnull().sum() == 0
    assert clean_data['Embarked'].isnull().sum() == 0
```

#### 🤖 **Model Tests:**
```python
# tests/test_models.py
def test_model_training():
    """Test model training process"""
    X, y = load_sample_data()
    model = TitanicClassifier()
    
    trained_models = model.train_all_models(X, y)
    
    assert 'random_forest' in trained_models
    assert hasattr(trained_models['random_forest'], 'predict')

def test_model_predictions():
    """Test model predictions"""
    model = load_trained_model()
    X_test = load_test_data()
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)
```

#### 🔄 **Pipeline Tests:**
```python
# tests/test_pipeline.py
def test_training_pipeline():
    """Test complete training pipeline"""
    pipeline = TrainingPipeline()
    results = pipeline.run('titanic')
    
    assert 'model' in results
    assert 'metrics' in results
    assert results['metrics']['accuracy'] > 0.7

def test_prediction_pipeline():
    """Test prediction pipeline"""
    pipeline = PredictionPipeline()
    sample_data = get_sample_input()
    
    prediction = pipeline.predict(sample_data)
    
    assert prediction in [0, 1]
```

### 🚀 **Running Tests:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests in parallel
pytest tests/ -n 4
```

### 📊 **Test Coverage:**
```bash
# Generate coverage report
coverage run -m pytest tests/
coverage report
coverage html

# View HTML report
open htmlcov/index.html
```

---

## 💡 Tips for Beginners

### 🎯 **Getting Started:**

1. **📚 Start with Notebooks**
   ```
   Begin with Jupyter tutorials:
   - 01_data_exploration.ipynb (understand your data)
   - 02_feature_engineering.ipynb (create features)
   - 03_model_training.ipynb (build models)
   ```

2. **🔍 Understand Your Data First**
   ```python
   # Always explore before modeling
   data.info()                    # Data types and missing values
   data.describe()                # Statistical summary
   data.hist(figsize=(12, 8))     # Distributions
   data.corr()                    # Correlations
   ```

3. **🧪 Start Simple, Then Improve**
   ```python
   # Begin with simple models
   from sklearn.linear_model import LogisticRegression
   simple_model = LogisticRegression()
   
   # Then try complex ones
   from sklearn.ensemble import RandomForestClassifier
   complex_model = RandomForestClassifier()
   ```

### 🔧 **MLOps Best Practices:**

1. **📊 Version Everything**
   ```bash
   # Version data with DVC
   dvc add data/raw/dataset.csv
   
   # Version code with Git
   git add src/models/new_model.py
   git commit -m "Add new model"
   
   # Version models with MLflow
   mlflow.sklearn.log_model(model, "model_v1")
   ```

2. **🧪 Track All Experiments**
   ```python
   # Always log experiments
   with mlflow.start_run():
       mlflow.log_params({"n_estimators": 100})
       mlflow.log_metrics({"accuracy": 0.85})
       mlflow.sklearn.log_model(model, "model")
   ```

3. **✅ Test Your Code**
   ```python
   # Write tests for critical functions
   def test_data_preprocessing():
       assert preprocessed_data.isnull().sum().sum() == 0
   
   def test_model_performance():
       assert model_accuracy > 0.7
   ```

### 🚀 **Advanced Tips:**

1. **🔄 Automate Your Pipeline**
   ```bash
   # Use DVC for automation
   dvc repro  # Runs entire pipeline
   
   # Use scripts for common tasks
   python scripts/run_pipeline.py --dataset titanic
   ```

2. **📈 Monitor Model Performance**
   ```python
   # Track model drift
   from evidently import ColumnMapping
   from evidently.dashboard import Dashboard
   
   dashboard = Dashboard(tabs=[DataDriftTab()])
   dashboard.calculate(reference_data, current_data)
   ```

3. **🐳 Containerize Everything**
   ```bash
   # Build consistent environments
   docker build -t ml-model .
   docker run -p 8000:8000 ml-model
   ```

---

## 🔧 Troubleshooting

### 🚨 **Common Issues and Solutions:**

#### ❌ **"ModuleNotFoundError" Errors**
```bash
# Problem: Python can't find your modules
# Solution 1: Install in development mode
pip install -e .

# Solution 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Solution 3: Use absolute imports
from src.data.data_loader import DataLoader
```

#### ❌ **"DVC Command Not Found"**
```bash
# Problem: DVC not installed or not in PATH
# Solution: Install DVC
pip install dvc

# Or with specific remote support
pip install 'dvc[s3]'  # For AWS S3
pip install 'dvc[gs]'  # For Google Cloud
```

#### ❌ **"MLflow Server Won't Start"**
```bash
# Problem: Port already in use
# Solution 1: Use different port
mlflow ui --port 5001

# Solution 2: Kill existing process
lsof -ti:5000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :5000   # Windows

# Solution 3: Check if MLflow is installed
pip install mlflow
```

#### ❌ **"Docker Build Failed"**
```bash
# Problem: Docker build issues
# Solution 1: Check Docker is running
docker --version

# Solution 2: Clean Docker cache
docker system prune -a

# Solution 3: Check Dockerfile syntax
docker build --no-cache -t ml-model .
```

#### ❌ **"Out of Memory" Errors**
```bash
# Problem: Large datasets causing memory issues
# Solution 1: Use data sampling
data_sample = data.sample(n=10000)

# Solution 2: Process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    process_chunk(chunk)

# Solution 3: Use more efficient data types
data['category_col'] = data['category_col'].astype('category')
```

#### ❌ **"Model Performance is Poor"**
```python
# Problem: Low model accuracy
# Solution 1: Check data quality
print(data.isnull().sum())  # Missing values
print(data.describe())      # Data distribution

# Solution 2: Try feature engineering
data['new_feature'] = data['feature1'] * data['feature2']

# Solution 3: Tune hyperparameters
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
```

### 🆘 **Getting Help:**

1. **📖 Check Logs**: Always look at error messages first
2. **🔍 Search Issues**: Check GitHub issues for similar problems
3. **📚 Read Documentation**: MLflow, DVC, and scikit-learn docs
4. **💬 Ask Community**: Stack Overflow, Reddit r/MachineLearning
5. **🐛 Report Bugs**: Create issues in this repository

---

## 🎯 Next Steps

### 🚀 **Enhance This Project:**

1. **📊 Add More Datasets:**
   ```python
   # Implement new datasets
   - Wine Quality (Classification)
   - California Housing (Regression)  
   - Iris (Multi-class Classification)
   - Customer Segmentation (Clustering)
   ```

2. **🤖 Advanced Models:**
   ```python
   # Try advanced algorithms
   - XGBoost, LightGBM
   - Neural Networks (TensorFlow/PyTorch)
   - Ensemble methods
   - AutoML (Auto-sklearn, TPOT)
   ```

3. **☁️ Cloud Deployment:**
   ```bash
   # Deploy to cloud platforms
   - AWS SageMaker
   - Google Cloud AI Platform
   - Azure Machine Learning
   - Heroku
   ```

4. **📊 Advanced MLOps:**
   ```python
   # Implement advanced features
   - Model monitoring and drift detection
   - A/B testing for models
   - Continuous training pipelines
   - Feature stores
   ```

### 📚 **Learn More About:**

- **🔄 MLOps**: [MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
- **🐳 Docker**: [Docker for Data Science](https://docker-curriculum.com/)
- **☁️ Cloud ML**: [AWS ML Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/)
- **📊 Advanced ML**: [Fast.ai Courses](https://www.fast.ai/)
- **🔧 Production ML**: [Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)

### 🎓 **Career Paths:**

1. **🤖 Machine Learning Engineer** - Build and deploy ML systems
2. **📊 Data Scientist** - Extract insights from data
3. **🔧 MLOps Engineer** - Manage ML infrastructure
4. **📈 AI Product Manager** - Guide AI product development
5. **🔬 Research Scientist** - Advance ML algorithms

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🎯 **Ways to Contribute:**

- 🐛 **Report Bugs**: Found an issue? Create an issue!
- ✨ **Add Features**: New models, datasets, or tools
- 📝 **Improve Documentation**: Make tutorials even better
- 🧪 **Add Tests**: More tests = more reliable code
- 🎨 **Enhance Visualizations**: Better plots and dashboards

### 📋 **Contribution Guidelines:**

1. **🍴 Fork the repository**
2. **🌿 Create a feature branch**: `git checkout -b feature/amazing-ml-feature`
3. **💾 Commit your changes**: `git commit -m 'Add amazing ML feature'`
4. **📤 Push to branch**: `git push origin feature/amazing-ml-feature`
5. **🔄 Open a Pull Request**

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 🎁 **What this means:**
- ✅ **Free to use** for personal and commercial projects
- ✅ **Free to modify** and distribute
- ✅ **No warranty** - use at your own risk
- ✅ **Attribution appreciated** but not required

---

## 🙏 Acknowledgments

- **🤖 Scikit-learn** team for excellent ML library
- **🧪 MLflow** team for experiment tracking tools
- **📊 DVC** team for data versioning solution
- **🐳 Docker** for containerization platform
- **📚 Jupyter** team for interactive notebooks
- **🐍 Python** community for amazing ecosystem

---

<div align="center">

### 🌟 **Star this repository if it helped you learn MLOps!** ⭐

**Made with ❤️ for aspiring ML engineers**

[🔝 Back to Top](#-building-machine-learning-pipelines-with-mlops)

</div>