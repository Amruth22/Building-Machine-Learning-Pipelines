#!/bin/bash
# =============================================================================
# Docker Entrypoint Script for ML Pipeline
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting ML Pipeline Container...${NC}"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo -e "${YELLOW}⏳ Waiting for $service_name at $host:$port...${NC}"
    
    while ! nc -z $host $port; do
        sleep 1
    done
    
    echo -e "${GREEN}✅ $service_name is ready!${NC}"
}

# Function to setup data directories
setup_directories() {
    echo -e "${BLUE}📁 Setting up directories...${NC}"
    
    mkdir -p data/{raw,processed,features}
    mkdir -p models trained_models
    mkdir -p plots reports logs
    mkdir -p mlruns
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Function to download datasets if missing
setup_data() {
    echo -e "${BLUE}📊 Checking datasets...${NC}"
    
    if [ ! -f "data/raw/titanic.csv" ] || [ ! -f "data/raw/housing.csv" ]; then
        echo -e "${YELLOW}📥 Downloading missing datasets...${NC}"
        python download_datasets.py || echo -e "${YELLOW}⚠️ Dataset download failed, will use fallback${NC}"
    else
        echo -e "${GREEN}✅ Datasets found${NC}"
    fi
}

# Function to run data pipeline
run_pipeline() {
    echo -e "${BLUE}🔄 Running data pipeline...${NC}"
    
    # Check if DVC is available
    if command -v dvc &> /dev/null; then
        echo -e "${BLUE}Running DVC pipeline...${NC}"
        dvc repro || echo -e "${YELLOW}⚠️ DVC pipeline failed, continuing...${NC}"
    else
        echo -e "${YELLOW}DVC not available, running manual pipeline...${NC}"
        
        # Run manual pipeline steps
        python src/data/data_loader.py || true
        python src/data/preprocessor.py || true
        python src/features/feature_engineer.py || true
    fi
}

# Function to train models if needed
train_models() {
    if [ "$TRAIN_MODELS" = "true" ]; then
        echo -e "${BLUE}🤖 Training models...${NC}"
        python train_models_simple.py || echo -e "${YELLOW}⚠️ Model training failed${NC}"
    fi
}

# Main execution based on command
case "$1" in
    "web")
        setup_directories
        setup_data
        
        # Wait for MLflow if configured
        if [ ! -z "$MLFLOW_TRACKING_URI" ]; then
            MLFLOW_HOST=$(echo $MLFLOW_TRACKING_URI | sed 's|http://||' | cut -d':' -f1)
            MLFLOW_PORT=$(echo $MLFLOW_TRACKING_URI | sed 's|http://||' | cut -d':' -f2)
            wait_for_service $MLFLOW_HOST $MLFLOW_PORT "MLflow"
        fi
        
        echo -e "${GREEN}🌐 Starting web application...${NC}"
        exec python web_app.py
        ;;
        
    "train")
        setup_directories
        setup_data
        run_pipeline
        
        echo -e "${GREEN}🤖 Starting model training...${NC}"
        exec python train_models_simple.py
        ;;
        
    "pipeline")
        setup_directories
        setup_data
        
        echo -e "${GREEN}🔄 Running complete pipeline...${NC}"
        run_pipeline
        train_models
        ;;
        
    "jupyter")
        setup_directories
        setup_data
        
        echo -e "${GREEN}📚 Starting Jupyter Lab...${NC}"
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
             --NotebookApp.token='' --NotebookApp.password=''
        ;;
        
    "mlflow")
        echo -e "${GREEN}🧪 Starting MLflow server...${NC}"
        exec mlflow server \
             --backend-store-uri ${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db} \
             --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT:-/mlflow/artifacts} \
             --host 0.0.0.0 \
             --port 5000
        ;;
        
    "bash")
        setup_directories
        echo -e "${GREEN}💻 Starting bash shell...${NC}"
        exec /bin/bash
        ;;
        
    *)
        # Default behavior - run the provided command
        setup_directories
        setup_data
        
        echo -e "${GREEN}🚀 Running command: $@${NC}"
        exec "$@"
        ;;
esac