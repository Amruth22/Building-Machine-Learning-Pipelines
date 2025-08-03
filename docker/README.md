# 🐳 Docker Deployment Guide

This directory contains Docker configurations for deploying the ML Pipeline in different environments.

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `Dockerfile.dev` | Development container with debugging tools |
| `docker-compose.dev.yml` | Development stack with hot reload |
| `docker-entrypoint.sh` | Flexible container startup script |
| `README.md` | This documentation |

## 🚀 Quick Start

### Production Deployment
```bash
# Build and start the complete MLOps stack
docker-compose up --build

# Access services:
# - ML Web App: http://localhost:5000
# - MLflow UI: http://localhost:5001
# - Jupyter: http://localhost:8888
```

### Development Environment
```bash
# Start development environment with hot reload
docker-compose -f docker/docker-compose.dev.yml up --build

# Access services:
# - ML Web App: http://localhost:5000
# - Jupyter Lab: http://localhost:8889
# - MLflow UI: http://localhost:5001
```

## 🛠️ Available Services

### Core Services (Always Running)
- **ml-app**: Main ML web application
- **mlflow**: Experiment tracking server
- **postgres**: Database for MLflow backend

### Optional Services (Use Profiles)
- **jupyter**: Notebook server
- **trainer**: Model training service
- **dvc-runner**: DVC pipeline execution
- **redis**: Caching layer
- **prometheus**: Monitoring
- **grafana**: Visualization dashboard

## 📋 Service Profiles

Use Docker Compose profiles to run specific service combinations:

```bash
# Run with training service
docker-compose --profile training up

# Run with monitoring stack
docker-compose --profile monitoring up

# Run with caching
docker-compose --profile cache up

# Run pipeline execution
docker-compose --profile pipeline up
```

## 🔧 Environment Variables

### ML Application
- `FLASK_ENV`: Environment (development/production)
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `PYTHONPATH`: Python module path

### MLflow
- `MLFLOW_BACKEND_STORE_URI`: Database connection string
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: Artifact storage path

### Training
- `TRAIN_MODELS`: Set to "true" to auto-train models

## 📊 Volume Mounts

### Persistent Data
- `./data`: Dataset storage
- `./models`: Trained model storage
- `./mlruns`: MLflow experiment data
- `postgres_data`: Database persistence
- `mlflow_artifacts`: MLflow artifacts

### Development Mounts
- `./`: Full project directory (dev only)
- `./notebooks`: Jupyter notebooks
- `./configs`: Configuration files

## 🚀 Deployment Commands

### Basic Operations
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f ml-app

# Stop services
docker-compose down

# Clean up everything
docker-compose down -v --remove-orphans
```

### Development Workflow
```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Access container shell
docker exec -it ml-pipeline-dev bash

# Run tests inside container
docker exec -it ml-pipeline-dev pytest

# View application logs
docker-compose -f docker/docker-compose.dev.yml logs -f ml-app-dev
```

### Training Workflow
```bash
# Run model training
docker-compose --profile training up trainer

# Run complete pipeline
docker-compose --profile pipeline up dvc-runner

# Train models with custom parameters
docker-compose run --rm trainer python train_models_simple.py
```

## 🔍 Container Entry Points

The containers support multiple entry points via the entrypoint script:

```bash
# Web application (default)
docker run ml-pipeline web

# Model training
docker run ml-pipeline train

# Complete pipeline
docker run ml-pipeline pipeline

# Jupyter notebook
docker run ml-pipeline jupyter

# MLflow server
docker run ml-pipeline mlflow

# Interactive shell
docker run -it ml-pipeline bash
```

## 🐛 Debugging

### Container Debugging
```bash
# Access running container
docker exec -it ml-pipeline-app bash

# Check container logs
docker logs ml-pipeline-app

# Inspect container
docker inspect ml-pipeline-app

# Check resource usage
docker stats
```

### Service Health Checks
```bash
# Check service health
docker-compose ps

# Test web app health
curl http://localhost:5000/health

# Test MLflow health
curl http://localhost:5001/health
```

## 📈 Monitoring

### Service Monitoring (Optional)
```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Log Monitoring
```bash
# Follow all logs
docker-compose logs -f

# Follow specific service
docker-compose logs -f ml-app

# View recent logs
docker-compose logs --tail=100 ml-app
```

## 🔒 Security Considerations

### Production Security
- Non-root user in containers
- Health checks enabled
- Resource limits configured
- Secrets management via environment variables

### Network Security
- Internal network for service communication
- Only necessary ports exposed
- Database not exposed externally

## 🚀 Scaling

### Horizontal Scaling
```bash
# Scale web application
docker-compose up --scale ml-app=3

# Scale with load balancer (requires nginx)
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up
```

### Resource Limits
Configure in docker-compose.yml:
```yaml
services:
  ml-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## 🔧 Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Use different ports
   docker-compose up -p 5001:5000
   ```

2. **Volume permission issues**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ./data ./models
   ```

3. **Memory issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   
   # Check memory usage
   docker stats
   ```

4. **Build failures**
   ```bash
   # Clean build
   docker-compose build --no-cache
   
   # Remove old images
   docker system prune -a
   ```

### Getting Help
- Check container logs: `docker-compose logs [service]`
- Inspect containers: `docker inspect [container]`
- Test connectivity: `docker exec -it [container] ping [service]`
- Verify mounts: `docker exec -it [container] ls -la /app`

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [MLflow Docker Guide](https://mlflow.org/docs/latest/tracking.html#running-a-tracking-server)
- [Production ML Deployment](https://ml-ops.org/content/deployment)