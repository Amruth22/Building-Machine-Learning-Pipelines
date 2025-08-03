#!/usr/bin/env python3
"""
Docker Setup and Management Script for ML Pipeline
Provides easy commands to manage Docker deployment
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

def run_command(cmd, check=True, shell=True):
    """Run a shell command"""
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_docker():
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker installation...")
    
    # Check if Docker is installed
    result = run_command("docker --version", check=False)
    if result.returncode != 0:
        print("❌ Docker is not installed!")
        print("Please install Docker from: https://docs.docker.com/get-docker/")
        sys.exit(1)
    
    # Check if Docker is running
    result = run_command("docker info", check=False)
    if result.returncode != 0:
        print("❌ Docker is not running!")
        print("Please start Docker and try again.")
        sys.exit(1)
    
    print("✅ Docker is installed and running")

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("🔧 Checking Docker Compose...")
    
    # Try docker compose (newer version)
    result = run_command("docker compose version", check=False)
    if result.returncode == 0:
        print("✅ Docker Compose (plugin) is available")
        return "docker compose"
    
    # Try docker-compose (older version)
    result = run_command("docker-compose --version", check=False)
    if result.returncode == 0:
        print("✅ Docker Compose (standalone) is available")
        return "docker-compose"
    
    print("❌ Docker Compose is not available!")
    print("Please install Docker Compose")
    sys.exit(1)

def build_images(compose_cmd):
    """Build Docker images"""
    print("🏗️ Building Docker images...")
    run_command(f"{compose_cmd} build")
    print("✅ Images built successfully")

def start_production(compose_cmd):
    """Start production environment"""
    print("🚀 Starting production environment...")
    run_command(f"{compose_cmd} up -d")
    
    print("\n🎉 Production environment started!")
    print("📊 Access your services:")
    print("  • ML Web App: http://localhost:5000")
    print("  • MLflow UI: http://localhost:5001")
    print("  • Jupyter: http://localhost:8888")
    
    # Wait a bit and check health
    print("\n⏳ Waiting for services to start...")
    time.sleep(10)
    check_services_health()

def start_development(compose_cmd):
    """Start development environment"""
    print("🛠️ Starting development environment...")
    run_command(f"{compose_cmd} -f docker/docker-compose.dev.yml up -d")
    
    print("\n🎉 Development environment started!")
    print("📊 Access your services:")
    print("  • ML Web App: http://localhost:5000")
    print("  • Jupyter Lab: http://localhost:8889")
    print("  • MLflow UI: http://localhost:5001")

def start_training(compose_cmd):
    """Start training environment"""
    print("🤖 Starting model training...")
    run_command(f"{compose_cmd} --profile training up trainer")
    print("✅ Model training completed")

def start_pipeline(compose_cmd):
    """Run complete pipeline"""
    print("🔄 Running complete pipeline...")
    run_command(f"{compose_cmd} --profile pipeline up dvc-runner")
    print("✅ Pipeline completed")

def start_monitoring(compose_cmd):
    """Start monitoring stack"""
    print("📈 Starting monitoring stack...")
    run_command(f"{compose_cmd} --profile monitoring up -d")
    
    print("\n📊 Monitoring services started:")
    print("  • Prometheus: http://localhost:9090")
    print("  • Grafana: http://localhost:3000 (admin/admin)")

def stop_services(compose_cmd):
    """Stop all services"""
    print("🛑 Stopping services...")
    run_command(f"{compose_cmd} down")
    
    # Also stop dev environment if running
    if Path("docker/docker-compose.dev.yml").exists():
        run_command(f"{compose_cmd} -f docker/docker-compose.dev.yml down", check=False)
    
    print("✅ All services stopped")

def clean_everything(compose_cmd):
    """Clean up everything"""
    print("🧹 Cleaning up everything...")
    run_command(f"{compose_cmd} down -v --remove-orphans")
    run_command("docker system prune -f")
    print("✅ Cleanup completed")

def show_logs(compose_cmd, service=None):
    """Show service logs"""
    if service:
        print(f"📋 Showing logs for {service}...")
        run_command(f"{compose_cmd} logs -f {service}")
    else:
        print("📋 Showing all logs...")
        run_command(f"{compose_cmd} logs -f")

def show_status(compose_cmd):
    """Show service status"""
    print("📊 Service Status:")
    run_command(f"{compose_cmd} ps")

def check_services_health():
    """Check if services are healthy"""
    print("🏥 Checking service health...")
    
    services = [
        ("ML Web App", "http://localhost:5000/health"),
        ("MLflow", "http://localhost:5001/health"),
    ]
    
    for name, url in services:
        result = run_command(f"curl -f {url}", check=False)
        if result.returncode == 0:
            print(f"✅ {name} is healthy")
        else:
            print(f"⚠️ {name} is not responding")

def interactive_shell(compose_cmd, service="ml-app"):
    """Open interactive shell in container"""
    print(f"💻 Opening shell in {service}...")
    run_command(f"{compose_cmd} exec {service} bash")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Docker Management for ML Pipeline")
    parser.add_argument("command", choices=[
        "check", "build", "start", "dev", "train", "pipeline", 
        "monitoring", "stop", "clean", "logs", "status", "health", "shell"
    ], help="Command to execute")
    parser.add_argument("--service", help="Specific service name")
    
    args = parser.parse_args()
    
    print("🐳 ML Pipeline Docker Manager")
    print("=" * 40)
    
    # Check Docker installation
    check_docker()
    compose_cmd = check_docker_compose()
    
    # Execute command
    if args.command == "check":
        print("✅ Docker environment is ready!")
        
    elif args.command == "build":
        build_images(compose_cmd)
        
    elif args.command == "start":
        build_images(compose_cmd)
        start_production(compose_cmd)
        
    elif args.command == "dev":
        build_images(compose_cmd)
        start_development(compose_cmd)
        
    elif args.command == "train":
        start_training(compose_cmd)
        
    elif args.command == "pipeline":
        start_pipeline(compose_cmd)
        
    elif args.command == "monitoring":
        start_monitoring(compose_cmd)
        
    elif args.command == "stop":
        stop_services(compose_cmd)
        
    elif args.command == "clean":
        clean_everything(compose_cmd)
        
    elif args.command == "logs":
        show_logs(compose_cmd, args.service)
        
    elif args.command == "status":
        show_status(compose_cmd)
        
    elif args.command == "health":
        check_services_health()
        
    elif args.command == "shell":
        interactive_shell(compose_cmd, args.service or "ml-app")

if __name__ == "__main__":
    main()