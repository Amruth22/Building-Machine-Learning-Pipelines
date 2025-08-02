"""
Alternative MLflow UI startup script to avoid compatibility issues
"""
import os
import sys
import subprocess
from pathlib import Path

def start_mlflow_ui():
    """Start MLflow UI with proper configuration"""
    print("🚀 Starting MLflow UI...")
    
    # Set the tracking URI
    mlruns_path = Path("mlruns").absolute()
    
    if not mlruns_path.exists():
        print("⚠️ MLflow runs directory not found. Creating it...")
        mlruns_path.mkdir(exist_ok=True)
    
    print(f"📁 MLflow tracking directory: {mlruns_path}")
    
    # Set environment variables
    os.environ['MLFLOW_TRACKING_URI'] = f"file:///{mlruns_path}"
    
    try:
        # Method 1: Try direct Python import
        print("🔄 Attempting to start MLflow UI (Method 1)...")
        
        import mlflow.server
        from mlflow.server import get_app
        
        # Start the server
        app = get_app(backend_store_uri=f"file:///{mlruns_path}")
        
        print("✅ MLflow UI starting...")
        print("🌐 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop")
        
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
        try:
            # Method 2: Try subprocess with python -m
            print("🔄 Attempting to start MLflow UI (Method 2)...")
            
            cmd = [
                sys.executable, "-m", "mlflow.server",
                "--backend-store-uri", f"file:///{mlruns_path}",
                "--host", "0.0.0.0",
                "--port", "5000"
            ]
            
            print("✅ MLflow UI starting...")
            print("🌐 Open your browser to: http://localhost:5000")
            print("🛑 Press Ctrl+C to stop")
            
            subprocess.run(cmd)
            
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            # Method 3: Simple file server
            print("🔄 Starting simple experiment viewer (Method 3)...")
            start_simple_viewer()

def start_simple_viewer():
    """Start a simple experiment viewer if MLflow fails"""
    try:
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import webbrowser
        import threading
        
        class MLflowHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory="mlruns", **kwargs)
        
        server = HTTPServer(('localhost', 8000), MLflowHandler)
        
        def run_server():
            server.serve_forever()
        
        # Start server in background
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        print("✅ Simple file server started")
        print("🌐 View MLflow files at: http://localhost:8000")
        print("🛑 Press Ctrl+C to stop")
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass
        
        # Keep running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            server.shutdown()
            
    except Exception as e:
        print(f"❌ All methods failed: {e}")
        print("\n📁 You can manually browse the mlruns/ directory to see your experiments")
        show_experiment_summary()

def show_experiment_summary():
    """Show a summary of experiments if MLflow UI can't start"""
    print("\n📊 EXPERIMENT SUMMARY:")
    print("=" * 50)
    
    mlruns_path = Path("mlruns")
    
    if not mlruns_path.exists():
        print("❌ No MLflow experiments found")
        return
    
    # List experiments
    experiments = list(mlruns_path.glob("*/"))
    
    if not experiments:
        print("❌ No experiments found in mlruns/")
        return
    
    for exp_dir in experiments:
        if exp_dir.name == "0":
            exp_name = "Default"
        else:
            exp_name = f"Experiment {exp_dir.name}"
        
        print(f"\n🧪 {exp_name}:")
        
        # List runs
        runs = list(exp_dir.glob("*/"))
        print(f"   📊 Total runs: {len(runs)}")
        
        for run_dir in runs[:3]:  # Show first 3 runs
            if run_dir.is_dir() and run_dir.name != "meta.yaml":
                print(f"   🔄 Run: {run_dir.name[:8]}...")
                
                # Try to read metrics
                metrics_file = run_dir / "metrics"
                if metrics_file.exists():
                    metric_files = list(metrics_file.glob("*"))
                    for metric_file in metric_files[:3]:  # Show first 3 metrics
                        try:
                            with open(metric_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    last_line = lines[-1].strip()
                                    parts = last_line.split()
                                    if len(parts) >= 2:
                                        metric_name = metric_file.name
                                        metric_value = parts[1]
                                        print(f"      📈 {metric_name}: {metric_value}")
                        except:
                            pass
    
    print(f"\n📁 Full experiment data available in: {mlruns_path.absolute()}")

def main():
    """Main function"""
    print("🔧 MLflow UI Startup Helper")
    print("=" * 50)
    
    start_mlflow_ui()

if __name__ == "__main__":
    main()