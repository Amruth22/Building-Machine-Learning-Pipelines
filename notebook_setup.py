"""
Universal Notebook Setup Script
Run this at the beginning of any Jupyter notebook to ensure proper environment setup
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def setup_notebook_environment():
    """
    Universal setup function for Jupyter notebooks
    Handles path navigation, imports, and data verification
    """
    print("🚀 UNIVERSAL NOTEBOOK SETUP")
    print("=" * 50)
    
    # Step 1: Navigate to project root
    original_dir = os.getcwd()
    print(f"📍 Starting directory: {original_dir}")
    
    if os.getcwd().endswith('notebooks'):
        os.chdir('..')
        print(f"📁 Changed to project root: {os.getcwd()}")
    else:
        print(f"📁 Already in project root: {os.getcwd()}")
    
    # Step 2: Add src to Python path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
        print(f"📦 Added to Python path: {src_path}")
    else:
        print(f"📦 Already in Python path: {src_path}")
    
    # Step 3: Import essential libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        
        # Configure plotting
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                print("⚠️ Using default matplotlib style")
        
        sns.set_palette("husl")
        warnings.filterwarnings('ignore')
        
        # Set pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 100)
        
        print("✅ Plotting and display configured")
        
    except ImportError as e:
        print(f"⚠️ Some plotting libraries not available: {e}")
    
    # Step 4: Try to import custom modules
    try:
        from data.data_loader import DataLoader
        from data.data_validator import DataValidator
        print("✅ Custom modules imported successfully")
        custom_modules_available = True
    except ImportError as e:
        print(f"⚠️ Custom modules not available: {e}")
        print("💡 This is normal if you haven't set up the project structure yet")
        custom_modules_available = False
    
    # Step 5: Check data availability
    data_files = {
        'data/raw/titanic.csv': 'Titanic dataset',
        'data/raw/housing.csv': 'Housing dataset'
    }
    
    available_data = {}
    missing_data = []
    
    for file_path, description in data_files.items():
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                available_data[description] = df.shape
                print(f"✅ {description}: {df.shape}")
            except Exception as e:
                print(f"⚠️ {description}: File exists but can't read - {e}")
                missing_data.append(file_path)
        else:
            print(f"❌ {description}: Not found at {file_path}")
            missing_data.append(file_path)
    
    # Step 6: Provide guidance
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    if custom_modules_available:
        print("✅ Custom modules: Available")
    else:
        print("⚠️ Custom modules: Not available")
    
    if available_data:
        print(f"✅ Data files: {len(available_data)} available")
        for desc, shape in available_data.items():
            print(f"   • {desc}: {shape}")
    else:
        print("❌ Data files: None available")
    
    if missing_data:
        print(f"\n🔧 To fix missing data, run:")
        print("   python download_datasets.py")
    
    print(f"\n🎯 Environment ready for analysis!")
    
    return {
        'custom_modules_available': custom_modules_available,
        'available_data': available_data,
        'missing_data': missing_data,
        'project_root': os.getcwd()
    }

def load_data_safely():
    """
    Safely load datasets with multiple fallback methods
    """
    print("\n📥 LOADING DATA SAFELY")
    print("=" * 30)
    
    datasets = {}
    
    # Try to load Titanic data
    try:
        # Method 1: Use DataLoader if available
        try:
            from data.data_loader import DataLoader
            loader = DataLoader()
            titanic_data = loader.load_titanic()
            if not titanic_data.empty:
                datasets['titanic'] = titanic_data
                print(f"✅ Titanic (DataLoader): {titanic_data.shape}")
            else:
                raise Exception("DataLoader returned empty DataFrame")
        except:
            # Method 2: Direct pandas loading
            titanic_data = pd.read_csv('data/raw/titanic.csv')
            datasets['titanic'] = titanic_data
            print(f"✅ Titanic (Direct): {titanic_data.shape}")
            
    except Exception as e:
        print(f"❌ Titanic loading failed: {e}")
        datasets['titanic'] = pd.DataFrame()
    
    # Try to load Housing data
    try:
        # Method 1: Use DataLoader if available
        try:
            from data.data_loader import DataLoader
            loader = DataLoader()
            housing_data = loader.load_housing()
            if not housing_data.empty:
                datasets['housing'] = housing_data
                print(f"✅ Housing (DataLoader): {housing_data.shape}")
            else:
                raise Exception("DataLoader returned empty DataFrame")
        except:
            # Method 2: Direct pandas loading
            housing_data = pd.read_csv('data/raw/housing.csv')
            datasets['housing'] = housing_data
            print(f"✅ Housing (Direct): {housing_data.shape}")
            
    except Exception as e:
        print(f"❌ Housing loading failed: {e}")
        datasets['housing'] = pd.DataFrame()
    
    return datasets

def create_sample_data_if_missing():
    """
    Create sample data if original data is missing
    """
    print("\n🔧 CREATING SAMPLE DATA")
    print("=" * 30)
    
    # Create data directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Create sample Titanic data if missing
    titanic_path = Path("data/raw/titanic.csv")
    if not titanic_path.exists():
        print("Creating sample Titanic data...")
        np.random.seed(42)
        n_samples = 891
        
        titanic_data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.binomial(1, 0.38, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Name': [f"Passenger, Mr. John {i}" for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29, 14, n_samples).clip(0.5, 80),
            'SibSp': np.random.poisson(0.5, n_samples).clip(0, 8),
            'Parch': np.random.poisson(0.4, n_samples).clip(0, 6),
            'Ticket': [f"TICKET{i}" for i in range(1, n_samples + 1)],
            'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 500),
            'Cabin': [f"C{i}" if np.random.random() > 0.77 else None for i in range(1, n_samples + 1)],
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
        }
        
        df = pd.DataFrame(titanic_data)
        df.to_csv(titanic_path, index=False)
        print(f"✅ Sample Titanic data created: {df.shape}")
    
    # Create sample Housing data if missing
    housing_path = Path("data/raw/housing.csv")
    if not housing_path.exists():
        print("Creating sample Housing data...")
        np.random.seed(42)
        n_samples = 506
        
        housing_data = {
            'CRIM': np.random.exponential(3, n_samples).clip(0, 90),
            'ZN': np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02]),
            'INDUS': np.random.uniform(0, 30, n_samples),
            'CHAS': np.random.binomial(1, 0.07, n_samples),
            'NOX': np.random.uniform(0.3, 0.9, n_samples),
            'RM': np.random.normal(6.3, 0.7, n_samples).clip(3, 9),
            'AGE': np.random.uniform(0, 100, n_samples),
            'DIS': np.random.uniform(1, 12, n_samples),
            'RAD': np.random.choice(range(1, 25), n_samples),
            'TAX': np.random.uniform(150, 800, n_samples),
            'PTRATIO': np.random.uniform(12, 22, n_samples),
            'B': np.random.uniform(300, 400, n_samples),
            'LSTAT': np.random.uniform(1, 40, n_samples),
        }
        
        # Create target with realistic correlations
        base_price = 20
        price_factors = (
            (housing_data['RM'] - 6) * 5 +
            -housing_data['CRIM'] * 0.5 +
            -housing_data['LSTAT'] * 0.3 +
            housing_data['CHAS'] * 3 +
            -housing_data['NOX'] * 10 +
            np.random.normal(0, 3, n_samples)
        )
        
        housing_data['MEDV'] = (base_price + price_factors).clip(5, 50)
        
        df = pd.DataFrame(housing_data)
        df.to_csv(housing_path, index=False)
        print(f"✅ Sample Housing data created: {df.shape}")

# Convenience function for notebooks
def quick_setup():
    """
    One-line setup for notebooks
    """
    setup_info = setup_notebook_environment()
    
    if setup_info['missing_data']:
        print("\n🔧 Creating sample data for missing files...")
        create_sample_data_if_missing()
    
    datasets = load_data_safely()
    
    print(f"\n🎉 Quick setup complete!")
    print(f"📊 Available datasets: {list(datasets.keys())}")
    
    return datasets

if __name__ == "__main__":
    # If run as script, perform full setup
    setup_info = setup_notebook_environment()
    create_sample_data_if_missing()
    datasets = load_data_safely()
    
    print(f"\n🎉 Setup script completed!")
    print(f"📊 Datasets ready: {list(datasets.keys())}")