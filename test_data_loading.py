"""
Test script to debug data loading issues
"""
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_file_existence():
    """Check if data files exist"""
    print("🔍 CHECKING FILE EXISTENCE")
    print("=" * 40)
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check for data files
    possible_locations = [
        "data/raw/titanic.csv",
        "data/raw/housing.csv",
        "./data/raw/titanic.csv",
        "./data/raw/housing.csv",
        Path("data") / "raw" / "titanic.csv",
        Path("data") / "raw" / "housing.csv"
    ]
    
    for location in possible_locations:
        path = Path(location)
        exists = path.exists()
        if exists:
            size = path.stat().st_size
            print(f"✅ {location} - EXISTS ({size} bytes)")
        else:
            print(f"❌ {location} - NOT FOUND")
    
    # List contents of data directory
    data_dir = Path("data")
    if data_dir.exists():
        print(f"\n📁 Contents of {data_dir.absolute()}:")
        for item in data_dir.rglob("*"):
            if item.is_file():
                print(f"  📄 {item}")
    else:
        print(f"\n❌ Data directory {data_dir.absolute()} does not exist")

def test_direct_loading():
    """Test loading data directly"""
    print("\n🧪 TESTING DIRECT DATA LOADING")
    print("=" * 40)
    
    # Try to load data directly
    data_files = [
        ("Titanic", "data/raw/titanic.csv"),
        ("Housing", "data/raw/housing.csv")
    ]
    
    for name, file_path in data_files:
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                print(f"✅ {name}: Loaded successfully - Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
            else:
                print(f"❌ {name}: File not found at {file_path}")
        except Exception as e:
            print(f"❌ {name}: Error loading - {e}")

def test_data_loader():
    """Test the DataLoader class"""
    print("\n🔧 TESTING DATALOADER CLASS")
    print("=" * 40)
    
    try:
        from data.data_loader import DataLoader
        
        # Initialize loader
        loader = DataLoader()
        print(f"DataLoader initialized")
        
        # Test Titanic loading
        print("Testing Titanic loading...")
        titanic_data = loader.load_titanic()
        print(f"Titanic result: {titanic_data.shape}")
        
        # Test Housing loading
        print("Testing Housing loading...")
        housing_data = loader.load_housing()
        print(f"Housing result: {housing_data.shape}")
        
        if titanic_data.empty and housing_data.empty:
            print("⚠️ Both datasets are empty - there's a loading issue")
        elif titanic_data.empty or housing_data.empty:
            print("⚠️ One dataset is empty")
        else:
            print("✅ Both datasets loaded successfully!")
            
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_data():
    """Create test data if files don't exist"""
    print("\n🔧 CREATING TEST DATA")
    print("=" * 30)
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test Titanic data
    titanic_path = data_dir / "titanic.csv"
    if not titanic_path.exists():
        print("Creating test Titanic data...")
        titanic_data = {
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 1, 0],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22, 38, 26, 35, 35],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05],
            'Cabin': [None, 'C85', None, 'C123', None],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        }
        
        df = pd.DataFrame(titanic_data)
        df.to_csv(titanic_path, index=False)
        print(f"✅ Test Titanic data created at {titanic_path}")
    
    # Create test Housing data
    housing_path = data_dir / "housing.csv"
    if not housing_path.exists():
        print("Creating test Housing data...")
        housing_data = {
            'CRIM': [0.00632, 0.02731, 0.02729, 0.03237, 0.06905],
            'ZN': [18.0, 0.0, 0.0, 0.0, 0.0],
            'INDUS': [2.31, 7.07, 7.07, 2.18, 2.18],
            'CHAS': [0, 0, 0, 0, 0],
            'NOX': [0.538, 0.469, 0.469, 0.458, 0.458],
            'RM': [6.575, 6.421, 7.185, 6.998, 7.147],
            'AGE': [65.2, 78.9, 61.1, 45.8, 54.2],
            'DIS': [4.0900, 4.9671, 4.9671, 6.0622, 6.0622],
            'RAD': [1, 2, 2, 3, 3],
            'TAX': [296, 242, 242, 222, 222],
            'PTRATIO': [15.3, 17.8, 17.8, 18.7, 18.7],
            'B': [396.90, 396.90, 392.83, 394.63, 396.90],
            'LSTAT': [4.98, 9.14, 4.03, 2.94, 5.33],
            'MEDV': [24.0, 21.6, 34.7, 33.4, 36.2]
        }
        
        df = pd.DataFrame(housing_data)
        df.to_csv(housing_path, index=False)
        print(f"✅ Test Housing data created at {housing_path}")

def main():
    """Main test function"""
    print("🧪 DATA LOADING DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Check file existence
    check_file_existence()
    
    # Test direct loading
    test_direct_loading()
    
    # If no data found, create test data
    if not Path("data/raw/titanic.csv").exists() or not Path("data/raw/housing.csv").exists():
        create_test_data()
        print("\n🔄 Retesting after creating test data...")
        test_direct_loading()
    
    # Test DataLoader
    test_data_loader()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 50)
    
    # Recommendations
    if Path("data/raw/titanic.csv").exists() and Path("data/raw/housing.csv").exists():
        print("✅ Data files exist. If DataLoader still fails:")
        print("   1. Restart Jupyter kernel")
        print("   2. Re-run the notebook cells")
        print("   3. Check that you're in the correct directory")
    else:
        print("❌ Data files missing. Run:")
        print("   python download_datasets.py")

if __name__ == "__main__":
    main()