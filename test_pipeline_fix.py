"""
Test script to verify pipeline fixes
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all imports work"""
    try:
        from data.data_loader import DataLoader
        from data.preprocessor import DataPreprocessor
        from features.feature_engineer import FeatureEngineer
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    try:
        from data.data_loader import DataLoader
        loader = DataLoader()
        
        # Try to load data (will create synthetic if not available)
        titanic_data = loader.load_titanic()
        if not titanic_data.empty:
            print(f"✅ Titanic data loaded: {titanic_data.shape}")
        else:
            print("⚠️ Titanic data is empty, but no error")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing"""
    try:
        from data.data_loader import DataLoader
        from data.preprocessor import DataPreprocessor
        
        loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        titanic_data = loader.load_titanic()
        if not titanic_data.empty:
            clean_data = preprocessor.preprocess_titanic(titanic_data)
            print(f"✅ Preprocessing successful: {clean_data.shape}")
        else:
            print("⚠️ No data to preprocess")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering with the fixes"""
    try:
        from data.data_loader import DataLoader
        from data.preprocessor import DataPreprocessor
        from features.feature_engineer import FeatureEngineer
        
        loader = DataLoader()
        preprocessor = DataPreprocessor()
        engineer = FeatureEngineer()
        
        # Load, preprocess, and engineer features
        titanic_data = loader.load_titanic()
        if not titanic_data.empty:
            clean_data = preprocessor.preprocess_titanic(titanic_data)
            featured_data = engineer.engineer_titanic_features(clean_data)
            print(f"✅ Feature engineering successful: {featured_data.shape}")
            print(f"✅ Features created: {list(featured_data.columns)}")
        else:
            print("⚠️ No data for feature engineering")
        
        return True
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Pipeline Components...")
    print("=" * 50)
    
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Feature Engineering", test_feature_engineering)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Pipeline should work now.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)