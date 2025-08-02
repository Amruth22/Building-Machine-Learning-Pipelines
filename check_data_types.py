"""
Check data types in the feature engineered datasets
"""
import pandas as pd

def check_titanic_data():
    """Check Titanic data types"""
    print("🚢 CHECKING TITANIC DATA TYPES")
    print("=" * 40)
    
    try:
        data = pd.read_csv('data/features/titanic_features.csv')
        print(f"Shape: {data.shape}")
        
        # Check data types
        dtypes = data.dtypes
        numerical_cols = dtypes[dtypes != 'object'].index.tolist()
        categorical_cols = dtypes[dtypes == 'object'].index.tolist()
        
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        if categorical_cols:
            print(f"⚠️ Categorical columns found: {categorical_cols}")
            for col in categorical_cols:
                unique_vals = data[col].unique()
                print(f"  {col}: {unique_vals[:5]}...")  # Show first 5 unique values
        else:
            print("✅ All columns are numerical!")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️ Missing values found:")
            print(missing[missing > 0])
        else:
            print("✅ No missing values!")
            
    except FileNotFoundError:
        print("❌ Titanic features file not found. Run the pipeline first.")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_housing_data():
    """Check Housing data types"""
    print("\n🏠 CHECKING HOUSING DATA TYPES")
    print("=" * 40)
    
    try:
        data = pd.read_csv('data/features/housing_features.csv')
        print(f"Shape: {data.shape}")
        
        # Check data types
        dtypes = data.dtypes
        numerical_cols = dtypes[dtypes != 'object'].index.tolist()
        categorical_cols = dtypes[dtypes == 'object'].index.tolist()
        
        print(f"Numerical columns: {len(numerical_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        if categorical_cols:
            print(f"⚠️ Categorical columns found: {categorical_cols}")
            for col in categorical_cols:
                unique_vals = data[col].unique()
                print(f"  {col}: {unique_vals[:5]}...")  # Show first 5 unique values
        else:
            print("✅ All columns are numerical!")
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️ Missing values found:")
            print(missing[missing > 0])
        else:
            print("✅ No missing values!")
            
    except FileNotFoundError:
        print("❌ Housing features file not found. Run the pipeline first.")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🔍 CHECKING DATA TYPES FOR MODEL TRAINING")
    print("=" * 50)
    
    check_titanic_data()
    check_housing_data()
    
    print("\n" + "=" * 50)
    print("💡 If you see categorical columns, regenerate features:")
    print("   python scripts/run_pipeline.py --dataset titanic --report")
    print("   python scripts/run_pipeline.py --dataset housing --report")

if __name__ == "__main__":
    main()