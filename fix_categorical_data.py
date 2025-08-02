"""
Quick fix for categorical data in existing feature files
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fix_titanic_data():
    """Fix categorical data in Titanic features"""
    print("🚢 Fixing Titanic categorical data...")
    
    try:
        data = pd.read_csv('data/features/titanic_features.csv')
        print(f"Original shape: {data.shape}")
        
        # Find categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'Survived' in categorical_cols:
            categorical_cols.remove('Survived')  # Don't encode target
        
        print(f"Categorical columns to fix: {categorical_cols}")
        
        # Fix each categorical column
        for col in categorical_cols:
            print(f"  Fixing {col}...")
            # Use one-hot encoding
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
        
        # Save fixed data
        data.to_csv('data/features/titanic_features.csv', index=False)
        print(f"✅ Fixed Titanic data saved. New shape: {data.shape}")
        
    except Exception as e:
        print(f"❌ Error fixing Titanic data: {e}")

def fix_housing_data():
    """Fix categorical data in Housing features"""
    print("🏠 Fixing Housing categorical data...")
    
    try:
        data = pd.read_csv('data/features/housing_features.csv')
        print(f"Original shape: {data.shape}")
        
        # Find categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'MEDV' in categorical_cols:
            categorical_cols.remove('MEDV')  # Don't encode target
        
        print(f"Categorical columns to fix: {categorical_cols}")
        
        # Fix each categorical column
        for col in categorical_cols:
            print(f"  Fixing {col}...")
            # Use one-hot encoding
            dummies = pd.get_dummies(data[col], prefix=col)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
        
        # Save fixed data
        data.to_csv('data/features/housing_features.csv', index=False)
        print(f"✅ Fixed Housing data saved. New shape: {data.shape}")
        
    except Exception as e:
        print(f"❌ Error fixing Housing data: {e}")

def main():
    """Main function"""
    print("🔧 FIXING CATEGORICAL DATA IN FEATURE FILES")
    print("=" * 50)
    
    fix_titanic_data()
    print()
    fix_housing_data()
    
    print("\n" + "=" * 50)
    print("✅ Data fixing completed!")
    print("Now try: python train_models_simple.py")

if __name__ == "__main__":
    main()