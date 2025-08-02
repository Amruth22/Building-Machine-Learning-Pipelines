"""
Analyze the results of our ML pipeline
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_titanic_features():
    """Analyze Titanic feature engineering results"""
    print("🚢 TITANIC FEATURE ANALYSIS")
    print("=" * 40)
    
    # Load original and engineered data
    original = pd.read_csv('data/raw/titanic.csv')
    processed = pd.read_csv('data/processed/titanic_clean.csv')
    features = pd.read_csv('data/features/titanic_features.csv')
    
    print(f"Original data: {original.shape}")
    print(f"Processed data: {processed.shape}")
    print(f"Feature engineered: {features.shape}")
    print(f"Features added: {features.shape[1] - original.shape[1]}")
    
    # Show new features
    original_cols = set(original.columns)
    new_features = [col for col in features.columns if col not in original_cols]
    print(f"\n🆕 New features created ({len(new_features)}):")
    for feature in new_features[:10]:  # Show first 10
        print(f"  - {feature}")
    if len(new_features) > 10:
        print(f"  ... and {len(new_features) - 10} more")
    
    # Survival rate analysis
    survival_rate = features['Survived'].mean()
    print(f"\n📊 Overall survival rate: {survival_rate:.1%}")
    
    return features

def analyze_housing_features():
    """Analyze Housing feature engineering results"""
    print("\n🏠 HOUSING FEATURE ANALYSIS")
    print("=" * 40)
    
    # Load original and engineered data
    original = pd.read_csv('data/raw/housing.csv')
    processed = pd.read_csv('data/processed/housing_clean.csv')
    features = pd.read_csv('data/features/housing_features.csv')
    
    print(f"Original data: {original.shape}")
    print(f"Processed data: {processed.shape}")
    print(f"Feature engineered: {features.shape}")
    print(f"Features added: {features.shape[1] - original.shape[1]}")
    
    # Show new features
    original_cols = set(original.columns)
    new_features = [col for col in features.columns if col not in original_cols]
    print(f"\n🆕 New features created ({len(new_features)}):")
    for feature in new_features[:10]:  # Show first 10
        print(f"  - {feature}")
    if len(new_features) > 10:
        print(f"  ... and {len(new_features) - 10} more")
    
    # Price statistics
    avg_price = features['MEDV'].mean()
    print(f"\n📊 Average house price: ${avg_price:.1f}k")
    print(f"Price range: ${features['MEDV'].min():.1f}k - ${features['MEDV'].max():.1f}k")
    
    return features

def create_simple_visualizations():
    """Create simple visualizations"""
    print("\n📈 CREATING VISUALIZATIONS...")
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    try:
        # Titanic survival analysis
        titanic = pd.read_csv('data/features/titanic_features.csv')
        
        plt.figure(figsize=(12, 4))
        
        # Survival by Sex
        plt.subplot(1, 3, 1)
        survival_by_sex = titanic.groupby('Sex')['Survived'].mean()
        survival_by_sex.plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Survival Rate by Gender')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        # Survival by Class
        plt.subplot(1, 3, 2)
        survival_by_class = titanic.groupby('Pclass')['Survived'].mean()
        survival_by_class.plot(kind='bar', color='green')
        plt.title('Survival Rate by Class')
        plt.ylabel('Survival Rate')
        plt.xticks(rotation=0)
        
        # Age distribution
        plt.subplot(1, 3, 3)
        titanic['Age'].hist(bins=20, alpha=0.7, color='orange')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/titanic_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Titanic plots saved to plots/titanic_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Could not create Titanic plots: {e}")
    
    try:
        # Housing price analysis
        housing = pd.read_csv('data/features/housing_features.csv')
        
        plt.figure(figsize=(12, 4))
        
        # Price distribution
        plt.subplot(1, 3, 1)
        housing['MEDV'].hist(bins=20, alpha=0.7, color='skyblue')
        plt.title('House Price Distribution')
        plt.xlabel('Price ($1000s)')
        plt.ylabel('Frequency')
        
        # Price vs Rooms
        plt.subplot(1, 3, 2)
        plt.scatter(housing['RM'], housing['MEDV'], alpha=0.6, color='green')
        plt.title('Price vs Number of Rooms')
        plt.xlabel('Average Rooms')
        plt.ylabel('Price ($1000s)')
        
        # Price vs Crime
        plt.subplot(1, 3, 3)
        plt.scatter(housing['CRIM'], housing['MEDV'], alpha=0.6, color='red')
        plt.title('Price vs Crime Rate')
        plt.xlabel('Crime Rate')
        plt.ylabel('Price ($1000s)')
        
        plt.tight_layout()
        plt.savefig('plots/housing_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Housing plots saved to plots/housing_analysis.png")
        
    except Exception as e:
        print(f"⚠️ Could not create Housing plots: {e}")

def main():
    """Main analysis function"""
    print("📊 ANALYZING PIPELINE RESULTS")
    print("=" * 50)
    
    # Analyze features
    titanic_data = analyze_titanic_features()
    housing_data = analyze_housing_features()
    
    # Create visualizations
    create_simple_visualizations()
    
    print("\n🎉 ANALYSIS COMPLETED!")
    print("=" * 50)
    print("📁 Check the following:")
    print("  - plots/ directory for visualizations")
    print("  - data/processed/ for cleaned data")
    print("  - data/features/ for engineered features")
    print("  - reports/ for pipeline reports")
    print("  - MLflow UI at http://localhost:5000")

if __name__ == "__main__":
    main()