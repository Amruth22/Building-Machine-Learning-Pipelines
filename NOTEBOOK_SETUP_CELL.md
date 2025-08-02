# Universal Notebook Setup Cell

Copy and paste this cell at the beginning of any Jupyter notebook in this project:

## Option 1: Quick Setup (Recommended)

```python
# =============================================================================
# UNIVERSAL NOTEBOOK SETUP - Copy this cell to any notebook
# =============================================================================

import os
import sys
from pathlib import Path

# Navigate to project root if we're in notebooks directory
if os.getcwd().endswith('notebooks'):
    os.chdir('..')
    print(f"📁 Changed to project root: {os.getcwd()}")

# Add src to Python path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configure environment
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

sns.set_palette("husl")

# Import custom modules with error handling
try:
    from data.data_loader import DataLoader
    from data.data_validator import DataValidator
    print("✅ Custom modules imported")
except ImportError as e:
    print(f"⚠️ Custom modules not available: {e}")

# Verify data files
data_files = ['data/raw/titanic.csv', 'data/raw/housing.csv']
for file in data_files:
    if Path(file).exists():
        print(f"✅ {file} found")
    else:
        print(f"❌ {file} missing - run: python download_datasets.py")

print("🎉 Notebook setup complete!")
```

## Option 2: Using Setup Script

```python
# Use the universal setup script
exec(open('notebook_setup.py').read())
datasets = quick_setup()

# Now you have datasets['titanic'] and datasets['housing'] ready to use
titanic_data = datasets.get('titanic', pd.DataFrame())
housing_data = datasets.get('housing', pd.DataFrame())

print(f"Titanic data: {titanic_data.shape}")
print(f"Housing data: {housing_data.shape}")
```

## Option 3: Safe Data Loading

```python
# Safe data loading with multiple fallbacks
def load_data_safely():
    datasets = {}
    
    # Load Titanic
    try:
        if Path('data/raw/titanic.csv').exists():
            datasets['titanic'] = pd.read_csv('data/raw/titanic.csv')
            print(f"✅ Titanic loaded: {datasets['titanic'].shape}")
        else:
            print("❌ Titanic data not found")
            datasets['titanic'] = pd.DataFrame()
    except Exception as e:
        print(f"❌ Titanic loading error: {e}")
        datasets['titanic'] = pd.DataFrame()
    
    # Load Housing
    try:
        if Path('data/raw/housing.csv').exists():
            datasets['housing'] = pd.read_csv('data/raw/housing.csv')
            print(f"✅ Housing loaded: {datasets['housing'].shape}")
        else:
            print("❌ Housing data not found")
            datasets['housing'] = pd.DataFrame()
    except Exception as e:
        print(f"❌ Housing loading error: {e}")
        datasets['housing'] = pd.DataFrame()
    
    return datasets

# Use it
datasets = load_data_safely()
titanic_data = datasets['titanic']
housing_data = datasets['housing']
```

## Features of Universal Setup:

✅ **Works on any PC** - Handles different operating systems  
✅ **Auto path detection** - Finds project root automatically  
✅ **Error handling** - Graceful fallbacks if modules missing  
✅ **Data verification** - Checks if data files exist  
✅ **Multiple options** - Choose the approach that works for you  
✅ **Plotting ready** - Configures matplotlib and seaborn  
✅ **Import safety** - Won't crash if custom modules missing  

## Troubleshooting:

If you get errors:

1. **"Module not found"** - Make sure you're in the project root directory
2. **"Data not found"** - Run `python download_datasets.py` first
3. **"Import errors"** - Use Option 3 (Safe Data Loading)
4. **"Path issues"** - The setup automatically handles path navigation

## Usage in Any Notebook:

1. Copy Option 1 code block
2. Paste as first cell in your notebook
3. Run the cell
4. Continue with your analysis

The setup is designed to work universally across different environments and handle common issues automatically.