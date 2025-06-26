import pandas as pd

# Load and prepare your dataset
df = pd.read_csv('/Users/dantonucci/Documents/gitLab/pybuildingcluster/src/pybuildingcluster/data/clustering.csv')

# Basic data preparation
print(f"Dataset shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")

# Optional: Clean data if needed
df_clean = df.dropna()  # or handle missing values appropriately

# For local generation (using open-source version)
from mostlyai.sdk import MostlyAI

# initialize SDK
mostly = MostlyAI()
from mostlyai.synthetic import SyntheticDataGenerator