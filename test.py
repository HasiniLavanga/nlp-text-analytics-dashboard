import pandas as pd

# Load Dataset
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())