import pandas as pd

# Load dataset
df = pd.read_csv("D:\projects\customer_churn_pred\data\churn.csv")

# Show first 5 rows
print(df.head())

# Dataset info
print(df.info())

# Missing values
print(df.isnull().sum())