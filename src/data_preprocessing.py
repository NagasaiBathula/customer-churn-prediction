import pandas as pd

# Load dataset
df = pd.read_csv("data/churn.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATA TYPES:")
print(df.dtypes)

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values again
print("\nMISSING VALUES AFTER CONVERSION:")
print(df.isnull().sum())

# Remove rows with missing values
df.dropna(inplace=True)

print("\nNEW DATASET SHAPE:")
print(df.shape)

print("\nUPDATED DATA TYPES:")
print(df.dtypes)