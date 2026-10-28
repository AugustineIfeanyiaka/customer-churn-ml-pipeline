import pandas as pd

# Load the dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Check original data type
print("Before cleaning:")
print(df["TotalCharges"].dtype)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check converted data type
print("\nAfter cleaning:")
print(df["TotalCharges"].dtype)

# Count missing values created by conversion
print("\nMissing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Display affected customer records
print("\nRows with missing TotalCharges:")
print(df[df["TotalCharges"].isnull()][
    ["customerID", "tenure", "TotalCharges"]
])
# Remove records with missing TotalCharges
df_clean = df.dropna(subset=["TotalCharges"]).copy()

# Remove duplicate customer records, if any
df_clean = df_clean.drop_duplicates(
    subset=["customerID"]
)

# Save cleaned dataset
df_clean.to_csv(
    "data/Telco-Customer-Churn-Cleaned.csv",
    index=False
)

print("\nOriginal dataset:", df.shape)
print("Cleaned dataset:", df_clean.shape)
print("Remaining missing values:",
      df_clean.isnull().sum().sum())