import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("data/churn.csv")

# -----------------------------------
# DATA CLEANING
# -----------------------------------

# Remove unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df.dropna(inplace=True)

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

df["AvgCharges"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)

# -----------------------------------
# ONE HOT ENCODING
# -----------------------------------

df = pd.get_dummies(df, drop_first=True)

# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

X = df.drop("Churn_Yes", axis=1)
joblib.dump(
    X.columns.tolist(),
    "models/model_columns.pkl"
)

y = df["Churn_Yes"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# RANDOM FOREST MODEL
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES:")
print(importance.head(10))

# -----------------------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------------------

top_features = importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# -----------------------------------
# SAVE MODEL
# -----------------------------------

joblib.dump(model, "models/churn_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY")