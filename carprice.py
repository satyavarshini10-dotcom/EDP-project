# ============================================================
# WEEK 2 - CAR PRICE PREDICTION
# Using Pandas and Linear Regression
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("CarPrice.csv")

print("="*60)
print("First 5 Rows")
print("="*60)
print(df.head())

# ============================================================
# Dataset Information
# ============================================================

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# Missing Values
# ============================================================

print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values

for column in df.columns:

    if df[column].dtype == "object":
        df[column].fillna(df[column].mode()[0], inplace=True)

    else:
        df[column].fillna(df[column].median(), inplace=True)

# ============================================================
# Remove Duplicate Rows
# ============================================================

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ============================================================
# Encode Categorical Columns
# ============================================================

encoder = LabelEncoder()

for column in df.columns:

    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

# ============================================================
# Correlation
# ============================================================

print("\nCorrelation Matrix")

print(df.corr())

# ============================================================
# Features and Target
# ============================================================

# Remove Car_ID if present

if "car_ID" in df.columns:
    df.drop("car_ID", axis=1, inplace=True)

# Target Column

y = df["price"]

X = df.drop("price", axis=1)

print("\nFeature Shape:", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples :", X_test.shape)

# ============================================================
# Train Model
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Mean Absolute Error :", mae)

print("Mean Squared Error  :", mse)

print("Root Mean Square Error :", rmse)

print("R2 Score :", r2)

# ============================================================
# Actual vs Predicted
# ============================================================

results = pd.DataFrame({

    "Actual Price": y_test.values,
    "Predicted Price": y_pred

})

print("\nFirst 10 Predictions")

print(results.head(10))

# ============================================================
# Save Predictions
# ============================================================

results.to_csv("car_price_predictions.csv", index=False)

print("\nPrediction file saved successfully.")

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({

    "Feature": X.columns,
    "Coefficient": model.coef_

})

importance = importance.sort_values(

    by="Coefficient",
    ascending=False

)

print("\nFeature Importance")

print(importance)

# ============================================================
# Plot Feature Importance
# ============================================================

plt.figure(figsize=(12,6))

plt.bar(

    importance["Feature"],
    importance["Coefficient"]

)

plt.xticks(rotation=90)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Coefficient")

plt.tight_layout()

plt.show()

# ============================================================
# Actual vs Predicted Graph
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Car Price")

plt.grid(True)

plt.show()

# ============================================================
# Prediction Example
# ============================================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPredicted Price for First Car")

print(prediction[0])

# ============================================================
# Model Score
# ============================================================

print("\nTraining Score :", model.score(X_train, y_train))

print("Testing Score :", model.score(X_test, y_test))

# ============================================================
# End of Project
# ============================================================

print("\nCar Price Prediction Project Completed Successfully")