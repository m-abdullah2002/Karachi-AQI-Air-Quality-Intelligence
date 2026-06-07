import os
import pandas as pd

input_path = "data/raw_air_quality.csv"
output_path = "data/engineered_features.csv"

print(" Starting Feature Engineering & Preprocessing Process...")

# 1. Check agar raw data csv maujood hai
if not os.path.exists(input_path):
    print(f" Error: '{input_path}' nahi mili! Pehle fetch_data.py run karo.")
    exit()

# 2. Load the Raw Data
df = pd.read_csv(input_path)
print(f" Raw Data Loaded: {len(df)} rows.")

# ----------------------------------------------------
# PREPROCESSING STEP: Datetime Parsing
# ----------------------------------------------------
print(" Preprocessing: Converting timestamp string to datetime...")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ----------------------------------------------------
# PART 1: Temporal (Time-Based) Features
# ----------------------------------------------------
print(" Extracting Temporal Features...")
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# ----------------------------------------------------
# PART 2: Lag Features (Time-Travel)
# ----------------------------------------------------
print(" Generating Lag Features for AQI & PM2.5...")
df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_2"] = df["aqi"].shift(2)
df["aqi_lag_24"] = df["aqi"].shift(24)
df["pm25_lag_1"] = df["pm25"].shift(1)

# ----------------------------------------------------
# PART 3: Rolling Window Features (Moving Statistics)
# ----------------------------------------------------
print(" Computing Rolling Window Statistics (3h & 24h)...")
df["aqi_roll_mean_3h"] = df["aqi"].rolling(window=3).mean()
df["aqi_roll_mean_24h"] = df["aqi"].rolling(window=24).mean()
df["aqi_roll_std_24h"] = df["aqi"].rolling(window=24).std()

# ----------------------------------------------------
# PART 4: Derived & Interaction Features
# ----------------------------------------------------
print(" Calculating AQI Delta and Weather Interactions...")
# AQI Change Rate (Lag 1 - Lag 2)
df["aqi_change_rate"] = df["aqi_lag_1"] - df["aqi_lag_2"]

# Interaction: Humidity and PM2.5 connection
df["humidity_pm25_interaction"] = df["humidity"] * df["pm25"]

# ----------------------------------------------------
# PREPROCESSING STEP: Cleaning New NaN Values
# ----------------------------------------------------
print(" Preprocessing: Dropping rows with missing values due to lags...")
initial_len = len(df)
df = df.dropna()
final_len = len(df)
print(f" Dropped {initial_len - final_len} rows containing NaN values.")

# 3. Save the Engineered Features
df.to_csv(output_path, index=False)
print(f" Success! Engineered dataset saved to '{output_path}' ({len(df)} rows)!")

print("\n New Engineered Columns Preview:")
print(df[["timestamp", "hour", "aqi", "aqi_lag_1", "aqi_lag_24", "aqi_change_rate"]].head())