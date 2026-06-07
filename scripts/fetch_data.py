import os
import requests
import pandas as pd

# 1. Karachi Coordinates
LAT = 24.8607
LON = 67.0011

# 2. Complete 1-Year Date Range (June 2025 to June 2026)
START_DATE = "2025-06-01"
END_DATE = "2026-06-01"

print("🚀 Starting 1-Year Historical Data Pipeline for Karachi...")

# ----------------------------------------------------
# PART A: Fetch Air Quality Data
# ----------------------------------------------------
print("📡 Fetching Air Quality Data...")
aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
}

aq_response = requests.get(aq_url, params=aq_params)
if aq_response.status_code != 200:
    print(f"❌ AQI API Failed: {aq_response.text}")
    exit()

aq_data = aq_response.json()["hourly"]
df_aq = pd.DataFrame({
    "timestamp": aq_data["time"],
    "pm25": aq_data["pm2_5"],
    "pm10": aq_data["pm10"],
    "co": aq_data["carbon_monoxide"],
    "no2": aq_data["nitrogen_dioxide"],
    "so2": aq_data["sulphur_dioxide"],
    "ozone": aq_data["ozone"],
    "aqi": aq_data["us_aqi"]
})
print(f"✅ Air Quality Records Fetched: {len(df_aq)}")

# ----------------------------------------------------
# PART B: Fetch Weather Data
# ----------------------------------------------------
print("📡 Fetching Weather Data...")
# Open-Meteo Archive API is used for historical weather data
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "surface_pressure"]
}

weather_response = requests.get(weather_url, params=weather_params)
if weather_response.status_code != 200:
    print(f"❌ Weather API Failed: {weather_response.text}")
    exit()

weather_data = weather_response.json()["hourly"]
df_weather = pd.DataFrame({
    "timestamp": weather_data["time"],
    "temperature": weather_data["temperature_2m"],
    "humidity": weather_data["relative_humidity_2m"],
    "rain": weather_data["rain"],
    "wind_speed": weather_data["wind_speed_10m"],
    "pressure": weather_data["surface_pressure"]
})
print(f"✅ Weather Records Fetched: {len(df_weather)}")

# ----------------------------------------------------
# PART C: Merge Weather + AQI on Timestamp
# ----------------------------------------------------
print("🔗 Merging Weather and Air Quality Data...")
final_df = pd.merge(df_aq, df_weather, on="timestamp", how="inner")

# Reordering columns for a professional look
columns_order = ["timestamp", "temperature", "humidity", "rain", "wind_speed", "pressure", "pm25", "pm10", "co", "no2", "so2", "ozone", "aqi"]
final_df = final_df[columns_order]

print("\n📊 First 5 rows of the Combined 1-Year Dataset:")
print(final_df.head())

# 5. Temporary Local Storage
output_path = "data/raw_air_quality.csv"
final_df.to_csv(output_path, index=False)
print(f"\n💾 Success! 1-Year Dataset saved to '{output_path}' ({len(final_df)} hourly rows)!")