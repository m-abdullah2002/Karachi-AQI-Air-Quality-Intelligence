import os
import sys
import tempfile

# Windows hardcoded /tmp path error bypass hack
if sys.platform.startswith("win"):
    # Windows ke temporary directory ka path lein (e.g., C:\Users\user\AppData\Local\Temp)
    win_temp = tempfile.gettempdir()
    os.environ["HOPSWORKS_TMP_DIR"] = win_temp
    
    # Agar Hopsworks cloud internally '/tmp' ka path dhoonde, to use Windows ke temp folder par force karein
    if not os.path.exists("/tmp"):
        try:
            # Puraane code ki jagah hum os.environ ke zariye '/tmp' ko override kar dete hain
            os.makedirs("/tmp", exist_ok=True)
        except Exception:
            # Agar drive root par permission na mile, to system automatically handle karega
            pass

import pandas as pd
import hopsworks

# 1. Hopsworks Cloud se connect karein
print("Connecting to Hopsworks Feature Store...")
api_key = "2sPuFtpKZd0DSSbf.aKg8zOCU5wwauNa7C2bpI3oWvb3Rv3I7j30IL3ircOOqh0P76uPyIaEMuLmBS3ar"

project = hopsworks.login(
    project="Karachi_AQI_2026",
    api_key_value=api_key
)

# Feature Store ka access lein
fs = project.get_feature_store()

# 2. Local CSV Data Load karein
csv_path = os.path.join("data", "engineered_features.csv")

if not os.path.exists(csv_path):
    print(f" Error: {csv_path} nahi mili! Pehle apna feature engineering wala script chalao.")
    exit()

print(f"Loading local data from {csv_path}...")
df = pd.read_csv(csv_path)

# Data verification aur saaf-safai
df.columns = [c.lower().replace(" ", "_").replace(".", "_") for c in df.columns]

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

print("Sample data to be uploaded:")
print(df.head(2))

# 3. Feature Group Banayein aur Data Upload Karein
print("\nCreating/Getting Feature Group on Hopsworks...")
aqi_fg = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    primary_key=["timestamp"],
    description="Karachi Hourly Air Quality Index (AQI) with engineered lag features.",
    online_enabled=False
)

print("Uploading data to cloud feature store (this might take a minute)...")
aqi_fg.insert(df)

print(" Success! Data successfully uploaded to Hopsworks Feature Store!")