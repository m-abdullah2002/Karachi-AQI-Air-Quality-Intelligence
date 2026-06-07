from fastapi import FastAPI
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

model = None
feature_group = None
model_type = None

# Modern Production Lifespan Management (Replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_group, model_type
    try:
        print("Initializing production system architecture via Lifespan...")
        project = hopsworks.login(api_key_value="2sPuFtpKZd0DSSbf.aKg8zOCU5wwauNa7C2bpI3oWvb3Rv3I7j30IL3ircOOqh0P76uPyIaEMuLmBS3ar")
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        feature_group = fs.get_feature_group(name="karachi_aqi_features", version=1)
        model_entries = mr.get_models("karachi_aqi_champion")
        latest_entry = max(model_entries, key=lambda m: m.version)
        
        model_path = latest_entry.download()
        files = os.listdir(model_path)
        pkl_file = next((f for f in files if f.endswith(".pkl")), None)
        
        if pkl_file:
            model = joblib.load(os.path.join(model_path, pkl_file))
            model_type = "sklearn"
            print("Production Backend Ready: Champion ML Engine Secure via Lifespan configuration.")
    except Exception as e:
        print(f"Lifespan Critical System Loading Error: {e}")
        
    yield
    # Shutdown logic (agar kuch clean-up karna ho toh yahan likha jata hai)
    print("Shutting down production engine instance...")

# Pass lifespan context directly during app instantiation
app = FastAPI(title="Karachi AQI Production Forecasting Engine", lifespan=lifespan)

@app.get("/predict_forecast")
def predict_forecast():
    global model, feature_group
    if model is None or feature_group is None:
        return {"status": "error", "message": "Engine cold, dependencies missing."}
    
    try:
        df_all = feature_group.select_all().read()
        
        if df_all.empty:
            return {"status": "error", "message": "Feature Group empty."}
        
        if "timestamp" in df_all.columns:
            df_all = df_all.sort_values(by="timestamp", ascending=True)
            
        latest_row = df_all.iloc[[-1]].copy()
        
        cols_to_drop = ["aqi", "timestamp", "date"]
        input_df = latest_row.drop(columns=[col for col in cols_to_drop if col in latest_row.columns])
        
        forecast_list = []
        base_time = datetime.now()
        current_features = input_df.values.astype(np.float32)
        
        for hour in range(1, 73, 6):
            pred = model.predict(current_features)[0]
            target_date = base_time + timedelta(hours=hour)
            
            simulated_temp = round(float(latest_row.get("temperature", [28.0]).values[0]) + np.sin(hour)*3, 1)
            simulated_hum = min(100, max(10, int(latest_row.get("humidity", [65]).values[0]) + int(np.cos(hour)*8)))
            
            forecast_list.append({
                "datetime": target_date.strftime("%Y-%m-%d %H:%M"),
                "day": target_date.strftime("%A"),
                "predicted_aqi": max(0.0, round(float(pred), 2)),
                "temperature": simulated_temp,
                "humidity": simulated_hum
            })
            
            if "temperature" in input_df.columns:
                t_idx = input_df.columns.get_loc("temperature")
                current_features[0, t_idx] = simulated_temp
            if "humidity" in input_df.columns:
                h_idx = input_df.columns.get_loc("humidity")
                current_features[0, h_idx] = simulated_hum
        
        return {
            "status": "success",
            "current": {
                "temperature": float(latest_row.get("temperature", [30.0]).values[0]),
                "humidity": float(latest_row.get("humidity", [70.0]).values[0]),
                "timestamp": base_time.strftime("%Y-%m-%d %H:%M")
            },
            "forecast": forecast_list
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)