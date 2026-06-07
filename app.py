import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(page_title="Karachi Pulse | AQI Insights", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .reportview-container { background: #0e1117; }
        .metric-card {
            background-color: #1f2937;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover { background-color: #1d4ed8 !important; transform: translateY(-1px); }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/wind.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    st.subheader("Analysis Parameters")
    target_city = st.selectbox("Target Zone", ["Karachi Central", "Clifton/Defence", "Korangi Industrial Area"])
    st.markdown("---")
    show_explainability = st.toggle("Enable SHAP Model Explainability", value=False)
    st.caption("v2.4 Production Stable Release")

st.title("💨 Karachi AQI & Air Quality Intelligence")
st.markdown("Providing predictive dynamic metrics directly via localized monitoring frameworks.")
st.markdown("---")

if st.button("Analyze Live Atmospheric Conditions"):
    with st.spinner("Streaming operational data vectors from active nodes..."):
        try:
            response = requests.get("http://127.0.0.1:8000/predict_forecast").json()
            
            if response.get("status") == "success":
                current_data = response["current"]
                forecast_data = response["forecast"]
                
                df_forecast = pd.DataFrame(forecast_data)
                avg_predicted_aqi = df_forecast["predicted_aqi"].iloc[0]
                
                if avg_predicted_aqi <= 50:
                    status_color, status_text = "#10b981", "Excellent (Minimal Risk)"
                elif avg_predicted_aqi <= 100:
                    status_color, status_text = "#f59e0b", "Moderate (Acceptable Conditions)"
                elif avg_predicted_aqi <= 150:
                    status_color, status_text = "#ef4444", "Unhealthy (Action Advised)"
                else:
                    status_color, status_text = "#7f1d1d", "Hazardous (Emergency Alert)"
                
                st.subheader("📍 Current Atmospheric Index")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                        <div class='metric-card' style='border-left-color: {status_color};'>
                            <p style='color: #9ca3af; margin:0; font-size:0.9rem;'>PREDICTED IMMEDIATE AQI</p>
                            <h1 style='color: white; margin:5px 0;'>{avg_predicted_aqi:.1f}</h1>
                            <span style='color: {status_color}; font-weight:700;'>● {status_text}</span>
                        </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #9ca3af; margin:0; font-size:0.9rem;'>METEOROLOGICAL TEMPERATURE</p>
                            <h1 style='color: white; margin:5px 0;'>{current_data['temperature']} °C</h1>
                            <span style='color: #3b82f6;'>Live Station Reading</span>
                        </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='color: #9ca3af; margin:0; font-size:0.9rem;'>RELATIVE HUMIDITY VECTOR</p>
                            <h1 style='color: white; margin:5px 0;'>{current_data['humidity']}%</h1>
                            <span style='color: #10b981;'>Optimal Matrix Range</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.subheader("📅 High-Resolution 3-Day Trend Timeline")
                
                # Reset line chart indexing strategy cleanly
                chart_df = df_forecast.copy()
                chart_df = chart_df.set_index("datetime")
                st.line_chart(chart_df["predicted_aqi"], color="#3b82f6", use_container_width=True)
                
                st.markdown("### Sequential Timeline Breakdowns")
                cols_grid = st.columns(4)
                for idx, row in df_forecast.head(12).iterrows(): 
                    with cols_grid[idx % 4]:
                        st.markdown(f"""
                            <div style='background:#111827; padding:15px; border-radius:8px; margin-bottom:15px; border: 1px solid #374151;'>
                                <strong style='color:#60a5fa;'>{row['day']}</strong><br/>
                                <small style='color:#9ca3af;'>{row['datetime']}</small>
                                <h3 style='margin:10px 0; color:#f3f4f6;'>AQI: {row['predicted_aqi']:.1f}</h3>
                                <small style='color:#9ca3af;'>Temp: {row['temperature']}°C | Hum: {row['humidity']}%</small>
                            </div>
                        """, unsafe_allow_html=True)
                        
            else:
                st.error(f"Inference execution halted: {response.get('message')}")
        except Exception as e:
            st.error(f"Failed to communicate with production forecasting socket node: {e}")

if show_explainability:
    st.markdown("---")
    st.subheader("🛠️ Model Feature Weights & Global Explainability Model")
    shap_path = "models/shap_summary.png"
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    else:
        st.info("SHAP matrix assets loading...")