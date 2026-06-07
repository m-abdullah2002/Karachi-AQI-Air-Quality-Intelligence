Karachi AQI Prediction System
This project is a real-time Air Quality Index (AQI) forecasting system for Karachi. It automatically fetches weather and air quality data, stores it, trains a machine learning model, and provides forecasts for the next 3 days.

How it works
The system is fully automated using GitHub Actions. It handles the following tasks without manual intervention:

Data Collection: Every hour, the system fetches live weather and air quality data from the Open-Meteo API.

Feature Store: This data is processed and saved into a feature store (Hopsworks) to keep a history of air quality metrics in Karachi.

Model Training: Every 24 hours, the system retrains the prediction model using the updated dataset to ensure it adapts to changing weather patterns.

Forecasting: After training, the model generates a 3-day AQI forecast.

Explainability: The system automatically generates and updates a SHAP summary plot, which helps in understanding which features (like temperature or humidity) are influencing the AQI predictions the most.

Project Structure
.github/workflows/: Contains the automation scripts that run the hourly and daily tasks.

scripts/: Contains the Python code for data fetching, feature engineering, and model training.

models/: Stores the trained model files and the automatically updated SHAP plots.

api.py / app.py: Core application files for managing the system.

Setup & Requirements
The project requires the following libraries:

pandas, numpy, scikit-learn

xgboost, lightgbm

hopsworks

shap, matplotlib

The automation relies on GitHub Actions. A Hopsworks API key is required to connect the system to the cloud feature store. This key is stored securely using GitHub Secrets.

Evaluation
This project demonstrates an automated MLOps pipeline where the entire lifecycle, from data ingestion to model deployment and monitoring, is handled by automated workflows.
