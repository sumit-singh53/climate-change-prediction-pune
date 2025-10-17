import requests
import pandas as pd
from datetime import datetime

def fetch_live_data(lat=18.52, lon=73.85):
    """
    Fetch live weather + air quality for Pune using Open-Meteo APIs.
    """
    print("🌍 Fetching live weather and air quality data from Open-Meteo...")

    # API URLs
    weather_url = "https://api.open-meteo.com/v1/forecast"
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # --- WEATHER ---
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "shortwave_radiation"],
        "timezone": "auto"
    }
    w = requests.get(weather_url, params=weather_params).json()
    wdata = w.get("current", {})

    # --- AIR QUALITY ---
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "timezone": "auto"
    }
    a = requests.get(air_url, params=air_params).json()
    adata = a.get("current", {})

    # Combine results
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temp_C": wdata.get("temperature_2m"),
        "humidity_pct": wdata.get("relative_humidity_2m"),
        "precip_mm": wdata.get("precipitation"),
        "solar_MJ": wdata.get("shortwave_radiation"),
        "pm25": adata.get("pm2_5"),
        "pm10": adata.get("pm10"),
        "co": adata.get("carbon_monoxide"),
        "no2": adata.get("nitrogen_dioxide"),
        "so2": adata.get("sulphur_dioxide"),
        "o3": adata.get("ozone"),
    }

    df = pd.DataFrame([data])
    df.to_csv("data/api/live_pune_env_data.csv", index=False)
    print("✅ Live API data saved to data/api/live_pune_env_data.csv")
    return df


if __name__ == "__main__":
    df = fetch_live_data()
    print(df)