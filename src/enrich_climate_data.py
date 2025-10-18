import os

import numpy as np
import pandas as pd
import requests

# --------------------------
# 1️⃣  Load Existing Dataset
# --------------------------
BASE_PATH = os.path.join("data", "processed")
input_file = os.path.join(BASE_PATH, "pune_climate_combined_1951_2024.csv")
df = pd.read_csv(input_file, parse_dates=["date"])
print(f"📘 Loaded base dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------
# 2️⃣  Add Global CO₂ Data (NOAA Mauna Loa)
# ------------------------------------------
try:
    print("🌍 Fetching CO₂ data (Mauna Loa)...")
    co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    # Detect header lines and skip them dynamically
    co2_raw = pd.read_csv(co2_url)
    co2_raw = co2_raw.rename(columns=str.lower)

    # Handle column name variations
    if (
        "year" in co2_raw.columns
        and "month" in co2_raw.columns
        and "average" in co2_raw.columns
    ):
        co2 = co2_raw[["year", "month", "average"]].rename(
            columns={"average": "co2_ppm"}
        )
    elif "decimal_date" in co2_raw.columns:
        co2_raw[["year", "month"]] = (
            co2_raw["decimal_date"].astype(str).str.split(".", expand=True)
        )
        co2 = co2_raw[["year", "month", "co2_ppm"]]
    else:
        raise ValueError(f"Unexpected CO₂ file format: {co2_raw.columns}")

    # Clean and merge
    co2["year"] = co2["year"].astype(int)
    co2["month"] = co2["month"].astype(int)
    co2["date"] = pd.to_datetime(dict(year=co2.year, month=co2.month, day=1))
    df = df.merge(co2[["date", "co2_ppm"]], on="date", how="left")
    df["co2_ppm"].ffill(inplace=True)
    print(f"✅ Added CO₂ data ({len(co2)} records).")

except Exception as e:
    print("⚠️ CO₂ data fetch failed:", e)


# -----------------------------------------
# 3️⃣  Add Air Quality (Open-Meteo API)
# -----------------------------------------
def fetch_openmeteo_air_quality(
    lat=18.52, lon=73.85, start_date="2020-01-01", end_date="2024-12-31"
):
    """
    Fetches daily air quality data for Pune from Open-Meteo Air Quality API.
    Returns DataFrame with ['date', 'pm25', 'pm10', 'no2', 'so2', 'o3', 'co'].
    """
    print(f"🌫️ Fetching air quality data from Open-Meteo ({start_date} → {end_date})...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "timezone": "auto",
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise ValueError(f"API request failed: {r.status_code}")

    data = r.json().get("daily", {})
    if not data:
        raise ValueError("No daily data returned from Open-Meteo API")

    aqi_df = pd.DataFrame(data)
    aqi_df.rename(
        columns={
            "time": "date",
            "pm2_5": "pm25",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3",
        },
        inplace=True,
    )
    aqi_df["date"] = pd.to_datetime(aqi_df["date"])

    print(f"✅ Air quality data fetched: {len(aqi_df)} days")
    return aqi_df


# --- Merge with main dataset
try:
    aqi = fetch_openmeteo_air_quality()
    df = df.merge(aqi, on="date", how="left")
    for col in ["pm25", "pm10", "co", "no2", "so2", "o3"]:
        df[col] = df[col].interpolate(limit_direction="both")
    print("✅ Added air quality (Open-Meteo) data successfully.")

except Exception as e:
    print("⚠️ Air quality data fetch failed:", e)

# -----------------------------
# 4️⃣  Save Enhanced Dataset
# -----------------------------
output_file = os.path.join(BASE_PATH, "pune_climate_final.csv")
df.to_csv(output_file, index=False)
print(f"💾 Enhanced dataset saved to {output_file}")
print("✅ Data enrichment complete!")
