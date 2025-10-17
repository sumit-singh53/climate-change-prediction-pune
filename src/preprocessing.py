import os

import numpy as np
import pandas as pd

def load_grd_file(file_path):
    """
    Load IMD .grd rainfall data and reshape to (days, lat, lon).
    """
    print(f"Loading IMD .grd file: {file_path}")
    data = np.fromfile(file_path, dtype=np.float32)
    print(f"Raw data size: {data.size}")

    n_lon, n_lat = 135, 129  # IMD grid specs
    n_days = int(data.size / (n_lon * n_lat))
    print(f"Detected days: {n_days}")

    data = data.reshape((n_days, n_lat, n_lon))
    data[data < -900] = np.nan  # Replace -999 with NaN
    return data

def extract_pune_rainfall(data):
    """
    Extract rainfall for Pune (18.52N, 73.85E) from IMD grid.
    """
    lats = np.linspace(6.5, 38.5, 129)
    lons = np.linspace(66.5, 100.0, 135)

    lat_idx = (np.abs(lats - 18.52)).argmin()
    lon_idx = (np.abs(lons - 73.85)).argmin()

    pune_rain = data[:, lat_idx, lon_idx]
    print(f"Pune rainfall extracted for {len(pune_rain)} days.")
    return pune_rain

def create_pune_dataframe(rainfall_array, start_year=2024):
    """
    Create daily rainfall DataFrame for Pune.
    """
    dates = pd.date_range(f"{start_year}-01-01", periods=len(rainfall_array))
    df = pd.DataFrame({"date": dates, "rainfall_mm": rainfall_array})

    output_path = "data/processed/pune_rainfall_2024.csv"  # fixed path
    df.to_csv(output_path, index=False)
    print(f"✅ Pune daily rainfall saved to {output_path}")
    return df

def merge_multi_years_manual(years, folder="data/raw/"):
    """
    Combine manually downloaded IMD .grd rainfall files for Pune into one dataset.
    """
    all_years = []
    for year in years:
        file_path = f"{folder}/Rainfall_ind{year}_rfp25.grd"
        if not os.path.exists(file_path):
            print(f"⚠️ File missing: {file_path}")
            continue

        data = load_grd_file(file_path)
        pune_rain = extract_pune_rainfall(data)
        df = pd.DataFrame({
            "date": pd.date_range(f"{year}-01-01", periods=len(pune_rain)),
            "rainfall_mm": pune_rain
        })
        all_years.append(df)
        print(f"✅ Processed year {year}")

    merged = pd.concat(all_years).reset_index(drop=True)
    merged.to_csv("data/processed/pune_rainfall_2014_2024.csv", index=False)
    print("✅ Combined multi-year rainfall saved to data/processed/pune_rainfall_2014_2024.csv")
    return merged

import os
import numpy as np
import pandas as pd

def merge_imd_years(start_year=1951, end_year=2024, folder="data/raw/"):
    """
    Combine IMD .grd rainfall data (0.25° grid) for Pune from multiple years into one DataFrame.
    """
    all_years = []
    for year in range(start_year, end_year + 1):
        file_path = f"{folder}/Rainfall_ind{year}_rfp25.grd"
        if not os.path.exists(file_path):
            print(f"⚠️ File missing: {file_path}")
            continue

        from src.preprocessing import load_grd_file, extract_pune_rainfall
        data = load_grd_file(file_path)
        pune_rain = extract_pune_rainfall(data)
        df = pd.DataFrame({
            "date": pd.date_range(f"{year}-01-01", periods=len(pune_rain)),
            "rainfall_mm": pune_rain
        })
        all_years.append(df)
        print(f"✅ Processed IMD data for {year}")

    merged = pd.concat(all_years).reset_index(drop=True)
    merged.dropna(subset=["rainfall_mm"], inplace=True)
    merged.to_csv("data/processed/pune_rainfall_1951_2024.csv", index=False)
    print("✅ IMD rainfall merged and saved to data/processed/pune_rainfall_1951_2024.csv")
    return merged

def merge_imd_nasa(imd_path, nasa_path, output_path="data/processed/pune_climate_combined_1951_2024.csv"):
    """
    Merge IMD rainfall data with NASA POWER temperature, humidity, and solar radiation.
    Handles DOY or (YEAR, MO, DY) NASA formats.
    """
    import numpy as np
    import pandas as pd

    print("🔄 Merging IMD and NASA datasets...")

    # Load IMD data
    imd = pd.read_csv(imd_path, parse_dates=["date"])

    # Load NASA data
    nasa = pd.read_csv(nasa_path, skiprows=11)
    nasa.rename(columns=lambda x: x.strip().upper(), inplace=True)
    nasa.replace(-999.0, np.nan, inplace=True)

    # 🧠 Handle NASA date formats robustly
    if {'YEAR', 'MO', 'DY'}.issubset(nasa.columns):
        nasa['date'] = pd.to_datetime(
            nasa[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'})
        )
    elif {'YEAR', 'DOY'}.issubset(nasa.columns):
        # ✅ This is your case
        nasa['date'] = pd.to_datetime(nasa['YEAR'].astype(str), format='%Y') + pd.to_timedelta(nasa['DOY'] - 1, unit='D')
    elif 'YYYYMMDD' in nasa.columns:
        nasa['date'] = pd.to_datetime(nasa['YYYYMMDD'], format='%Y%m%d')
    else:
        raise ValueError(f"⚠️ Unknown NASA date format. Columns found: {nasa.columns.tolist()}")

    # Keep only the relevant columns
    cols = [c for c in ['T2M', 'RH2M', 'ALLSKY_SFC_SW_DWN'] if c in nasa.columns]
    nasa = nasa[['date'] + cols]
    nasa.rename(columns={
        'T2M': 'temp_C',
        'RH2M': 'humidity_pct',
        'ALLSKY_SFC_SW_DWN': 'solar_MJ'
    }, inplace=True)

    # Merge both datasets
    combined = pd.merge(nasa, imd, on='date', how='outer')
    combined.sort_values(by='date', inplace=True)
    combined.fillna(inplace=True)
    combined.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to {output_path}")
    return combined