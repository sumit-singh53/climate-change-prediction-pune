import os

import pandas as pd


def merge_final_dataset(
    climate_path="data/processed/pune_climate_with_co2.csv",
    air_path="data/processed/pune_air_quality_v3.csv",
    save_path="data/processed/pune_final_merged.csv",
):
    """
    Merge:
      1️⃣ Climate + CO₂ data
      2️⃣ Air quality (OpenAQ v3)
    into a single, date-aligned dataset for model training.
    """

    print("🌦️ Loading climate + CO₂ data...")
    climate_df = pd.read_csv(climate_path, parse_dates=["date"])
    print(f"   → {len(climate_df)} rows, {climate_df.columns.tolist()}")

    print("🌫️ Loading air quality data...")
    air_df = pd.read_csv(air_path, parse_dates=["date"])
    print(f"   → {len(air_df)} rows, {air_df['station'].nunique()} stations")

    # Pivot AQ data: each parameter → separate column (averaged by date)
    print("📊 Aggregating air quality data by date and parameter...")
    aq_pivot = (
        air_df.groupby(["date", "parameter"])["value"].mean().unstack().reset_index()
    )

    print(f"   → Air quality features: {aq_pivot.columns.tolist()}")

    # Merge both datasets on date
    print("🔗 Merging datasets...")
    merged = pd.merge(climate_df, aq_pivot, on="date", how="left")

    # Optional: drop extreme missing data
    merged = merged.sort_values("date")
    merged.interpolate(limit_direction="both", inplace=True)

    # Restrict to overlapping period (post-2016)
    merged = merged[merged["date"] >= "2016-01-01"]

    print("🧹 Final cleanup...")
    merged.dropna(how="all", inplace=True)

    # Save final dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)

    print(f"\n✅ Final merged dataset saved → {save_path}")
    print(f"📈 Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
    print(
        f"🕒 Date range: {merged['date'].min().date()} → {merged['date'].max().date()}"
    )

    return merged


if __name__ == "__main__":
    merge_final_dataset()
