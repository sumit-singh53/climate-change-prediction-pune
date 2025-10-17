#!/usr/bin/env python3
"""
Phase 4 — EDA for Climate Change Prediction (Pune)
--------------------------------------------------
Inputs:
  - A CSV file (e.g., pune_climate_with_co2.csv) containing at least a date column
    and columns for temperature, rainfall, humidity (and optionally CO2).

What it does (aligned to Phase 4 goals):
  1) Load & sanity-check dataset, infer key columns if not provided.
  2) Clean & set daily DateTimeIndex; resample to monthly/annual.
  3) Compute aggregates (monthly, annual) and 12-month rolling trends.
  4) Fit simple linear trends on annual series (slope/decade, Δ first→last).
  5) Compute monthly correlation matrix (incl. CO2 if present).
  6) Save tables (CSV) and charts (PNG), plus a concise EDA_summary.txt.

Usage:
  python phase4_eda.py --csv pune_climate_with_co2.csv --outdir outputs/phase4 \
      --date date --temp temperature --rain rainfall --hum humidity --co2 co2

If column names are unknown, you can omit them; the script will try to infer.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Utilities ----------

def find_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first column whose lowercase name contains any keyword."""
    cols_lower = {c.lower(): c for c in cols}
    for key in keywords:
        for c_low, orig in cols_lower.items():
            if key in c_low:
                return orig
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True)


def linear_trend_annual(series: pd.Series):
    """Fit y = a*year + b on annual series indexed by DatetimeIndex. Returns (slope/decade, slope/year, delta)."""
    s = series.dropna()
    if len(s) < 3:
        return np.nan, np.nan, np.nan
    years = s.index.year.astype(float)
    y = s.values.astype(float)
    a, b = np.polyfit(years, y, 1)
    slope_per_decade = a * 10.0
    delta = y[-1] - y[0]
    return slope_per_decade, a, delta


# ---------- Core EDA ----------

def run_eda(csv_path: Path,
            outdir: Path,
            date_col: Optional[str] = None,
            temp_col: Optional[str] = None,
            rain_col: Optional[str] = None,
            hum_col: Optional[str] = None,
            co2_col: Optional[str] = None,
            daily_freq: str = "D") -> Dict[str, Optional[str]]:
    """
    Returns a mapping of the columns used for reference/reporting.
    """
    ensure_dir(outdir)

    # 1) Load
    df = pd.read_csv(csv_path)
    orig_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    # 2) Column inference
    if not date_col:
        date_col = find_col(df.columns, ["date", "time", "dt"])
        if date_col is None:
            date_col = df.columns[0]

    if not temp_col:
        temp_col = find_col(df.columns, ["temp", "tmean", "t_avg", "tavg", "temperature"])
    if not rain_col:
        rain_col = find_col(df.columns, ["rain", "precip", "prcp"])
    if not hum_col:
        hum_col = find_col(df.columns, ["humid", "rh"])
    if not co2_col:
        co2_col = find_col(df.columns, ["co2", "carbon"])

    # 3) Parse date & set index
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True, dayfirst=False)
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq(daily_freq)

    # 4) Coerce numerics (light-touch)
    for c in [temp_col, rain_col, hum_col, co2_col]:
        if c is not None and c in df.columns:
            df[c] = coerce_numeric(df[c])

    # Save minimal cleaned daily frame
    save_df(df, outdir / "daily_cleaned.csv")

    # 5) Aggregates
    monthly = df.resample("MS").agg({
        temp_col: "mean" if temp_col else "mean",
        rain_col: "sum" if rain_col else "sum",
        hum_col: "mean" if hum_col else "mean",
        co2_col: "mean" if co2_col else "mean",
    })
    annual = df.resample("Y").agg({
        temp_col: "mean" if temp_col else "mean",
        rain_col: "sum" if rain_col else "sum",
        hum_col: "mean" if hum_col else "mean",
        co2_col: "mean" if co2_col else "mean",
    })

    # Index cosmetics for saved versions
    annual_out = annual.copy()
    annual_out.index = annual_out.index.year

    save_df(monthly, outdir / "monthly_aggregates.csv")
    save_df(annual_out, outdir / "annual_aggregates.csv")

    # 6) Rolling 12-month means
    keep_cols = [c for c in [temp_col, rain_col, hum_col] if c is not None]
    roll = monthly[keep_cols].rolling(12, min_periods=6).mean()
    save_df(roll, outdir / "monthly_rolling_12mo.csv")

    # 7) Trends (annual)
    trends = {}
    if temp_col in annual:
        slope_dec, slope_year, delta = linear_trend_annual(annual[temp_col])
        trends["temperature"] = {"slope_per_decade": slope_dec, "delta_first_last": delta, "units": "°C (approx)"}
    if rain_col in annual:
        slope_dec, slope_year, delta = linear_trend_annual(annual[rain_col])
        trends["rainfall"] = {"slope_per_decade": slope_dec, "delta_first_last": delta, "units": "mm (annual sum)"}
    if hum_col in annual:
        slope_dec, slope_year, delta = linear_trend_annual(annual[hum_col])
        trends["humidity"] = {"slope_per_decade": slope_dec, "delta_first_last": delta, "units": "% RH (approx)"}
    trends_df = pd.DataFrame(trends).T
    save_df(trends_df, outdir / "trend_summary_annual.csv")

    # 8) Correlation matrix (monthly)
    corr_summary = None
    if co2_col is not None:
        use_cols = [co2_col] + [c for c in [temp_col, rain_col, hum_col] if c is not None]
    else:
        use_cols = [c for c in [temp_col, rain_col, hum_col] if c is not None]

    if len(use_cols) >= 2:
        corr_summary = monthly[use_cols].corr().round(3)
        save_df(corr_summary, outdir / "monthly_correlations.csv")

    # 9) Plots — one figure per chart
    plt.figure()
    if temp_col is not None:
        monthly[temp_col].plot()
        roll[temp_col].plot()
        plt.title("Temperature — Monthly Mean and 12-mo Rolling")
        plt.xlabel("Date"); plt.ylabel("Temperature")
        plt.tight_layout()
        plt.savefig(outdir / "plot_temperature_trend.png", dpi=150)
    plt.close()

    plt.figure()
    if rain_col is not None:
        monthly[rain_col].plot()
        roll[rain_col].plot()
        plt.title("Rainfall — Monthly Total and 12-mo Rolling")
        plt.xlabel("Date"); plt.ylabel("Rainfall")
        plt.tight_layout()
        plt.savefig(outdir / "plot_rainfall_trend.png", dpi=150)
    plt.close()

    plt.figure()
    if hum_col is not None:
        monthly[hum_col].plot()
        roll[hum_col].plot()
        plt.title("Humidity — Monthly Mean and 12-mo Rolling")
        plt.xlabel("Date"); plt.ylabel("Relative Humidity")
        plt.tight_layout()
        plt.savefig(outdir / "plot_humidity_trend.png", dpi=150)
    plt.close()

    if rain_col is not None:
        plt.figure()
        annual[rain_col].plot(kind="bar")
        plt.title("Annual Rainfall (Sum)")
        plt.xlabel("Year"); plt.ylabel("Rainfall (annual sum)")
        plt.tight_layout()
        plt.savefig(outdir / "plot_annual_rainfall_bar.png", dpi=150)
        plt.close()

    if co2_col is not None and temp_col is not None:
        aligned = monthly[[co2_col, temp_col]].dropna()
        if not aligned.empty:
            plt.figure()
            plt.scatter(aligned[co2_col], aligned[temp_col], s=12)
            plt.title("CO2 vs Temperature (Monthly)")
            plt.xlabel("CO2"); plt.ylabel("Temperature")
            plt.tight_layout()
            plt.savefig(outdir / "plot_co2_vs_temp_scatter.png", dpi=150)
            plt.close()

    # 10) Summary text
    info_lines = []
    info_lines.append("Phase 4 — EDA Summary")
    info_lines.append(f"CSV: {csv_path.name}")
    info_lines.append(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    info_lines.append(f"Detected columns: {dict(date_index=df.index.name, temp=temp_col, rain=rain_col, hum=hum_col, co2=co2_col)}")

    def add_trend_line(label, key):
        t = trends.get(key, {})
        if t:
            info_lines.append(f"- {label} trend: slope≈{t['slope_per_decade']:.3f}/decade; Δ(first→last)≈{t['delta_first_last']:.3f} ({t['units']})")

    add_trend_line("Temperature", "temperature")
    add_trend_line("Rainfall", "rainfall")
    add_trend_line("Humidity", "humidity")

    if corr_summary is not None:
        info_lines.append("\nMonthly correlations (top pairs by |r|):")
        pairs = corr_summary.unstack().dropna()
        # remove self-pairs
        pairs = pairs[[a != b for a, b in pairs.index]]
        pairs = pairs.reindex(pairs.abs().sort_values(ascending=False).index).drop_duplicates()
        for (a, b), val in pairs.head(6).items():
            info_lines.append(f"  • {a} vs {b}: r={val:.3f}")

    (outdir / "EDA_summary.txt").write_text("\n".join(info_lines), encoding="utf-8")

    # Also save a quick README for reproducibility
    (outdir / "README_PHASE4.txt").write_text(
        "This folder contains Phase-4 EDA outputs:\n"
        "- daily_cleaned.csv (daily cleaned frame)\n"
        "- monthly_aggregates.csv, annual_aggregates.csv\n"
        "- monthly_rolling_12mo.csv\n"
        "- trend_summary_annual.csv\n"
        "- monthly_correlations.csv (if ≥2 variables)\n"
        "- plot_*.png (PNG charts)\n"
        "- EDA_summary.txt (key takeaways)\n",
        encoding="utf-8"
    )

    return {
        "date_index": df.index.name,
        "temperature_column": temp_col,
        "rainfall_column": rain_col,
        "humidity_column": hum_col,
        "co2_column": co2_col
    }


def main():
    p = argparse.ArgumentParser(description="Phase 4 — EDA for Pune climate + CO2")
    p.add_argument("--csv", required=True, help="Path to CSV (e.g., pune_climate_with_co2.csv)")
    p.add_argument("--outdir", default="outputs/phase4", help="Directory to write EDA outputs")
    p.add_argument("--date", dest="date_col", default=None, help="Date column name (optional)")
    p.add_argument("--temp", dest="temp_col", default=None, help="Temperature column name (optional)")
    p.add_argument("--rain", dest="rain_col", default=None, help="Rainfall/precip column name (optional)")
    p.add_argument("--hum",  dest="hum_col",  default=None, help="Humidity column name (optional)")
    p.add_argument("--co2",  dest="co2_col",  default=None, help="CO2 column name (optional)")
    p.add_argument("--freq", dest="daily_freq", default="D", help="Base frequency for daily index (default D)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    mapping = run_eda(
        csv_path=csv_path,
        outdir=outdir,
        date_col=args.date_col,
        temp_col=args.temp_col,
        rain_col=args.rain_col,
        hum_col=args.hum_col,
        co2_col=args.co2_col,
        daily_freq=args.daily_freq
    )

    print("Phase 4 EDA complete.")
    print("Column mapping used:", mapping)
    print("Outputs saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
