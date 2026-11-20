"""
Step 1 - Clean and merge the raw hourly data into one daily dataset.

Input : data/raw/lahore_air_quality_hourly.csv  (Open-Meteo Air Quality API, CAMS)
        data/raw/lahore_weather_hourly.csv      (Open-Meteo Historical Weather API, ERA5)
Output: data/processed/lahore_daily.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Open-Meteo CSVs have 3 lines of location metadata before the real header.
AQ_COLS = {
    "pm2_5 (μg/m³)": "pm25",
    "pm10 (μg/m³)": "pm10",
    "nitrogen_dioxide (μg/m³)": "no2",
    "carbon_monoxide (μg/m³)": "co",
    "ozone (μg/m³)": "o3",
    "sulphur_dioxide (μg/m³)": "so2",
}
WX_COLS = {
    "temperature_2m (°C)": "temp",
    "relative_humidity_2m (%)": "rh",
    "wind_speed_10m (km/h)": "wind_speed",
    "wind_direction_10m (°)": "wind_dir",
    "precipitation (mm)": "precip",
    "surface_pressure (hPa)": "pressure",
    "boundary_layer_height (m)": "blh",
}


def load(path, cols):
    df = pd.read_csv(path, skiprows=3, parse_dates=["time"])
    return df.rename(columns=cols).set_index("time")


def main():
    aq = load(RAW / "lahore_air_quality_hourly.csv", AQ_COLS)
    wx = load(RAW / "lahore_weather_hourly.csv", WX_COLS)
    hourly = aq.join(wx, how="inner")

    # The first ~3 days of air-quality data are empty (model spin-up) -> drop them.
    hourly = hourly.loc[hourly["pm25"].first_valid_index():]

    # Wind direction is circular (359° is next to 1°), so average it as u/v vectors.
    rad = np.deg2rad(hourly["wind_dir"])
    hourly["wind_u"] = -hourly["wind_speed"] * np.sin(rad)
    hourly["wind_v"] = -hourly["wind_speed"] * np.cos(rad)

    daily = pd.DataFrame({
        "pm25": hourly["pm25"].resample("D").mean(),
        "pm25_max": hourly["pm25"].resample("D").max(),
        "pm10": hourly["pm10"].resample("D").mean(),
        "no2": hourly["no2"].resample("D").mean(),
        "co": hourly["co"].resample("D").mean(),
        "o3": hourly["o3"].resample("D").mean(),
        "so2": hourly["so2"].resample("D").mean(),
        "temp": hourly["temp"].resample("D").mean(),
        "temp_min": hourly["temp"].resample("D").min(),
        "temp_max": hourly["temp"].resample("D").max(),
        "rh": hourly["rh"].resample("D").mean(),
        "wind_speed": hourly["wind_speed"].resample("D").mean(),
        "wind_u": hourly["wind_u"].resample("D").mean(),
        "wind_v": hourly["wind_v"].resample("D").mean(),
        "precip": hourly["precip"].resample("D").sum(),
        "pressure": hourly["pressure"].resample("D").mean(),
        "blh": hourly["blh"].resample("D").mean(),   # has a gap Jan-Jun 2024 in the source
    })
    # Keep only complete days (24 hourly PM2.5 values).
    counts = hourly["pm25"].resample("D").count()
    daily = daily[counts == 24]
    daily.index.name = "date"

    daily.round(3).to_csv(OUT / "lahore_daily.csv")
    print(f"Hourly rows: {len(hourly):,}  ->  daily rows: {len(daily):,}")
    print(f"Period: {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"Days with missing boundary-layer height: {daily['blh'].isna().sum()}")


if __name__ == "__main__":
    main()
