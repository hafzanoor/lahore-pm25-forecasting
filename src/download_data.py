"""
Step 0 - Download the raw data from Open-Meteo (free, no API key).

Air quality: Open-Meteo Air Quality API (CAMS global atmospheric composition model)
Weather    : Open-Meteo Historical Weather API (ERA5 reanalysis)
Location   : Lahore, Pakistan (31.55 N, 74.34 E)
"""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

LAT, LON = 31.5497, 74.3436
START, END = "2022-08-01", "2026-10-31"
COMMON = f"latitude={LAT}&longitude={LON}&start_date={START}&end_date={END}&timezone=Asia%2FKarachi&format=csv"

URLS = {
    "lahore_air_quality_hourly.csv":
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        "hourly=pm2_5,pm10,nitrogen_dioxide,carbon_monoxide,ozone,sulphur_dioxide&" + COMMON,
    "lahore_weather_hourly.csv":
        "https://archive-api.open-meteo.com/v1/archive?"
        "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
        "precipitation,surface_pressure,boundary_layer_height&" + COMMON,
}

if __name__ == "__main__":
    for name, url in URLS.items():
        print(f"Downloading {name} ...")
        with urlopen(url, timeout=120) as r:
            (RAW / name).write_bytes(r.read())
    print("Done.")
