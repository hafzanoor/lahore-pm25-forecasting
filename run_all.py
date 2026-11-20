"""Run the full pipeline: prepare data -> exploratory analysis -> forecasting models.

If data/raw/ is empty, run `python src/download_data.py` first.
"""
import subprocess
import sys

for step in ["src/prepare_data.py", "src/eda.py", "src/forecast.py"]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, step], check=True)
