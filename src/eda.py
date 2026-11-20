"""
Step 2 - Exploratory analysis: how bad is Lahore's air, when, and what drives it?

Output: figures/01-05_*.png and results/eda_summary.json
"""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
style.apply()

WHO_24H = 15  # WHO 2021 guideline for 24-hour mean PM2.5 (µg/m³)
# US EPA AQI breakpoints for 24-hour PM2.5 (2024 revision), µg/m³
AQI_BINS = [0, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for\nsensitive groups",
              "Unhealthy", "Very unhealthy", "Hazardous"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load():
    return pd.read_csv(ROOT / "data/processed/lahore_daily.csv",
                       parse_dates=["date"], index_col="date")


def fig_timeseries(d):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d.index, d["pm25"], color=style.GRID, lw=0.8, label="Daily mean")
    ax.plot(d.index, d["pm25"].rolling(30, center=True).mean(),
            color=style.BLUE, lw=2, label="30-day average")
    ax.axhline(WHO_24H, color=style.TEXT_2, lw=1, ls="--")
    ax.text(d.index[5], WHO_24H + 6, f"WHO 24-h guideline ({WHO_24H} µg/m³)",
            color=style.TEXT_2, fontsize=8.5)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Daily PM2.5 in Lahore, Aug 2022 – Oct 2026")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.savefig(FIG / "01_pm25_timeseries.png")
    plt.close(fig)


def fig_monthly(d):
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [d.loc[d.index.month == m, "pm25"].values for m in range(1, 13)]
    bp = ax.boxplot(data, tick_labels=MONTHS, patch_artist=True, widths=0.55,
                    showfliers=False, medianprops=dict(color=style.TEXT, lw=1.5))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(style.ORANGE if i in (10, 11, 0) else "#b7d3f6")
        box.set_edgecolor(style.TEXT_2)
    ax.axhline(WHO_24H, color=style.TEXT_2, lw=1, ls="--")
    ax.set_ylabel("Daily PM2.5 (µg/m³)")
    ax.set_title("Smog season: PM2.5 by month (Nov–Jan highlighted)")
    fig.savefig(FIG / "02_pm25_by_month.png")
    plt.close(fig)


def fig_aqi_days(d):
    winters = {}
    for y in range(2022, 2026):
        s = d.loc[f"{y}-10-01":f"{y + 1}-01-31", "pm25"]
        if len(s) > 100:
            winters[f"Oct {y} –\nJan {y + 1}"] = pd.cut(
                s, AQI_BINS, labels=AQI_LABELS).value_counts().reindex(AQI_LABELS)
    tab = pd.DataFrame(winters).T
    fig, ax = plt.subplots(figsize=(10, 4.2))
    left = np.zeros(len(tab))
    for i, cat in enumerate(AQI_LABELS):
        vals = tab[cat].values
        ax.barh(tab.index, vals, left=left, color=style.BLUE_RAMP[i],
                edgecolor=style.SURFACE, lw=2, label=cat.replace("\n", " "))
        for y, (l, v) in enumerate(zip(left, vals)):
            if v >= 8:
                ax.text(l + v / 2, y, int(v), ha="center", va="center", fontsize=8.5,
                        color=style.TEXT if i < 2 else "white")
        left += vals
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.set_xlabel("Number of days")
    ax.grid(axis="y", visible=False)
    ax.set_title("Air-quality category of each day, smog seasons (Oct–Jan)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(FIG / "03_aqi_days_per_winter.png")
    plt.close(fig)
    return tab


def fig_weather(d):
    cols = {"temp": "Temperature (°C)", "wind_speed": "Wind speed (km/h)",
            "rh": "Relative humidity (%)", "blh": "Boundary-layer height (m)"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    corrs = {}
    winter = d.index.month.isin([11, 12, 1])
    for ax, (c, label) in zip(axes, cols.items()):
        sub = d[[c, "pm25"]].dropna()
        w = winter[d.index.isin(sub.index)]
        ax.scatter(sub.loc[~w, c], sub.loc[~w, "pm25"], s=9, alpha=0.45,
                   color=style.BLUE, lw=0, label="Feb–Oct")
        ax.scatter(sub.loc[w, c], sub.loc[w, "pm25"], s=9, alpha=0.6,
                   color=style.ORANGE, lw=0, label="Nov–Jan")
        r = sub[c].corr(sub["pm25"], method="spearman")
        corrs[c] = round(float(r), 3)
        ax.set_xlabel(label)
        ax.text(0.97, 0.95, f"Spearman ρ = {r:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=style.TEXT_2)
    axes[0].set_ylabel("Daily PM2.5 (µg/m³)")
    axes[0].legend(loc="upper left", markerscale=2)
    fig.suptitle("How weather relates to PM2.5", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=style.TEXT)
    fig.tight_layout()
    fig.savefig(FIG / "04_weather_vs_pm25.png")
    plt.close(fig)
    return corrs


def fig_autocorr(d):
    s = d["pm25"].asfreq("D").interpolate()
    lags = range(1, 31)
    ac = [s.autocorr(l) for l in lags]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(lags, ac, color=style.BLUE, width=0.6)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(0, 1)
    ax.set_title("Today's PM2.5 strongly predicts the next days")
    fig.savefig(FIG / "05_autocorrelation.png")
    plt.close(fig)
    return {f"lag_{l}": round(float(a), 3) for l, a in zip([1, 7, 30], [ac[0], ac[6], ac[29]])}


def main():
    d = load()
    fig_timeseries(d)
    fig_monthly(d)
    tab = fig_aqi_days(d)
    corrs = fig_weather(d)
    ac = fig_autocorr(d)

    monthly = d["pm25"].groupby(d.index.month).mean()
    summary = {
        "days": len(d),
        "period": [str(d.index.min().date()), str(d.index.max().date())],
        "pm25_mean": round(float(d["pm25"].mean()), 1),
        "pm25_max_daily": round(float(d["pm25"].max()), 1),
        "pm25_max_daily_date": str(d["pm25"].idxmax().date()),
        "pct_days_above_who": round(float((d["pm25"] > WHO_24H).mean() * 100), 1),
        "monthly_mean": {MONTHS[m - 1]: round(float(v), 1) for m, v in monthly.items()},
        "worst_month_vs_best_month_ratio": round(float(monthly.max() / monthly.min()), 1),
        "winter_aqi_days": {k.replace("\n", " "): {c.replace("\n", " "): int(v) for c, v in r.items()}
                            for k, r in tab.iterrows()},
        "spearman_with_pm25": corrs,
        "autocorrelation": ac,
    }
    (RES / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
