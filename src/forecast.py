"""
Step 3 - Forecast tomorrow's daily-mean PM2.5 in Lahore.

Two settings are compared:
  A. "Observations only": uses only what is known by the end of today
     (today's pollution and weather).
  B. "With weather forecast": also uses tomorrow's weather. In practice this would
     come from a weather forecast; here we use observed (reanalysis) weather as a
     stand-in, i.e. we assume a perfect weather forecast. This is an upper bound.

Models: persistence baseline (tomorrow = today), Ridge regression,
Random Forest, Histogram Gradient Boosting.

Validation is chronological: train on Aug 2022 – Jun 2025, test on Jul 2025 – Oct 2026
(which includes the full 2025-26 smog season). No shuffling, so no future leakage.

Output: results/model_metrics.csv, results/test_predictions.csv,
        results/feature_importance.csv, figures/06-08_*.png
"""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import style

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
style.apply()

SPLIT_DATE = "2025-07-01"
SEED = 42
AQI_BINS = [0, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
WEATHER = ["temp", "temp_min", "temp_max", "rh", "wind_speed",
           "wind_u", "wind_v", "precip", "pressure"]
# Boundary-layer height is left out of the models: the source has a 6-month gap (Jan–Jun 2024).


NICE = {"pm25_lag0": "PM2.5 today (daily mean)", "pm25_lag1": "PM2.5 yesterday",
        "pm25_lag2": "PM2.5 2 days ago", "pm25_lag6": "PM2.5 6 days ago",
        "pm25_mean3": "PM2.5 3-day average", "pm25_mean7": "PM2.5 7-day average",
        "pm25_max_today": "PM2.5 today (hourly peak)", "season_sin": "Season (sine)",
        "season_cos": "Season (cosine)", "temp": "Temperature", "temp_min": "Min temperature",
        "temp_max": "Max temperature", "rh": "Humidity", "wind_speed": "Wind speed",
        "wind_u": "Wind east-west", "wind_v": "Wind north-south", "precip": "Rain",
        "pressure": "Pressure", "pm10": "PM10", "no2": "NO2", "co": "CO", "o3": "Ozone", "so2": "SO2"}


def pretty(col):
    if col in NICE:
        return NICE[col]
    base, _, when = col.rpartition("_")
    return f"{NICE.get(base, base)} {when}"


def build_features(d):
    d = d.asfreq("D")  # make gaps explicit so lags line up with calendar days
    X = pd.DataFrame(index=d.index)
    for lag in [0, 1, 2, 6]:
        X[f"pm25_lag{lag}"] = d["pm25"].shift(lag)
    X["pm25_mean3"] = d["pm25"].rolling(3).mean()
    X["pm25_mean7"] = d["pm25"].rolling(7).mean()
    X["pm25_max_today"] = d["pm25_max"]
    for c in ["pm10", "no2", "co", "o3", "so2"]:
        X[f"{c}_today"] = d[c]
    for c in WEATHER:
        X[f"{c}_today"] = d[c]
    doy = (d.index + pd.Timedelta(days=1)).dayofyear
    X["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    X_fc = X.copy()
    for c in WEATHER:
        X_fc[f"{c}_tomorrow"] = d[c].shift(-1)

    y = d["pm25"].shift(-1).rename("pm25_tomorrow")
    keep = X_fc.notna().all(axis=1) & y.notna()
    return X[keep], X_fc[keep], y[keep]


def models():
    return {
        "Ridge regression": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                               max_features=0.5, random_state=SEED, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400,
                                                           max_leaf_nodes=15, min_samples_leaf=20,
                                                           l2_regularization=1.0, random_state=SEED),
    }


def scores(y, p, idx):
    smog = idx.month.isin([11, 12, 1])
    cat_true = pd.cut(y, AQI_BINS, labels=False)
    cat_pred = pd.cut(pd.Series(p, index=y.index), AQI_BINS, labels=False)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "MAE_smog_season": mean_absolute_error(y[smog], p[smog]),
        "AQI_category_accuracy_%": float((cat_true == cat_pred).mean() * 100),
    }


def main():
    d = pd.read_csv(ROOT / "data/processed/lahore_daily.csv", parse_dates=["date"], index_col="date")
    X, X_fc, y = build_features(d)
    tr, te = y.index < SPLIT_DATE, y.index >= SPLIT_DATE
    print(f"Train: {tr.sum()} days ({y.index[tr].min().date()} – {y.index[tr].max().date()})")
    print(f"Test : {te.sum()} days ({y.index[te].min().date()} – {y.index[te].max().date()})")

    rows, preds = [], pd.DataFrame({"actual": y[te]})
    p = X.loc[te, "pm25_lag0"].values
    rows.append({"setting": "Baseline", "model": "Persistence (tomorrow = today)", **scores(y[te], p, y.index[te])})
    preds["Persistence"] = p

    fitted = {}
    for setting, feats in [("A. Observations only", X), ("B. With weather forecast", X_fc)]:
        for name, m in models().items():
            m.fit(feats[tr], y[tr])
            p = m.predict(feats[te])
            rows.append({"setting": setting, "model": name, **scores(y[te], p, y.index[te])})
            preds[f"{name} [{setting[0]}]"] = p
            fitted[(setting, name)] = m

    metrics = pd.DataFrame(rows)
    base = metrics.loc[0, "MAE"]
    metrics["MAE_improvement_vs_baseline_%"] = (1 - metrics["MAE"] / base) * 100
    metrics = metrics.round(2)
    metrics.to_csv(RES / "model_metrics.csv", index=False)
    preds.round(2).to_csv(RES / "test_predictions.csv", index_label="date")
    print(metrics.to_string(index=False))

    # Best model per setting (by MAE)
    best = {s: metrics[metrics.setting == s].sort_values("MAE").iloc[0]["model"]
            for s in ["A. Observations only", "B. With weather forecast"]}

    # ---- Figure 6: model comparison (MAE) ----
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels, vals, cols = [], [], []
    for _, r in metrics.iterrows():
        tag = {"Baseline": "", "A. Observations only": " (A)", "B. With weather forecast": " (B)"}[r.setting]
        labels.append(r.model.replace(" (tomorrow = today)", "") + tag)
        vals.append(r.MAE)
        cols.append({"Baseline": style.TEXT_2, "A. Observations only": style.BLUE,
                     "B. With weather forecast": style.ORANGE}[r.setting])
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=cols, height=0.6)
    for yy, v in zip(ypos, vals):
        ax.text(v + 0.3, yy, f"{v:.1f}", va="center", fontsize=8.5, color=style.TEXT)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error on test period (µg/m³) — lower is better")
    ax.set_title("Next-day PM2.5 forecast error by model")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.TEXT_2, label="Baseline"),
                       Patch(color=style.BLUE, label="A: observations only"),
                       Patch(color=style.ORANGE, label="B: + tomorrow's weather")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_xlim(0, max(vals) * 1.1)
    fig.savefig(FIG / "06_model_comparison.png")
    plt.close(fig)

    # ---- Figure 7: forecast vs actual over the test period ----
    col_a = f"{best['A. Observations only']} [A]"
    col_b = f"{best['B. With weather forecast']} [B]"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(preds.index, preds["actual"], color=style.TEXT, lw=1.4, label="Actual")
    ax.plot(preds.index, preds[col_a], color=style.BLUE, lw=1.4, label=f"{best['A. Observations only']} (A)")
    ax.plot(preds.index, preds[col_b], color=style.ORANGE, lw=1.4, label=f"{best['B. With weather forecast']} (B)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Forecast vs actual, test period Jul 2025 – Oct 2026")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.savefig(FIG / "07_forecast_vs_actual.png")
    plt.close(fig)

    # ---- Figure 8: permutation importance of best setting-B model ----
    m = fitted[("B. With weather forecast", best["B. With weather forecast"])]
    pi = permutation_importance(m, X_fc[te], y[te], n_repeats=20, random_state=SEED,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=X_fc.columns).sort_values(ascending=False)
    imp.round(3).to_csv(RES / "feature_importance.csv", header=["MAE_increase_when_shuffled"])
    top = imp.head(10)[::-1]
    top.index = [pretty(c) for c in top.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top.values, color=style.BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in MAE when the feature is shuffled (µg/m³)")
    ax.set_title(f"What drives the forecast? ({best['B. With weather forecast']}, B)")
    fig.savefig(FIG / "08_feature_importance.png")
    plt.close(fig)

    (RES / "best_models.json").write_text(json.dumps(best, indent=2))
    print("\nBest:", best)
    print("\nTop features:\n", imp.head(10).round(2))


if __name__ == "__main__":
    main()
