import numpy as np
import pandas as pd
import os
import pickle
import json
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Import ecology model
from model.ecology import compute_ecology

# ----------------------------
# PATHS
# ----------------------------
DATA_DIR = os.path.join("..", "data")
PKL_DIR = os.path.join(DATA_DIR, "pkl_models")
CSV_DIR = os.path.join(DATA_DIR, "csv")

RESULT_DIR = "eco_results"
os.makedirs(RESULT_DIR, exist_ok=True)
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)

# ----------------------------
# LOAD STATIC RESOURCES
# ----------------------------

TREND_FILE_METEO = os.path.join(CSV_DIR, "meteo_hourly_trend.csv")
TREND_FILE_AIR = os.path.join(CSV_DIR, "air_quality_hourly_trend.csv")
TUNING_FILE_AIR = os.path.join(CSV_DIR, "air_quality_tuning_params.csv")

try:
    df_trend_meteo = pd.read_csv(TREND_FILE_METEO)
    df_trend_air = pd.read_csv(TREND_FILE_AIR)
    N_TRENDS = len(df_trend_meteo)
    
    trends_meteo = {
        "rad": df_trend_meteo["Radiation_Wm2_Trend"].values,
    }
    trends_air = {
        "co2": df_trend_air["CO2_Trend"].values,
        "co": df_trend_air["CO_Trend"].values,
        "pm10": df_trend_air["PM10_Trend"].values,
        "pm25": df_trend_air["PM2_5_Trend"].values,
    }
except FileNotFoundError as e:
    sys.exit(f"Error: Trend file not found: {e}")

try:
    tuning_air = pd.read_csv(TUNING_FILE_AIR, index_col=0)
except FileNotFoundError:
    sys.exit(f"Error: Tuning file not found at {TUNING_FILE_AIR}")

# Parametri AR(1) per inquinanti
PARAMS_AIR = {
    "co2": {
        "phi": tuning_air.loc["CO2_Residual", "phi"],
        "sigma": tuning_air.loc["CO2_Residual", "sigma"],
        "bias_std": tuning_air.loc["CO2_Residual", "bias"],
        "max_err": 50.0,
        "file": "co2_gmm.pkl",
    },
    "co": {
        "phi": tuning_air.loc["CO_Residual", "phi"],
        "sigma": tuning_air.loc["CO_Residual", "sigma"],
        "bias_std": tuning_air.loc["CO_Residual", "bias"],
        "max_err": 2.0,
        "file": "co_gmm.pkl",
    },
    "pm10": {
        "phi": tuning_air.loc["PM10_Residual", "phi"],
        "sigma": tuning_air.loc["PM10_Residual", "sigma"],
        "bias_std": tuning_air.loc["PM10_Residual", "bias"],
        "max_err": 30.0,
        "file": "pm10_gmm.pkl",
    },
    "pm25": {
        "phi": tuning_air.loc["PM2_5_Residual", "phi"],
        "sigma": tuning_air.loc["PM2_5_Residual", "sigma"],
        "bias_std": tuning_air.loc["PM2_5_Residual", "bias"],
        "max_err": 20.0,
        "file": "pm2_5_gmm.pkl",
    },
}

# Parametro radiazione (per CO2 uptake)
PARAMS_METEO = {
    "rad": {
        "phi": 0.6,  # Radiazione ha correlazione più bassa
        "sigma": 50.0,
        "bias_std": 30.0,
        "max_err": None,
        "file": "radiation_gmm.pkl",
    },
}

# Carica tutti i modelli GMM
models_air = {}
for key, config in PARAMS_AIR.items():
    path = os.path.join(PKL_DIR, config["file"])
    with open(path, "rb") as f:
        models_air[key] = pickle.load(f)
        models_air[key].random_state = None

models_meteo = {}
for key, config in PARAMS_METEO.items():
    path = os.path.join(PKL_DIR, config["file"])
    with open(path, "rb") as f:
        models_meteo[key] = pickle.load(f)
        models_meteo[key].random_state = None

# ----------------------------
# ECOLOGICAL COEFFICIENTS
# ----------------------------

CO2_UPTAKE_RATE = 0.35        # g CO₂ removed per m² per hour per (W/m² / 100)
CO_REMOVAL_RATE = 0.12        # base removal factor (units consistent with previous impl)
PM_DEPOSITION_RATE = 0.10     # base deposition factor
O2_PRODUCTION_FACTOR = 0.72   # g O2 produced per g CO2 fixed
GREEN_ROOF_AREA = 600.0       # m²

# New parameters to scale removal with ambient concentrations
CO_REF = 400.0      # reference CO concentration (µg/m³) for normalization
PM_REF = 25.0       # reference PM concentration (µg/m³) for normalization
CO_REMOVAL_EFF = 1.0    # tunable efficiency multiplier for CO removal
PM_DEPOSITION_EFF = 1.0 # tunable efficiency multiplier for PM removal


# ===========================
# STOCHASTIC DAY GENERATOR
# ===========================

def generate_stochastic_day_eco():
    """Generate one full air quality day based on stochastic AR(1) models."""
    n_samples = N_TRENDS

    # Day-to-day bias
    bias = {
        "rad": np.random.normal(0, PARAMS_METEO["rad"]["bias_std"]),
        "co2": np.clip(np.random.normal(0, PARAMS_AIR["co2"]["bias_std"]), -50, 50),
        "co": np.random.normal(0, PARAMS_AIR["co"]["bias_std"]),
        "pm10": np.random.normal(0, PARAMS_AIR["pm10"]["bias_std"]),
        "pm25": np.random.normal(0, PARAMS_AIR["pm25"]["bias_std"]),
    }

    output = {"rad": [], "co2": [], "co": [], "pm10": [], "pm25": []}
    residuals = {"rad": 0.0, "co2": 0.0, "co": 0.0, "pm10": 0.0, "pm25": 0.0}

    for i in range(n_samples):
        # Radiazione (meteo)
        innovation_rad = models_meteo["rad"].sample(1)[0][0][0]
        cfg_rad = PARAMS_METEO["rad"]
        residuals["rad"] = cfg_rad["phi"] * residuals["rad"] + cfg_rad["sigma"] * innovation_rad
        if cfg_rad["max_err"]:
            residuals["rad"] = np.clip(residuals["rad"], -cfg_rad["max_err"], cfg_rad["max_err"])
        
        raw_rad = trends_meteo["rad"][i] + bias["rad"] + residuals["rad"]
        if trends_meteo["rad"][i] < 5:
            rad_val = 0.0
        else:
            rad_val = np.clip(raw_rad, 0, 1400)
        output["rad"].append(rad_val)

        # Inquinanti (aria)
        for key in PARAMS_AIR:
            innovation = models_air[key].sample(1)[0][0][0]
            cfg = PARAMS_AIR[key]
            residuals[key] = cfg["phi"] * residuals[key] + cfg["sigma"] * innovation
            if cfg["max_err"]:
                residuals[key] = np.clip(residuals[key], -cfg["max_err"], cfg["max_err"])

        # CO2
        co2_val = trends_air["co2"][i] + bias["co2"] + residuals["co2"]
        output["co2"].append(np.clip(co2_val, 300, 800))

        # CO
        co_val = trends_air["co"][i] + bias["co"] + residuals["co"]
        output["co"].append(max(0.1, co_val))

        # PM10
        pm10_val = trends_air["pm10"][i] + bias["pm10"] + residuals["pm10"]
        output["pm10"].append(max(0.1, pm10_val))

        # PM2.5
        pm25_val = trends_air["pm25"][i] + bias["pm25"] + residuals["pm25"]
        output["pm25"].append(max(0.1, pm25_val))

    return output


# ===========================
# ECOLOGY SIMULATOR
# ===========================

def simulate_ecology(weather_day):
    """Run ecological simulation for one full day."""
    rad = weather_day["rad"]
    co2 = weather_day["co2"]
    co = weather_day["co"]
    pm10 = weather_day["pm10"]
    pm25 = weather_day["pm25"]

    n = len(rad)
    results = {
        "Radiation": [],
        "CO2": [],
        "CO": [],
        "PM10": [],
        "PM25": [],
        "CO2_Removed_g": [],
        "CO_Removed_ug": [],
        "PM10_Removed_ug": [],
        "PM25_Removed_ug": [],
        "O2_Produced_g": [],
    }

    for t in range(n):
        co2_r, co_r, pm10_r, pm25_r, o2 = compute_ecology(rad[t], co2[t], co[t], pm10[t], pm25[t])

        results["Radiation"].append(rad[t])
        results["CO2"].append(co2[t])
        results["CO"].append(co[t])
        results["PM10"].append(pm10[t])
        results["PM25"].append(pm25[t])
        results["CO2_Removed_g"].append(co2_r)
        results["CO_Removed_ug"].append(co_r)
        results["PM10_Removed_ug"].append(pm10_r)
        results["PM25_Removed_ug"].append(pm25_r)
        results["O2_Produced_g"].append(o2)

    return results


# ===========================
# MONTE CARLO SIMULATION
# ===========================

def run_monte_carlo_ecology(n_sims=100):
    """Run Monte Carlo simulations and save all outputs."""
    mc_results = []
    weather_samples = []

    for run in range(n_sims):
        if (run + 1) % 10 == 0 or run == 0:
            print(f"Running ecology simulation {run + 1} / {n_sims}")

        daily_weather = generate_stochastic_day_eco()

        # Save weather sample
        weather_samples.append(pd.DataFrame({
            "run_id": run,
            "hour": range(len(daily_weather["rad"])),
            **daily_weather
        }))

        daily_res = simulate_ecology(daily_weather)

        # Save ecology results
        mc_results.append(pd.DataFrame({
            "run_id": run,
            "hour": range(len(daily_res["CO2_Removed_g"])),
            **daily_res
        }))

    # Save weather samples
    df_weather = pd.concat(weather_samples, ignore_index=True)
    df_weather.to_csv(os.path.join(RESULT_DIR, "weather_samples_eco.csv"), index=False)

    # Save MC results
    df_mc = pd.concat(mc_results, ignore_index=True)
    df_mc.to_csv(os.path.join(RESULT_DIR, "simulation_results_eco.csv"), index=False)

    return df_mc


# ----------------------------
# PLOTTING
# ----------------------------

def plot_ecology_mc(df):
    """Plot aggregated results from Monte Carlo simulations."""
    
    # 1) Pollutant concentrations (mean ± std by hour)
    plt.figure(figsize=(14, 6))
    hourly_stats_co2 = df.groupby("hour")["CO2"].agg(["mean", "std"])
    hourly_stats_pm10 = df.groupby("hour")["PM10"].agg(["mean", "std"])
    hourly_stats_pm25 = df.groupby("hour")["PM25"].agg(["mean", "std"])
    
    hours = hourly_stats_co2.index
    plt.plot(hours, hourly_stats_co2["mean"], label="CO2 (ppm)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_stats_co2["mean"] - hourly_stats_co2["std"],
                     hourly_stats_co2["mean"] + hourly_stats_co2["std"],
                     alpha=0.2)
    plt.plot(hours, hourly_stats_pm10["mean"], label="PM10 (µg/m³)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_stats_pm10["mean"] - hourly_stats_pm10["std"],
                     hourly_stats_pm10["mean"] + hourly_stats_pm10["std"],
                     alpha=0.2)
    plt.plot(hours, hourly_stats_pm25["mean"], label="PM2.5 (µg/m³)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_stats_pm25["mean"] - hourly_stats_pm25["std"],
                     hourly_stats_pm25["mean"] + hourly_stats_pm25["std"],
                     alpha=0.2)
    plt.legend()
    plt.title("MC: Air Pollutant Concentrations (mean ± std)")
    plt.xlabel("Hour")
    plt.ylabel("Concentration")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "pollutants_mc.png"), dpi=150)
    plt.close()

    # 2) Hourly removals (mean ± std)
    plt.figure(figsize=(14, 6))
    hourly_co2_removed = df.groupby("hour")["CO2_Removed_g"].agg(["mean", "std"])
    hourly_pm10_removed = df.groupby("hour")["PM10_Removed_ug"].agg(["mean", "std"])
    hourly_pm25_removed = df.groupby("hour")["PM25_Removed_ug"].agg(["mean", "std"])
    
    hours = hourly_co2_removed.index
    plt.plot(hours, hourly_co2_removed["mean"], label="CO₂ removed (g/h)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_co2_removed["mean"] - hourly_co2_removed["std"],
                     hourly_co2_removed["mean"] + hourly_co2_removed["std"],
                     alpha=0.2)
    plt.plot(hours, hourly_pm10_removed["mean"], label="PM10 removed (µg/h)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_pm10_removed["mean"] - hourly_pm10_removed["std"],
                     hourly_pm10_removed["mean"] + hourly_pm10_removed["std"],
                     alpha=0.2)
    plt.plot(hours, hourly_pm25_removed["mean"], label="PM2.5 removed (µg/h)", linewidth=2)
    plt.fill_between(hours, 
                     hourly_pm25_removed["mean"] - hourly_pm25_removed["std"],
                     hourly_pm25_removed["mean"] + hourly_pm25_removed["std"],
                     alpha=0.2)
    plt.legend()
    plt.title("MC: Ecological Removal per Hour (mean ± std)")
    plt.xlabel("Hour")
    plt.ylabel("Removal")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "ecology_removal_mc.png"), dpi=150)
    plt.close()

    # 3) Cumulative removal (per run)
    plt.figure(figsize=(14, 6))
    for run_id in df["run_id"].unique()[:20]:  # Plot first 20 runs
        run_data = df[df["run_id"] == run_id].sort_values("hour")
        cumsum_co2 = run_data["CO2_Removed_g"].cumsum()
        plt.plot(run_data["hour"], cumsum_co2, alpha=0.3, linewidth=0.8)
    
    # Mean cumulative
    mean_cumsum = df.groupby("hour")["CO2_Removed_g"].sum().cumsum()
    plt.plot(mean_cumsum.index, mean_cumsum.values, 'r-', linewidth=3, label="Mean")
    plt.legend()
    plt.title("MC: Cumulative CO₂ Removal (20 runs + mean)")
    plt.xlabel("Hour")
    plt.ylabel("Cumulative CO₂ (g)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "ecology_cumulative_mc.png"), dpi=150)
    plt.close()

    # 4) Daily totals distribution (boxplot style)
    daily_totals = df.groupby("run_id").agg({
        "CO2_Removed_g": "sum",
        "CO_Removed_ug": "sum",
        "PM10_Removed_ug": "sum",
        "PM25_Removed_ug": "sum",
        "O2_Produced_g": "sum"
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    daily_totals["CO2_Removed_g"].hist(bins=20, ax=axes[0], edgecolor='black')
    axes[0].set_title("Daily CO₂ Removal Distribution")
    axes[0].set_xlabel("g CO₂/day")
    axes[0].axvline(daily_totals["CO2_Removed_g"].mean(), color='r', linestyle='--', label='Mean')
    axes[0].legend()

    daily_totals["CO_Removed_ug"].hist(bins=20, ax=axes[1], edgecolor='black')
    axes[1].set_title("Daily CO Removal Distribution")
    axes[1].set_xlabel("µg CO/day")
    axes[1].axvline(daily_totals["CO_Removed_ug"].mean(), color='r', linestyle='--', label='Mean')
    axes[1].legend()

    daily_totals["PM10_Removed_ug"].hist(bins=20, ax=axes[2], edgecolor='black')
    axes[2].set_title("Daily PM10 Removal Distribution")
    axes[2].set_xlabel("µg PM10/day")
    axes[2].axvline(daily_totals["PM10_Removed_ug"].mean(), color='r', linestyle='--', label='Mean')
    axes[2].legend()

    daily_totals["PM25_Removed_ug"].hist(bins=20, ax=axes[3], edgecolor='black')
    axes[3].set_title("Daily PM2.5 Removal Distribution")
    axes[3].set_xlabel("µg PM2.5/day")
    axes[3].axvline(daily_totals["PM25_Removed_ug"].mean(), color='r', linestyle='--', label='Mean')
    axes[3].legend()

    daily_totals["O2_Produced_g"].hist(bins=20, ax=axes[4], edgecolor='black')
    axes[4].set_title("Daily O₂ Production Distribution")
    axes[4].set_xlabel("g O₂/day")
    axes[4].axvline(daily_totals["O2_Produced_g"].mean(), color='r', linestyle='--', label='Mean')
    axes[4].legend()

    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "daily_totals_distribution.png"), dpi=150)
    plt.close()

    # Save summary statistics
    summary = {
        "CO2_removed_mean_g": float(daily_totals["CO2_Removed_g"].mean()),
        "CO2_removed_std_g": float(daily_totals["CO2_Removed_g"].std()),
        "CO2_removed_min_g": float(daily_totals["CO2_Removed_g"].min()),
        "CO2_removed_max_g": float(daily_totals["CO2_Removed_g"].max()),
        "CO_removed_mean_ug": float(daily_totals["CO_Removed_ug"].mean()),
        "CO_removed_std_ug": float(daily_totals["CO_Removed_ug"].std()),
        "PM10_removed_mean_ug": float(daily_totals["PM10_Removed_ug"].mean()),
        "PM10_removed_std_ug": float(daily_totals["PM10_Removed_ug"].std()),
        "PM25_removed_mean_ug": float(daily_totals["PM25_Removed_ug"].mean()),
        "PM25_removed_std_ug": float(daily_totals["PM25_Removed_ug"].std()),
        "O2_produced_mean_g": float(daily_totals["O2_Produced_g"].mean()),
        "O2_produced_std_g": float(daily_totals["O2_Produced_g"].std()),
    }

    with open(os.path.join(RESULT_DIR, "ecology_summary_mc.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return summary


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    print("Starting Monte Carlo ecology simulation...")
    df_results = run_monte_carlo_ecology(n_sims=100)
    print("Simulation complete. Generating plots...")
    summary = plot_ecology_mc(df_results)
    print("\n=== ECOLOGY MC SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {RESULT_DIR}/")
