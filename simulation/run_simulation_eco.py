import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, pickle

# ----------------------------
# PATHS
# ----------------------------
PKL_DIR = "data\\pkl_models"
CSV_DIR = "data\\csv"
PLOT_DIR = "data\\plots"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ----------------------------
# LOAD TREND DATA
# ----------------------------

def load_hourly_trends():
    meteo_trend = pd.read_csv(os.path.join(CSV_DIR, "meteo_hourly_trend.csv"))
    air_trend = pd.read_csv(os.path.join(CSV_DIR, "air_quality_hourly_trend.csv"))
    return meteo_trend, air_trend


# ----------------------------
# LOAD GMM MODELS
# ----------------------------

def load_gmm(name):
    filename = os.path.join(PKL_DIR, f"{name.lower()}_gmm.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing GMM model: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


# ----------------------------
# ECOLOGICAL FORMULAE
# ----------------------------

# Default ecological coefficients (tweakable)
CO2_UPTAKE_RATE = 0.35        # g CO₂ removed per m² per hour per (W/m² / 100)
CO_REMOVAL_RATE = 0.12        # µg/m³ removed per hour per m²
PM_DEPOSITION_RATE = 0.10     # µg/m³ removed per hour per m² for PM10/PM2.5
O2_PRODUCTION_FACTOR = 0.72   # g O2 produced per g CO2 fixed
GREEN_ROOF_AREA = 600.0       # m²


def compute_ecology(radiation, co2, co, pm10, pm25):
    """
    Compute ecological removal based on simple linear models.
    """

    # Light-driven CO₂ uptake
    co2_removed_g = CO2_UPTAKE_RATE * (radiation / 100.0) * GREEN_ROOF_AREA

    # Convert mg/m³ removed (air volume is arbitrary simplified)
    co_removed = CO_REMOVAL_RATE * GREEN_ROOF_AREA
    pm10_removed = PM_DEPOSITION_RATE * GREEN_ROOF_AREA
    pm25_removed = PM_DEPOSITION_RATE * GREEN_ROOF_AREA

    # O2 production linked to CO2 fixation
    o2_prod = co2_removed_g * O2_PRODUCTION_FACTOR

    return co2_removed_g, co_removed, pm10_removed, pm25_removed, o2_prod


# ----------------------------
# SAMPLING FROM GMM
# ----------------------------

def sample_gmm(gmm_model):
    """Draw one random sample from a fitted GMM."""
    sample, _ = gmm_model.sample(1)
    return float(sample)


# ----------------------------
# MAIN ECO SIMULATION
# ----------------------------

def run_ecology_simulation(hours=24):
    meteo_trend, air_trend = load_hourly_trends()

    # Load all GMM residual models
    gmm_rad = load_gmm("radiation")
    gmm_temp = load_gmm("temperature")
    gmm_wind = load_gmm("windspeed")
    gmm_co2   = load_gmm("co2")
    gmm_pm10  = load_gmm("pm10")
    gmm_pm25  = load_gmm("pm2_5")
    gmm_co    = load_gmm("co")

    results = []

    for h in range(hours):
        hour_idx = h % 24

        # Mean trend
        rad_tr = meteo_trend["Radiation_Wm2_Trend"].iloc[hour_idx]
        temp_tr = meteo_trend["Temp_C_Trend"].iloc[hour_idx]
        wind_tr = meteo_trend["Wind_ms_Trend"].iloc[hour_idx]

        co2_tr   = air_trend["CO2_Trend"].iloc[hour_idx]
        pm10_tr  = air_trend["PM10_Trend"].iloc[hour_idx]
        pm25_tr  = air_trend["PM2_5_Trend"].iloc[hour_idx]
        co_tr    = air_trend["CO_Trend"].iloc[hour_idx]

        # Residual sample
        rad = rad_tr   + sample_gmm(gmm_rad)
        temp = temp_tr + sample_gmm(gmm_temp)
        wind = wind_tr + sample_gmm(gmm_wind)

        co2  = co2_tr  + sample_gmm(gmm_co2)
        pm10 = pm10_tr + sample_gmm(gmm_pm10)
        pm25 = pm25_tr + sample_gmm(gmm_pm25)
        co   = co_tr   + sample_gmm(gmm_co)

        # Ecological effects
        co2_r, co_r, pm10_r, pm25_r, o2 = compute_ecology(rad, co2, co, pm10, pm25)

        results.append({
            "hour": h,
            "Radiation": rad,
            "Temperature": temp,
            "Wind": wind,
            "CO2": co2,
            "CO": co,
            "PM10": pm10,
            "PM2.5": pm25,
            "CO2_Removed_g": co2_r,
            "CO_Removed_ug": co_r,
            "PM10_Removed_ug": pm10_r,
            "PM25_Removed_ug": pm25_r,
            "O2_Produced_g": o2
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(CSV_DIR, "ecology_timeseries.csv"), index=False)

    summary = {
        "CO2_removed_total_g": df["CO2_Removed_g"].sum(),
        "CO_removed_total_ug": df["CO_Removed_ug"].sum(),
        "PM10_removed_total_ug": df["PM10_Removed_ug"].sum(),
        "PM25_removed_total_ug": df["PM25_Removed_ug"].sum(),
        "O2_produced_total_g": df["O2_Produced_g"].sum()
    }

    with open(os.path.join(CSV_DIR, "ecology_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return df, summary


# ----------------------------
# PLOTTING
# ----------------------------

def plot_ecology(df):
    # 1) Concentrations
    plt.figure(figsize=(12, 6))
    plt.plot(df["hour"], df["CO2"], label="CO2 (ppm)")
    plt.plot(df["hour"], df["PM10"], label="PM10 (µg/m³)")
    plt.plot(df["hour"], df["PM2.5"], label="PM2.5 (µg/m³)")
    plt.plot(df["hour"], df["CO"], label="CO (µg/m³)")
    plt.legend()
    plt.title("Simulated Air Pollutants")
    plt.xlabel("Hour")
    plt.ylabel("Concentration")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "pollutants_timeseries.png"))
    plt.close()

    # 2) Hourly removals
    plt.figure(figsize=(12, 6))
    plt.plot(df["hour"], df["CO2_Removed_g"], label="CO₂ removed (g/h)")
    plt.plot(df["hour"], df["PM10_Removed_ug"], label="PM10 removed (µg/h)")
    plt.plot(df["hour"], df["PM25_Removed_ug"], label="PM2.5 removed (µg/h)")
    plt.legend()
    plt.title("Ecological Removal per Hour")
    plt.xlabel("Hour")
    plt.ylabel("Removal")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "ecology_removal_timeseries.png"))
    plt.close()

    # 3) Cumulative
    plt.figure(figsize=(12, 6))
    plt.plot(df["hour"], df["CO2_Removed_g"].cumsum(), label="CO₂ cumulative (g)")
    plt.plot(df["hour"], df["PM10_Removed_ug"].cumsum(), label="PM10 cumulative (µg)")
    plt.plot(df["hour"], df["PM25_Removed_ug"].cumsum(), label="PM2.5 cumulative (µg)")
    plt.legend()
    plt.title("Cumulative Ecological Impact")
    plt.xlabel("Hour")
    plt.ylabel("Cumulative removal")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "ecology_cumulative.png"))
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    df, summary = run_ecology_simulation(hours=24)
    plot_ecology(df)
    print("Ecology simulation complete.")
    print("Summary:", summary)
