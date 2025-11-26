import pandas as pd
import numpy as np
import pickle
import os
import sys

# --- Import physics modules ---
from model import parameters as p
from model import roof as roof_model
from model import aerodynamics as aero_model
from model import facade as facade_model
from blocks.roof_block import RoofBlock
from blocks.aerodynamics_block import AerodynamicsBlock
from blocks.facade_block import FacadeBlock
from blocks.master import Master

np.random.seed(42)

RESULT_DIR = "csv_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===========================
# Load static resources
# ===========================
DATA_DIR = os.path.join("..", "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
PKL_DIR = os.path.join(DATA_DIR, "pkl_models")

TREND_FILE = os.path.join(CSV_DIR, "meteo_hourly_trend.csv")
TUNING_FILE = os.path.join(CSV_DIR, "meteo_tuning_params.csv")

try:
    df_trend = pd.read_csv(TREND_FILE)
    N_TRENDS = len(df_trend)
    trends = {
        "temp": df_trend["Temp_C_Trend"].values,
        "wind": df_trend["Wind_ms_Trend"].values,
        "rad": df_trend["Radiation_Wm2_Trend"].values,
    }
except FileNotFoundError:
    sys.exit(f"Error: Trend file not found at {TREND_FILE}")

try:
    tuning_df = pd.read_csv(TUNING_FILE, index_col=0)
except FileNotFoundError:
    sys.exit(f"Error: Tuning file not found at {TUNING_FILE}")

PARAMS = {
    "temp": {
        "phi": tuning_df.loc["Temperature_Residual", "phi"],
        "sigma": tuning_df.loc["Temperature_Residual", "sigma"],
        "bias_std": tuning_df.loc["Temperature_Residual", "bias"],
        "max_err": 3.0,
        "file": "temperature_gmm.pkl",
    },
    "wind": {
        "phi": tuning_df.loc["Wind_Speed_Residual", "phi"],
        "sigma": tuning_df.loc["Wind_Speed_Residual", "sigma"],
        "bias_std": tuning_df.loc["Wind_Speed_Residual", "bias"],
        "max_err": 4.0,
        "file": "wind_speed_gmm.pkl",
    },
    "rad": {
        "phi": tuning_df.loc["Radiation_Residual", "phi"],
        "sigma": tuning_df.loc["Radiation_Residual", "sigma"],
        "bias_std": tuning_df.loc["Radiation_Residual", "bias"],
        "max_err": None,
        "file": "radiation_gmm.pkl",
    },
}

models = {}
for key, config in PARAMS.items():
    path = os.path.join(PKL_DIR, config["file"])
    with open(path, "rb") as f:
        models[key] = pickle.load(f)
        models[key].random_state = None

# ===========================
# Weather generator
# ===========================


def generate_stochastic_day():
    """Generate one full weather day based on stochastic AR(1) models."""
    n_samples = N_TRENDS

    bias = {
        "temp": np.clip(np.random.normal(0, PARAMS["temp"]["bias_std"]), -15, 15),
        "wind": np.random.normal(0, PARAMS["wind"]["bias_std"]),
        "rad": np.random.normal(0, PARAMS["rad"]["bias_std"]),
    }

    output = {"temp": [], "wind": [], "rad": []}
    residuals = {"temp": 0.0, "wind": 0.0, "rad": 0.0}

    for i in range(n_samples):
        for key in PARAMS:
            innovation = models[key].sample(1)[0][0][0]
            cfg = PARAMS[key]
            residuals[key] = cfg["phi"] * residuals[key] + cfg["sigma"] * innovation
            if cfg["max_err"]:
                residuals[key] = np.clip(residuals[key], -cfg["max_err"], cfg["max_err"])

        # Temperature
        temp_val = trends["temp"][i] + bias["temp"] + residuals["temp"]
        output["temp"].append(np.clip(temp_val, -20, 55))

        # Wind
        raw_wind = trends["wind"][i] + bias["wind"] + residuals["wind"]
        output["wind"].append(max(0.05, raw_wind))

        # Radiation
        if trends["rad"][i] < 5:
            rad_val = 0.0
        else:
            raw_rad = trends["rad"][i] + bias["rad"] + residuals["rad"]
            rad_val = np.clip(raw_rad, 0, 1400)
        output["rad"].append(rad_val)

    return output

# ===========================
# Physics simulator
# ===========================


def simulate_physics(weather):
    """Run the physical simulation for a full day."""
    T_env = weather["temp"]
    wind = weather["wind"]
    rad = weather["rad"]

    n = len(T_env)

    roof_cement = RoofBlock(T_env[0], p.alpha_cement, 0.0, p.C_cement, roof_model, p)
    roof_green = RoofBlock(T_env[0], p.alpha_green, p.k_evap_green, p.C_green, roof_model, p)
    aero = AerodynamicsBlock(aero_model)
    facade = FacadeBlock(facade_model)
    master = Master(roof_cement, roof_green, aero, facade, p)

    # Spin-up
    for _ in range(p.SPIN_CYCLES):
        for t in range(n):
            master.do_step(T_env[t], rad[t], wind[t])

    # Main simulation
    results = {
        "T_cement": [],
        "T_green": [],
        "H_mix": [],
        "T_air_cement": [],
        "T_air_green": [],
        "T_air_fac_cement": [],
        "T_air_fac_green": [],
        "Rise_cement": [],
        "Rise_green": [],
        "Floors_cement": [],
        "Floors_green": [],
    }

    for t in range(n):
        out = master.do_step(T_env[t], rad[t], wind[t])
        T_c = out["T_c"]
        T_g = out["T_g"]
        H_mix = out["H_mix"]
        T_air_c = out["T_air_c"]
        T_air_g = out["T_air_g"]
        T_air_fac_c = out["T_air_fac_c"]
        T_air_fac_g = out["T_air_fac_g"]
        Rise_c = out["Rise_c"]
        Rise_g = out["Rise_g"]
        Floors_c = out["Floors_c"]
        Floors_g = out["Floors_g"]

        results["T_cement"].append(T_c)
        results["T_green"].append(T_g)
        results["H_mix"].append(H_mix)
        results["T_air_cement"].append(T_air_c)
        results["T_air_green"].append(T_air_g)
        results["T_air_fac_cement"].append(T_air_fac_c)
        results["T_air_fac_green"].append(T_air_fac_g)
        results["Rise_cement"].append(Rise_c)
        results["Rise_green"].append(Rise_g)
        results["Floors_cement"].append(Floors_c)
        results["Floors_green"].append(Floors_g)

    return results

# ===========================
# Monte Carlo simulation
# ===========================


def run_monte_carlo(n_sims=1000):
    """Run Monte Carlo simulations and save all outputs."""
    mc_results = []
    weather_samples = []

    for run in range(n_sims):
        if (run + 1) % 10 == 0 or run == 0:
            print(f"Running simulation {run + 1} / {n_sims}")

        daily_weather = generate_stochastic_day()

        # Save weather sample
        weather_samples.append(pd.DataFrame({
            "run_id": run,
            "hour": range(len(daily_weather["temp"])),
            **daily_weather
        }))

        daily_res = simulate_physics(daily_weather)

        # Save simulation results
        mc_results.append(pd.DataFrame({
            "run_id": run,
            "hour": range(len(daily_res["T_cement"])),
            **daily_res
        }))

    # Save weather samples
    df_weather = pd.concat(weather_samples, ignore_index=True)
    df_weather.to_csv(os.path.join(RESULT_DIR, "weather_samples.csv"), index=False)

    # Save MC results
    df_mc = pd.concat(mc_results, ignore_index=True)
    df_mc.to_csv(os.path.join(RESULT_DIR, "simulation_results.csv"), index=False)

    return df_mc

# ===========================
# Main
# ===========================


def main():
    print("Starting Monte Carlo simulation...")
    results = run_monte_carlo()
    print("Simulation complete.")
    print(results.head())


if __name__ == "__main__":
    main()
