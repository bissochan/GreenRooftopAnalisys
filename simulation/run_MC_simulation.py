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
from model.ecology import compute_ecology, GREEN_ROOF_AREA
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

METEO_TREND_FILE = os.path.join(CSV_DIR, "meteo_hourly_trend.csv")
METEO_TUNING_FILE = os.path.join(CSV_DIR, "meteo_tuning_params.csv")
AIR_TREND_FILE = os.path.join(CSV_DIR, "air_quality_hourly_trend.csv")
AIR_TUNING_FILE = os.path.join(CSV_DIR, "air_quality_tuning_params.csv")


def load_resource_file():
    print("Loading datasets...")
    try:
        df_trend_meteo = pd.read_csv(METEO_TREND_FILE)
        df_trend_air = pd.read_csv(AIR_TREND_FILE)

        if len(df_trend_meteo) != len(df_trend_air):
            raise ValueError("Meteo and Air trend files must have the same number of entries.")
        n = len(df_trend_meteo)

        trends = {
            "temp": df_trend_meteo["Temp_C_Trend"].values,
            "wind": df_trend_meteo["Wind_ms_Trend"].values,
            "rad": df_trend_meteo["Radiation_Wm2_Trend"].values,
            "co2": df_trend_air["CO2_ppm_Trend"].values,
            "co": df_trend_air["CO_ugm3_Trend"].values,
            "pm10": df_trend_air["PM10_ugm3_Trend"].values,
            "pm25": df_trend_air["PM2_5_ugm3_Trend"].values,
        }

        tuning_meteo = pd.read_csv(METEO_TUNING_FILE, index_col=0)
        tuning_air = pd.read_csv(AIR_TUNING_FILE, index_col=0)

        params_config = {
            "temp": ("Temperature_Residual", "temperature_gmm.pkl"),
            "wind": ("Wind_Speed_Residual", "wind_speed_gmm.pkl"),
            "rad": ("Radiation_Residual", "radiation_gmm.pkl"),
            "co2": ("CO2_Residual", "co2_gmm.pkl"),
            "co": ("CO_Residual", "co_gmm.pkl"),
            "pm10": ("PM10_Residual", "pm10_gmm.pkl"),
            "pm25": ("PM2_5_Residual", "pm2_5_gmm.pkl"),
        }

        params = {}
        models = {}

        for key, (residual_name, model_file) in params_config.items():
            if key in {"temp", "wind", "rad"}:
                tuning_df = tuning_meteo
            else:
                tuning_df = tuning_air

            params[key] = {
                "phi": tuning_df.loc[residual_name, "phi"],
                "sigma": tuning_df.loc[residual_name, "sigma"],
                "bias_std": tuning_df.loc[residual_name, "bias"],
                "max_err": 3.0 if key == "temp" else 4.0 if key == "wind" else 50.0 if key == "co2" else 2.0 if key == "co" else 30.0 if key == "pm10" else 20.0 if key == "pm25" else None,
                "file": model_file,
            }

            with open(os.path.join(PKL_DIR, model_file), "rb") as f:
                models[key] = pickle.load(f)
                models[key].random_state = None

        print("Datasets loaded successfully.")
        return trends, params, models, n
    except FileNotFoundError as e:
        sys.exit(f"Error: Trend or tuning file not found: {e}")

# ===========================
# Weather generator
# ===========================


def generate_stochastic_day(n_steps, trends, params, models):
    """Generate one full weather day based on stochastic AR(1) models."""

    bias = {
        "temp": np.clip(np.random.normal(0, params["temp"]["bias_std"]), -15, 15),
        "wind": np.random.normal(0, params["wind"]["bias_std"]),
        "rad": np.random.normal(0, params["rad"]["bias_std"]),
        "co2": np.clip(np.random.normal(0, params["co2"]["bias_std"]), -50, 50),
        "co": np.random.normal(0, params["co"]["bias_std"]),
        "pm10": np.random.normal(0, params["pm10"]["bias_std"]),
        "pm25": np.random.normal(0, params["pm25"]["bias_std"]),
    }

    output = {k: [] for k in trends.keys()}
    residuals = {k: 0.0 for k in trends.keys()}

    for i in range(n_steps):
        for key in params:
            innovation = models[key].sample(1)[0][0][0]
            cfg = params[key]
            residuals[key] = cfg["phi"] * residuals[key] + cfg["sigma"] * innovation
            if cfg["max_err"]:
                residuals[key] = np.clip(residuals[key], -cfg["max_err"], cfg["max_err"])

        # Temperature
        temp_val = trends["temp"][i] + bias["temp"] + residuals["temp"]
        output["temp"].append(np.clip(temp_val, -20, 55))

        # Wind
        raw_wind = trends["wind"][i] + bias["wind"] + residuals["wind"]
        output["wind"].append(max(0.8, raw_wind))

        # Radiation
        if trends["rad"][i] < 5:
            rad_val = 0.0
        else:
            raw_rad = trends["rad"][i] + bias["rad"] + residuals["rad"]
            rad_val = np.clip(raw_rad, 0, 1400)
        output["rad"].append(rad_val)

        # CO2
        co2_val = trends["co2"][i] + bias["co2"] + residuals["co2"]
        output["co2"].append(np.clip(co2_val, 300, 800))

        # CO
        co_val = trends["co"][i] + bias["co"] + residuals["co"]
        output["co"].append(max(0.1, co_val))

        # PM10
        pm10_val = trends["pm10"][i] + bias["pm10"] + residuals["pm10"]
        output["pm10"].append(max(0.1, pm10_val))

        # PM2.5
        pm25_val = trends["pm25"][i] + bias["pm25"] + residuals["pm25"]
        output["pm25"].append(max(0.1, pm25_val))

    return output

# ===========================
# Physics simulator
# ===========================


def run_physics_simulation(daily_data, n_steps):
    """Run the physical simulation for a full day."""
    T_env = daily_data["temp"]
    wind = daily_data["wind"]
    rad = daily_data["rad"]

    roof_cement = RoofBlock(T_env[0], p.alpha_cement, 0.0, p.C_cement, roof_model, p)
    roof_green = RoofBlock(T_env[0], p.alpha_green, p.k_evap_green, p.C_green, roof_model, p)
    aero = AerodynamicsBlock(aero_model)
    facade = FacadeBlock(facade_model)
    master = Master(roof_cement, roof_green, aero, facade, p)

    # Spin-up
    for _ in range(p.SPIN_CYCLES):
        for t in range(n_steps):
            master.do_step(T_env[t], rad[t], wind[t])

    # Main simulation
    results = []
    for t in range(n_steps):
        out = master.do_step(T_env[t], rad[t], wind[t])
        res = {
            "T_cement": out["T_c"],
            "T_green": out["T_g"],
            "H_mix": out["H_mix"],
            "T_air_cement": out["T_air_c"],
            "T_air_green": out["T_air_g"],
            "T_air_fac_cement": out["T_air_fac_c"],
            "T_air_fac_green": out["T_air_fac_g"],
            "Rise_cement": out["Rise_c"],
            "Rise_green": out["Rise_g"],
            "Floors_cement": out["Floors_c"],
            "Floors_green": out["Floors_g"],
        }
        results.append(res)

    return results



# 1. CORREGGI run_ecology_simulation
def run_ecology_simulation(daily_data, n_steps, wind_speeds, h_mix_values):
    eco_results = []
    
    for t in range(n_steps):
        wind_ms = wind_speeds[t]
        h_eff = min(20.0, h_mix_values[t] * 3)  # Altezza effettiva realistica
        area = GREEN_ROOF_AREA
        
        air_flow_m3_per_h = wind_ms * area * h_eff * 0.01  # 0.1 = efficienza cattura ~10%
        
        co2_rem, co_rem, pm10_rem, pm25_rem, o2_prod = compute_ecology(
            daily_data['rad'][t], daily_data['co2'][t], daily_data['co'][t],
            daily_data['pm10'][t], daily_data['pm25'][t],
            air_flow_m3_per_h=air_flow_m3_per_h
        )
        
        res = {
            "CO2_Removed_g": co2_rem, "CO_Removed_ug": co_rem,
            "PM10_Removed_ug": pm10_rem, "PM25_Removed_ug": pm25_rem,
            "O2_Produced_g": o2_prod, "Air_Flow_m3h": air_flow_m3_per_h
        }
        eco_results.append(res)
    return eco_results



# ===========================
# Monte Carlo simulation
# ===========================


def run_monte_carlo(n_sims=5000):
    """Run Monte Carlo simulations and save all outputs."""
    trends, params, models, n_steps = load_resource_file()

    weather_samples = []
    air_samples = []
    sim_results = []    # physics + eco simulation results

    for run in range(n_sims):
        if (run + 1) % 10 == 0 or run == 0:
            print(f"Running simulation {run + 1} / {n_sims}")

        daily_sample = generate_stochastic_day(n_steps, trends, params, models)

        # Save weather sample
        for t in range(n_steps):
            weather_samples.append({
                "run_id": run,
                "hour": t,
                "temp": daily_sample["temp"][t],
                "wind": daily_sample["wind"][t],
                "rad": daily_sample["rad"][t],
            })

            air_samples.append({
                "run_id": run,
                "hour": t,
                "rad": daily_sample["rad"][t],
                "co2": daily_sample["co2"][t],
                "co": daily_sample["co"][t],
                "pm10": daily_sample["pm10"][t],
                "pm25": daily_sample["pm25"][t],
            })

        phys_res = run_physics_simulation(daily_sample, n_steps)
        eco_res = run_ecology_simulation(daily_sample, n_steps, daily_sample["wind"], [r["H_mix"] for r in phys_res])

        for t in range(n_steps):
            H_mix = phys_res[t]["H_mix"]
            area = GREEN_ROOF_AREA

            co = daily_sample["co"][t]
            pm10 = daily_sample["pm10"][t]
            pm25 = daily_sample["pm25"][t]

            co_mass = co * area * H_mix
            pm10_mass = pm10 * area * H_mix
            pm25_mass = pm25 * area * H_mix

            co_removed = eco_res[t]["CO_Removed_ug"]
            pm10_removed = eco_res[t]["PM10_Removed_ug"]
            pm25_removed = eco_res[t]["PM25_Removed_ug"]

            co_pct = min(100.0, (co_removed / max(co_mass, 1e-6) * 100))
            pm10_pct = min(100.0, (pm10_removed / max(pm10_mass, 1e-6) * 100))
            pm25_pct = min(100.0, (pm25_removed / max(pm25_mass, 1e-6) * 100))

            # Concentrazioni FINALI
            co_final = max(0.0, co - (co_removed / (area * H_mix)))
            pm10_final = max(0.0, pm10 - (pm10_removed / (area * H_mix)))
            pm25_final = max(0.0, pm25 - (pm25_removed / (area * H_mix)))


            sim_results.append({
                "run_id": run,
                "hour": t,
                **phys_res[t],
                **eco_res[t],
                "CO_Initial_ugm3": co,
                "CO_Final_ugm3": co_final,
                "CO2_ppm": daily_sample["co2"][t],
                "PM10_Initial_ugm3": pm10,
                "PM10_Final_ugm3": pm10_final,
                "PM25_Initial_ugm3": pm25,
                "PM25_Final_ugm3": pm25_final,
                "CO_Mass_ug": co_mass,
                "PM10_Mass_ug": pm10_mass,
                "PM25_Mass_ug": pm25_mass,
                "CO_Pct_Removed": co_pct,
                "PM10_Pct_Removed": pm10_pct,
                "PM25_Pct_Removed": pm25_pct,
                "Air_Flow_m3h": eco_res[t]["Air_Flow_m3h"]
            })


    # Save daily samples
    df_weather = pd.DataFrame(weather_samples)
    df_air = pd.DataFrame(air_samples)
    df_sim = pd.DataFrame(sim_results)

    df_weather.to_csv(os.path.join(RESULT_DIR, "weather_samples.csv"), index=False)
    df_air.to_csv(os.path.join(RESULT_DIR, "air_quality_samples.csv"), index=False)
    df_sim.to_csv(os.path.join(RESULT_DIR, "simulation_results.csv"), index=False)

# ===========================
# Main
# ===========================


if __name__ == "__main__":
    print("Starting Monte Carlo simulation...")
    run_monte_carlo()
    print("Simulation complete.")
