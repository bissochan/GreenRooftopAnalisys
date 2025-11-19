import pandas as pd
import numpy as np
import pickle
import os
import sys

np.random.seed()

# --- Configuration & Paths ---
CSV_DIR = 'data/csv'
PKL_DIR = 'data/pkl_models'
TREND_FILE = os.path.join(CSV_DIR, 'hourly_trend.csv')
TUNING_FILE = os.path.join(CSV_DIR, 'tuning_params.csv')

# --- Load tuning params ---
try:
    tuning_df = pd.read_csv(TUNING_FILE, index_col=0)
except FileNotFoundError:
    sys.exit(f"Error: Tuning parameter file not found at '{TUNING_FILE}'")

# Build PARAMS dict dynamically
PARAMS = {
    'temp': {
        'phi': tuning_df.loc['temp', 'phi'],
        'sigma': tuning_df.loc['temp', 'sigma'],
        'max_err': 3.0,
        'file': 'temperature_gmm.pkl'
    },
    'wind': {
        'phi': tuning_df.loc['wind', 'phi'],
        'sigma': tuning_df.loc['wind', 'sigma'],
        'max_err': 4.0,
        'file': 'wind_speed_gmm.pkl'
    },
    'rad': {
        'phi': tuning_df.loc['rad', 'phi'],
        'sigma': tuning_df.loc['rad', 'sigma'],
        'max_err': None,
        'file': 'radiation_gmm.pkl'
    }
}

# --- Tuned Parameters ---
PARAMS = {
    'temp': {'phi': 0.9947, 'sigma': 0.1033, 'max_err': 3.0, 'file': 'temperature_gmm.pkl'},
    'wind': {'phi': 0.8921, 'sigma': 0.4519, 'max_err': 4.0, 'file': 'wind_speed_gmm.pkl'},
    'rad':  {'phi': 0.9337, 'sigma': 0.3581, 'max_err': None, 'file': 'radiation_gmm.pkl'}
}

# --- Load Resources ---
try:
    df_trend = pd.read_csv(TREND_FILE)
except FileNotFoundError:
    sys.exit(f"Error: Trend file not found at '{TREND_FILE}'")

# --- Load GMM models ---
models = {}
for key, config in PARAMS.items():
    path = os.path.join(PKL_DIR, config['file'])
    try:
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
            models[key].random_state = None
    except Exception as e:
        sys.exit(f"Error loading {key} model: {e}")

# --- Scenario Generation ---
scenario_bias = {
    'temp': np.random.normal(0, 7.16),
    'wind': np.random.normal(0, 1.10),
    'rad':  np.clip(np.random.normal(1.0, 0.3), 0.4, 1.5)
}

print(f"SCENARIO -> Temp Offset: {scenario_bias['temp']:+.2f}°C | Rad Gain: {scenario_bias['rad']:.2f}x")

# --- Simulation Loop ---
n_samples = len(df_trend)
output_data = {'temp': [], 'wind': [], 'rad': []}
residuals = {'temp': 0.0, 'wind': 0.0, 'rad': 0.0}

trends = {
    'temp': df_trend["Temp_Trend"].values,
    'wind': df_trend["Wind_Trend"].values,
    'rad':  df_trend["Rad_Trend"].values
}

for i in range(n_samples):
    # Update residuals
    for key in PARAMS:
        innovation = models[key].sample(1)[0][0][0]
        p = PARAMS[key]
        residuals[key] = (p['phi'] * residuals[key]) + (innovation * p['sigma'])
        if p['max_err']:
            residuals[key] = np.clip(residuals[key], -p['max_err'], p['max_err'])

    # Generate synthetic data
    val_temp = trends['temp'][i] + scenario_bias['temp'] + residuals['temp']
    output_data['temp'].append(val_temp)

    val_wind = max(0, trends['wind'][i] + scenario_bias['wind'] + residuals['wind'])
    output_data['wind'].append(val_wind)

    if trends['rad'][i] < 10:
        val_rad = 0.0
        residuals['rad'] = 0.0
    else:
        val_rad = max(0, (trends['rad'][i] * scenario_bias['rad']) + residuals['rad'])
    output_data['rad'].append(val_rad)

# --- Save Output ---
df_out = pd.DataFrame({
    "hour": df_trend["hour"],
    "trend_temp": trends['temp'],
    "synthetic_temp": output_data['temp'],
    "trend_wind": trends['wind'],
    "synthetic_wind": output_data['wind'],
    "trend_rad": trends['rad'],
    "synthetic_rad": output_data['rad']
})

print("\n===== DATA SNAPSHOT =====")
print(df_out)

out_file = os.path.join(CSV_DIR, 'synthetic_data_scenario.csv')
df_out.to_csv(out_file, index=False)
print(f"\nSaved to {out_file}")
