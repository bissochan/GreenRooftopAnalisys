# --- build_nodel.py aggiornato ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pickle
import sys
import os

OPTIMAL_K = {
    'Radiation': 6,
    'Temperature': 3,
    'WindSpeed': 2,
    'CO2': 4,
    'PM10': 4,
    'PM2_5': 4,
    'CO': 3
}

PLOT_DIR = 'plots'
PKL_DIR = 'pkl_models'
CSV_DIR = 'csv'
RESIDUALS_FILE_METEO = os.path.join(CSV_DIR, 'meteo_residuals.csv')
RESIDUALS_FILE_AIR = os.path.join(CSV_DIR, 'air_quality_residuals.csv')



def gaussian_pdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_residuals(residuals_path):
    try:
        data = pd.read_csv(residuals_path, index_col='Timestamp', parse_dates=True)
        print(f"--- Residuals Loaded from {residuals_path} ---")
        return data
    except Exception as e:
        print(f"Error loading residuals file: {e}")
        sys.exit()


def fit_and_save_gmm(residuals_series, variable_name):
    k = OPTIMAL_K[variable_name.replace(" ", "")]
    residuals = residuals_series.dropna().values
    if len(residuals) == 0:
        print(f"Warning: No valid residuals for {variable_name}")
        return None, None
    residuals_reshaped = residuals.reshape(-1, 1)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(residuals_reshaped)
    os.makedirs(PKL_DIR, exist_ok=True)
    model_filename = os.path.join(PKL_DIR, f"{variable_name.lower().replace(' ', '_')}_gmm.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"Saved final GMM model to {model_filename}")
    return gmm, residuals


def plot_final_gmm_fit(residuals_for_plot, gmm, variable_name, unit):
    if gmm is None or residuals_for_plot is None or len(residuals_for_plot) == 0:
        print(f"Skipping plot for {variable_name} due to missing model or data.")
        return
    k = OPTIMAL_K[variable_name.replace(" ", "")]
    plt.figure(figsize=(12, 6))
    plt.hist(residuals_for_plot, bins=150, density=True, alpha=0.6)
    x = np.linspace(residuals_for_plot.min(), residuals_for_plot.max(), 1000).reshape(-1, 1)
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    for i in range(k):
        plt.plot(x, weights[i] * gaussian_pdf(x.flatten(), means[i], stds[i]), '--')
    pdf = np.exp(gmm.score_samples(x))
    plt.plot(x, pdf, 'k-', linewidth=2)
    plt.title(f'Final GMM Fit ({k} components) to {variable_name} Residuals')
    plt.xlabel(f'Residual {variable_name} ({unit})')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    filepath = os.path.join(PLOT_DIR, f"{variable_name.lower().replace(' ', '_')}_gmm.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved final fit plot: {filepath}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # METEO
    meteo_res = load_residuals(RESIDUALS_FILE_METEO)
    for var, unit in zip(['Radiation_Residual', 'Temperature_Residual', 'WindSpeed_Residual'], ['W/m²', '°C', 'm/s']):
        gmm, res = fit_and_save_gmm(meteo_res[var], var.replace('_Residual', ''))
        plot_final_gmm_fit(res, gmm, var.replace('_Residual', ''), unit)


    # AIR QUALITY
    air_res = load_residuals(RESIDUALS_FILE_AIR)
    for var, unit in zip(['CO2_Residual', 'PM10_Residual', 'PM2_5_Residual', 'CO_Residual'], ['ppm', 'μg/m³', 'μg/m³', 'μg/m³']):
        gmm, res = fit_and_save_gmm(air_res[var], var.replace('_Residual', ''))
        plot_final_gmm_fit(res, gmm, var.replace('_Residual', ''), unit)


if __name__ == "__main__":
    main()
