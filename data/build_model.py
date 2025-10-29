import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pickle
import sys
import os

OPTIMAL_K = {
    'Radiation': 3,     # Check plot ./plots/radiation_bic.png
    'Temperature': 2,    # Check plot ./plots/temperature_bic.png
    'WindSpeed': 4       # Check plot ./plots/wind_speed_bic.png
}
PLOT_DIR = 'plots'
PKL_DIR = 'pkl_models'
CSV_DIR = 'csv'
RESIDUALS_FILE = os.path.join(CSV_DIR, 'calculated_residuals.csv')


def gaussian_pdf(x, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_residuals(residuals_path):
    try:
        residuals_data = pd.read_csv(residuals_path, index_col='Timestamp', parse_dates=True)
        print(f"--- Residuals Loaded from {residuals_path} ---")
        print(residuals_data.head())
        return residuals_data
    except FileNotFoundError:
        print(f"Error: Residuals file '{residuals_path}' not found. Run analyze_data.py first.")
        sys.exit()
    except Exception as e:
        print(f"Error loading residuals file: {e}")
        sys.exit()


def fit_and_save_gmm(residuals_series, variable_name):
    k = OPTIMAL_K[variable_name.replace(" ", "")]

    print(f"\nFitting final GMM ({k} components) for {variable_name}...")
    residuals = residuals_series.dropna().values  # Use the passed Series

    if variable_name == 'Radiation':
        print("Filtering Radiation residuals: using only non-zero daytime noise (abs(residual) > 0.1).")
        residuals = residuals[np.abs(residuals) > 0.1]

    if len(residuals) == 0:
        print(f"Error: No valid residuals left for {variable_name} after filtering.")
        return None, None  # Return None if no data

    residuals_reshaped = residuals.reshape(-1, 1)

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(residuals_reshaped)

    model_filename = os.path.join(PKL_DIR, f"{variable_name.lower()}_gmm.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(gmm, f)

    print(f"Saved final GMM model to {model_filename}")

    return gmm, residuals


def plot_final_gmm_fit(residuals_for_plot, gmm, variable_name, unit):
    k = OPTIMAL_K[variable_name.replace(" ", "")]
    if gmm is None or residuals_for_plot is None or len(residuals_for_plot) == 0:
        print(f"Skipping plot for {variable_name} due to missing model or data.")
        return

    plt.figure(figsize=(12, 6))
    plt.hist(residuals_for_plot, bins=150, density=True, alpha=0.6, label='Residuals Histogram (Filtered for Rad)')

    x_min = residuals_for_plot.min()
    x_max = residuals_for_plot.max()
    x_range = x_max - x_min
    buffer = 0.1 * x_range if x_range > 1e-6 else 1.0
    x = np.linspace(x_min - buffer, x_max + buffer, 1000).reshape(-1, 1)

    weights = gmm.weights_
    means = gmm.means_.flatten()
    std_devs = np.sqrt(gmm.covariances_.flatten())

    # Plot individual components
    for i in range(k):
        comp_pdf = weights[i] * gaussian_pdf(x.flatten(), means[i], std_devs[i])
        plt.plot(x, comp_pdf, '--', label=f'Comp {i+1} (μ={means[i]:.2f}, σ={std_devs[i]:.2f})')

    # Plot total GMM PDF
    log_prob = gmm.score_samples(x)
    pdf = np.exp(log_prob)
    plt.plot(x, pdf, 'k-', linewidth=2, label=f'Total GMM Fit ({k} components)')

    plt.title(f'Final GMM Fit ({k} components) to {variable_name} Residuals')
    plt.xlabel(f'Residual {variable_name} ({unit})')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(x.min(), x.max())

    filepath = os.path.join(PLOT_DIR, f"{variable_name.lower()}_gmm.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved final fit plot: {filepath}")


def main():
    print(f"Using optimal k values based on prior analysis: {OPTIMAL_K}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    residuals_data = load_residuals(RESIDUALS_FILE)

    # Fit and save Gaussian Mixture Models
    gmm_rad, rad_residuals_fitted = fit_and_save_gmm(residuals_data['Rad_Residual'], 'Radiation')
    gmm_temp, temp_residuals_fitted = fit_and_save_gmm(residuals_data['Temp_Residual'], 'Temperature')
    gmm_wind, wind_residuals_fitted = fit_and_save_gmm(residuals_data['Wind_Residual'], 'WindSpeed')

    print("")

    plot_final_gmm_fit(rad_residuals_fitted, gmm_rad, 'Radiation', 'W/m²')
    plot_final_gmm_fit(temp_residuals_fitted, gmm_temp, 'Temperature', '°C')
    plot_final_gmm_fit(wind_residuals_fitted, gmm_wind, 'Wind Speed', 'm/s')

    print("\n--- Script Finished ---")
    print("Saved: *_gmm.pkl (final models based on chosen k)")
    print(f"Saved final GMM fit plots in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
