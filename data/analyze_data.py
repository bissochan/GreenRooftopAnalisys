import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import sys
import os

PLOT_DIR = 'plots'
CSV_DIR = 'csv'

def load_and_clean_data(csv_path):
    try:
        data = pd.read_csv(csv_path, skiprows=9, header=0)  # Skip metadata rows (9 rows)
    except FileNotFoundError:
        print(f"Error: Data file '{csv_path}' not found. Please download it first.")
        sys.exit()

    data = data.rename(columns={
        'timestamp': 'Timestamp',
        'Basel Temperature [2 m elevation corrected]': 'Temp_C',
        'Basel Wind Speed [10 m]': 'Wind_kmh',
        'Basel Shortwave Radiation': 'Radiation_Wm2'
    })
    data = data[['Timestamp', 'Temp_C', 'Wind_kmh', 'Radiation_Wm2']]

    data['Timestamp'] = pd.to_datetime(data['Timestamp'])  # Convert to datetime
    data['Wind_ms'] = data['Wind_kmh'] * (1000 / 3600)  # Convert km/h to m/s
    data = data.set_index('Timestamp')
    data = data.drop(columns=['Wind_kmh'])  # Drop original km/h column

    print("--- Data Loaded and Cleaned (first 5 rows) ---")
    print(data.head())
    return data

def calculate_detrend(data):
    data['hour'] = data.index.hour
    hourly_trend = data.groupby('hour')[['Temp_C', 'Wind_ms', 'Radiation_Wm2']].mean()

    hourly_trend = hourly_trend.rename(columns={
        'Temp_C': 'Temp_Trend',
        'Wind_ms': 'Wind_Trend',
        'Radiation_Wm2': 'Rad_Trend'
    })

    data = data.join(hourly_trend, on='hour')
    data['Temp_Residual'] = data['Temp_C'] - data['Temp_Trend']
    data['Wind_Residual'] = data['Wind_ms'] - data['Wind_Trend']
    data['Rad_Residual'] = data['Radiation_Wm2'] - data['Rad_Trend']

    # Save the trend data and residual data
    hourly_trend.to_csv(os.path.join(CSV_DIR, 'hourly_trend.csv'))
    print("\n--- Deterministic 24-hour Trend (Saved to hourly_trend.csv) ---")
    print(hourly_trend.head())

    residual_df = data[['Temp_Residual', 'Wind_Residual', 'Rad_Residual']]
    residual_filename = os.path.join(CSV_DIR, 'calculated_residuals.csv')
    residual_df.to_csv(residual_filename)
    print(f"\n--- Calculated Residuals (Saved to {residual_filename}) ---")
    print(residual_df.head())

    return data

def plot_histogram(residuals, variable_name, unit):
    plt.figure(figsize=(10, 6))
    residuals.hist(bins=100, density=True, alpha=0.7)
    plt.xlabel(f'Detrended {variable_name} ({unit})')
    plt.ylabel('Density')
    plt.title(f'Histogram of Detrended {variable_name} Data (Residuals)')
    filename = os.path.join(PLOT_DIR, f'{variable_name.lower().replace(" ", "_")}_residuals_hist.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def analyze_bic_for_gmm(residuals, max_k, variable_name, output_dir):
    print(f"\nAnalyzing BIC for GMM components for {variable_name}...")
    residuals = residuals.dropna().values

    # Filter out near-zero residuals for Radiation
    if variable_name == 'Radiation':
        print("Filtering Radiation residuals: keeping only non-zero daytime noise for BIC analysis.")
        residuals = residuals[np.abs(residuals) > 0.1]

    if len(residuals) == 0:
        print(f"Warning: No valid residuals left for {variable_name} after filtering. Skipping BIC analysis.")
        return []

    residuals_reshaped = residuals.reshape(-1, 1)
    k_range = range(1, max_k + 1)
    bic_scores = []

    for k in k_range:
        try:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(residuals_reshaped)
            bic = gmm.bic(residuals_reshaped)
            bic_scores.append(bic)
            print(f"  k={k}: BIC = {bic:.2f}")
        except ValueError as e:
            print(f"  k={k}: Error fitting GMM - {e}. Skipping this k.")
            bic_scores.append(np.inf)

    # Plot BIC scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, bic_scores, 'o-')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('BIC Score')
    plt.title(f'BIC Model Selection for {variable_name} Residuals')
    valid_k = [k for k, score in zip(k_range, bic_scores) if score != np.inf]
    if valid_k:
        plt.xticks(valid_k)
    plt.grid(True)
    bic_plot_filename = os.path.join(output_dir, f'{variable_name.lower().replace(" ", "_")}_bic.png')
    plt.savefig(bic_plot_filename)
    plt.close()
    print(f"Saved BIC plot: {bic_plot_filename}")

    return bic_scores

def main():
    CSV_FILE = 'meteoblue_data_2020_2025.csv'
    MAX_K_COMPONENTS = 8

    # Create output directory
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    cleaned_data = load_and_clean_data(CSV_FILE)
    detrended_data = calculate_detrend(cleaned_data)

    # Plot initial analysis graphs
    plot_histogram(detrended_data['Rad_Residual'], 'Radiation', 'W/m²')
    plot_histogram(detrended_data['Temp_Residual'], 'Temperature', '°C')
    plot_histogram(detrended_data['Wind_Residual'], 'Wind Speed', 'm/s')

    # Run BIC analysis
    analyze_bic_for_gmm(detrended_data['Rad_Residual'], MAX_K_COMPONENTS, 'Radiation', PLOT_DIR)
    analyze_bic_for_gmm(detrended_data['Temp_Residual'], MAX_K_COMPONENTS, 'Temperature', PLOT_DIR)
    analyze_bic_for_gmm(detrended_data['Wind_Residual'], MAX_K_COMPONENTS, 'Wind Speed', PLOT_DIR)

    print("\n--- BIC Analysis Finished ---")
    print("Saved: hourly_trend.csv")
    print(f"Saved analysis plots in: {PLOT_DIR}")

if __name__ == "__main__":
    main()