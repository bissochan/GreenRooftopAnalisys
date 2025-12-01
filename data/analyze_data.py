import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import sys
import os

PLOT_DIR = 'plots'
CSV_DIR = 'csv'


def load_and_clean_data(csv_path, data_type='meteo'):
    try:
        data = pd.read_csv(csv_path, skiprows=3, header=0)
    except FileNotFoundError:
        print(f"Error: Data file '{csv_path}' not found.")
        sys.exit()

    if data_type == 'meteo':
        data = data.rename(columns={
            'time': 'Timestamp',
            'temperature_2m (°C)': 'Temp_C',
            'wind_speed_10m (m/s)': 'Wind_ms',
            'shortwave_radiation (W/m²)': 'Radiation_Wm2'
        })
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
        data = data.set_index('Timestamp')
    elif data_type == 'air_quality':
        data = data.rename(columns={
            'time': 'Timestamp',
            'carbon_dioxide (ppm)': 'CO2_ppm',
            'pm10 (μg/m³)': 'PM10_ugm3',
            'pm2_5 (μg/m³)': 'PM2_5_ugm3',
            'carbon_monoxide (μg/m³)': 'CO_ugm3'
        })
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data = data.set_index('Timestamp')
    else:
        print("Unknown data type")
        sys.exit()

    print(f"--- {data_type} Data Loaded Successfully ---")
    print(data.head())
    return data


def calculate_detrend(data, data_type='meteo'):
    data['hour'] = data.index.hour

    if data_type == 'meteo':
        columns = ['Temp_C', 'Wind_ms', 'Radiation_Wm2']
        residual_name_map = {
            'Temp_C': 'Temperature_Residual',
            'Wind_ms': 'Wind_Speed_Residual',
            'Radiation_Wm2': 'Radiation_Residual'
        }

    else:
        columns = ['CO2_ppm', 'PM10_ugm3', 'PM2_5_ugm3', 'CO_ugm3']
        residual_name_map = {
            'CO2_ppm': 'CO2_Residual',
            'PM10_ugm3': 'PM10_Residual',
            'PM2_5_ugm3': 'PM2_5_Residual',
            'CO_ugm3': 'CO_Residual'
        }

    # Hourly trend calculation
    hourly_trend = data.groupby('hour')[columns].mean()
    trend_columns = {col: f"{col}_Trend" for col in columns}
    hourly_trend = hourly_trend.rename(columns=trend_columns)

    # Join trend to original data
    data = data.join(hourly_trend, on='hour')

    # Calculate residuals
    for col in columns:
        residual_col = residual_name_map[col]
        data[residual_col] = data[col] - data[f"{col}_Trend"]

        if data_type == 'air_quality' and col == 'CO2':
            data[residual_col] = data[residual_col].dropna()

    trend_file = os.path.join(CSV_DIR, f'{data_type}_hourly_trend.csv')
    residual_file = os.path.join(CSV_DIR, f'{data_type}_residuals.csv')
    hourly_trend.to_csv(trend_file)
    data[[residual_name_map[col] for col in columns]].to_csv(residual_file)

    print(f"\n--- {data_type} Trend Saved to {trend_file} ---")
    print(f"--- {data_type} Residuals Saved to {residual_file} ---")

    return data


def calculate_tuning_params(data, data_type='meteo'):
    if data_type == 'meteo':
        variables = ['Temperature_Residual', 'Wind_Speed_Residual', 'Radiation_Residual']
    else:
        variables = ['CO2_Residual', 'PM10_Residual', 'PM2_5_Residual', 'CO_Residual']

    results = {}

    for col in variables:
        series = data[col].dropna()
        phi = series.autocorr(lag=1)
        if abs(phi) >= 1:
            sigma = 0.0
        else:
            sigma = np.sqrt(1 - phi**2)
        daily_means = series.resample('D').mean()
        bias = daily_means.std()
        results[col] = {
            'phi': round(phi, 4),
            'sigma': round(sigma, 4),
            'bias': round(bias, 4)
        }

    out_file = os.path.join(CSV_DIR, f'{data_type}_tuning_params.csv')
    pd.DataFrame(results).T.to_csv(out_file)
    print(f"\n--- {data_type} Tuning Parameters Saved to {out_file} ---")
    print(pd.DataFrame(results).T)
    return results


def plot_histogram(residuals, variable_name, unit):
    plt.figure(figsize=(10, 6))
    residuals.hist(bins=100, density=True, alpha=0.7)
    plt.xlabel(f'Detrended {variable_name} ({unit})')
    plt.ylabel('Density')
    plt.title(f'Histogram of Detrended {variable_name} Data (Residuals)')
    filename = os.path.join(PLOT_DIR, f'{variable_name.lower().replace(" ", "_").replace(".", "_")}_residuals_hist.png')
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def analyze_bic_for_gmm(residuals, max_k, variable_name, output_dir):
    print(f"\nAnalyzing BIC for GMM components for {variable_name}...")
    residuals = residuals.dropna().values
    if len(residuals) == 0:
        print(f"Warning: No valid residuals left for {variable_name}. Skipping BIC analysis.")
        return []

    residuals_reshaped = residuals.reshape(-1, 1)
    k_range = range(1, max_k + 1)
    bic_scores = []
    for k in k_range:
        try:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(residuals_reshaped)
            bic_scores.append(gmm.bic(residuals_reshaped))
        except ValueError:
            bic_scores.append(np.inf)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, bic_scores, 'o-')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('BIC Score')
    plt.title(f'BIC Model Selection for {variable_name} Residuals')
    plt.grid(True)
    bic_plot_filename = os.path.join(output_dir, f'{variable_name.lower().replace(" ", "_").replace(".", "_")}_bic.png')
    plt.savefig(bic_plot_filename)
    plt.close()
    print(f"Saved BIC plot: {bic_plot_filename}")
    return bic_scores


def main():
    MAX_K_COMPONENTS = 8
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # METEO
    meteo_data = load_and_clean_data('open-meteo.csv', data_type='meteo')
    meteo_detrended = calculate_detrend(meteo_data, data_type='meteo')
    calculate_tuning_params(meteo_detrended, data_type='meteo')
    plot_histogram(meteo_detrended['Radiation_Residual'], 'Radiation', 'W/m²')
    plot_histogram(meteo_detrended['Temperature_Residual'], 'Temperature', '°C')
    plot_histogram(meteo_detrended['Wind_Speed_Residual'], 'Wind Speed', 'm/s')
    analyze_bic_for_gmm(meteo_detrended['Radiation_Residual'], MAX_K_COMPONENTS, 'Radiation', PLOT_DIR)
    analyze_bic_for_gmm(meteo_detrended['Temperature_Residual'], MAX_K_COMPONENTS, 'Temperature', PLOT_DIR)
    analyze_bic_for_gmm(meteo_detrended['Wind_Speed_Residual'], MAX_K_COMPONENTS, 'Wind Speed', PLOT_DIR)

    # AIR QUALITY
    air_data = load_and_clean_data('air_quality.csv', data_type='air_quality')
    air_detrended = calculate_detrend(air_data, data_type='air_quality')
    calculate_tuning_params(air_detrended, data_type='air_quality')
    plot_histogram(air_detrended['CO2_Residual'], 'CO2', 'ppm')
    plot_histogram(air_detrended['PM10_Residual'], 'PM10', 'μg/m³')
    plot_histogram(air_detrended['PM2_5_Residual'], 'PM2.5', 'μg/m³')
    plot_histogram(air_detrended['CO_Residual'], 'CO', 'μg/m³')
    analyze_bic_for_gmm(air_detrended['CO2_Residual'], MAX_K_COMPONENTS, 'CO2', PLOT_DIR)
    analyze_bic_for_gmm(air_detrended['PM10_Residual'], MAX_K_COMPONENTS, 'PM10', PLOT_DIR)
    analyze_bic_for_gmm(air_detrended['PM2_5_Residual'], MAX_K_COMPONENTS, 'PM2.5', PLOT_DIR)
    analyze_bic_for_gmm(air_detrended['CO_Residual'], MAX_K_COMPONENTS, 'CO', PLOT_DIR)


if __name__ == "__main__":
    main()
