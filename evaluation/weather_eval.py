import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

DATA_DIR = os.path.join("..", "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
SIM_DIR = os.path.join("..", "simulation")

WEATHER_SAMPLES_FILE = os.path.join(SIM_DIR, "csv_results", "weather_samples.csv")
TREND_FILE = os.path.join(CSV_DIR, "meteo_hourly_trend.csv")
REAL_DATA_FILE = os.path.join(DATA_DIR, "open-meteo.csv")

OUTPUT_DIR = "results"
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "weather_evaluation_report.txt")
N_SIMULATIONS = 100


def load_data():
    try:
        df_sim = pd.read_csv(WEATHER_SAMPLES_FILE)
        if 'hour' not in df_sim.columns:
            df_sim['hour'] = df_sim.index % 24

        df_trend = pd.read_csv(TREND_FILE)
        trend_map = {
            'Temp_C_Trend': 'Temp_Trend',
            'Wind_ms_Trend': 'Wind_Trend',
            'Radiation_Wm2_Trend': 'Rad_Trend'
        }
        df_trend = df_trend.rename(columns=trend_map)
        if 'hour' not in df_trend.columns:
            df_trend['hour'] = df_trend.index

        df_real = pd.read_csv(REAL_DATA_FILE, skiprows=2)
        df_real = df_real.rename(columns={
            'time': 'Timestamp',
            'temperature_2m (°C)': 'temp_real',
            'wind_speed_10m (m/s)': 'wind_real',
            'shortwave_radiation (W/m²)': 'rad_real'
        })
        df_real['Timestamp'] = pd.to_datetime(df_real['Timestamp'], unit='s')
        df_real['hour'] = df_real['Timestamp'].dt.hour

        df_real_hourly = df_real.groupby('hour').agg(
            real_temp_std=('temp_real', 'std'),
            real_wind_std=('wind_real', 'std'),
            real_rad_std=('rad_real', 'std')
        ).reset_index()

        return df_sim, df_trend, df_real, df_real_hourly

    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None, None, None


def save_plot_mean(df_comp, var, title, unit, ci_error):
    plt.figure(figsize=(8, 5))
    trend_col = f'{var.capitalize()}_Trend'
    gen_mean_col = f'gen_{var}_mean'

    if trend_col in df_comp.columns:
        plt.plot(df_comp['hour'], df_comp[trend_col], label='Historical Trend', color='black', linestyle='--')

    plt.plot(df_comp['hour'], df_comp[gen_mean_col], label='Generated Mean', color='red', linewidth=2)
    plt.fill_between(df_comp['hour'],
                     df_comp[gen_mean_col] - ci_error,
                     df_comp[gen_mean_col] + ci_error,
                     color='red', alpha=0.15, label='95% CI')

    plt.title(f"{title} Mean Comparison")
    plt.xlabel("Hour")
    plt.ylabel(f"Value {unit}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_mean.png"))
    plt.close()


def save_plot_std(df_comp, var, title, unit):
    plt.figure(figsize=(8, 5))
    real_std_col = f'real_{var}_std'
    gen_std_col = f'gen_{var}_std'

    plt.plot(df_comp['hour'], df_comp[real_std_col], label='Real Data Std', color='green', linewidth=2)
    plt.plot(df_comp['hour'], df_comp[gen_std_col], label='Generated Std', color='purple', linestyle='--')

    plt.title(f"{title} Variability (Std Dev)")
    plt.xlabel("Hour")
    plt.ylabel(f"Std Dev {unit}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_std.png"))
    plt.close()


def save_plot_ecdf(real_data, gen_data, var, title, unit):
    plt.figure(figsize=(8, 5))
    x_real = np.sort(real_data)
    y_real = np.arange(1, len(x_real) + 1) / len(x_real)
    x_gen = np.sort(gen_data)
    y_gen = np.arange(1, len(x_gen) + 1) / len(x_gen)

    plt.plot(x_real, y_real, label='Real Data', color='green')
    plt.plot(x_gen, y_gen, label='Generated Data', color='purple', linestyle='--')

    plt.title(f"{title} ECDF")
    plt.xlabel(f"Value {unit}")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_ecdf.png"))
    plt.close()


def save_plot_qq(gen_data, var, title):
    plt.figure(figsize=(8, 5))
    stats.probplot(gen_data, dist="norm", plot=plt)
    plt.title(f"{title} QQ Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_qq.png"))
    plt.close()


def run_evaluation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_sim, df_trend, df_real, df_real_hourly = load_data()
    if df_sim is None:
        return

    df_sim_hourly = df_sim.groupby('hour').agg(
        gen_temp_mean=('temp', 'mean'),
        gen_temp_std=('temp', 'std'),
        gen_wind_mean=('wind', 'mean'),
        gen_wind_std=('wind', 'std'),
        gen_rad_mean=('rad', 'mean'),
        gen_rad_std=('rad', 'std'),
    ).reset_index()

    df_comp = pd.merge(df_sim_hourly, df_trend, on='hour', how='left')
    df_comp = pd.merge(df_comp, df_real_hourly, on='hour', how='left')

    report = []
    report.append("==============================================")
    report.append("      WEATHER DATA PERFORMANCE EVALUATION")
    report.append("==============================================\n")

    t_val = stats.t.ppf(0.975, N_SIMULATIONS - 1)

    variables = [
        ('temp', 'Temperature', '(°C)'),
        ('wind', 'Wind Speed', '(m/s)'),
        ('rad', 'Radiation', '(W/m2)')
    ]

    for var, title, unit in variables:
        print(f"Processing {title}...")
        gen_std_col = f'gen_{var}_std'
        ci_error = t_val * (df_comp[gen_std_col] / np.sqrt(N_SIMULATIONS))
        save_plot_mean(df_comp, var, title, unit, ci_error)
        save_plot_std(df_comp, var, title, unit)

        real_series = df_real[f'{var}_real'].dropna().values
        gen_series = df_sim[var].dropna().values

        save_plot_ecdf(real_series, gen_series, var, title, unit)

        gen_sample = np.random.choice(gen_series, size=min(len(gen_series), 2000), replace=False)
        save_plot_qq(gen_sample, var, title)

        trend_col = f'{var.capitalize()}_Trend'
        gen_mean_col = f'gen_{var}_mean'
        real_std_col = f'real_{var}_std'

        if trend_col in df_comp.columns:
            bias_trend = (df_comp[gen_mean_col] - df_comp[trend_col]).abs().mean()
        else:
            bias_trend = np.nan

        std_error = (df_comp[gen_std_col] - df_comp[real_std_col]).abs().mean()

        p_real = np.percentile(real_series, [5, 50, 95, 99])
        p_gen = np.percentile(gen_series, [5, 50, 95, 99])

        cov_real = np.std(real_series) / np.mean(real_series) if np.mean(real_series) != 0 else 0
        cov_gen = np.std(gen_series) / np.mean(gen_series) if np.mean(gen_series) != 0 else 0

        ks_stat, ks_p = stats.ks_2samp(real_series, gen_series)

        report.append(f"[{title.upper()}]")
        report.append("-" * 40)
        report.append(f"MEAN BIAS (vs Trend)      : {bias_trend:.4f} {unit}")
        report.append(f"STD DEV ERROR (vs Real)   : {std_error:.4f} {unit}")
        report.append(f"COEFF. VARIATION (R / G)  : {cov_real:.3f} / {cov_gen:.3f}")
        report.append(f"KS TEST STATISTIC         : {ks_stat:.4f}")
        report.append("PERCENTILES (Real vs Gen) :")
        report.append(f"  P05 (Low)    : {p_real[0]:6.1f} | {p_gen[0]:6.1f}")
        report.append(f"  P50 (Median) : {p_real[1]:6.1f} | {p_gen[1]:6.1f}")
        report.append(f"  P95 (High)   : {p_real[2]:6.1f} | {p_gen[2]:6.1f}")
        report.append(f"  P99 (Extreme): {p_real[3]:6.1f} | {p_gen[3]:6.1f}")
        report.append("\n")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\nProcessing Complete.")
    print(f"Plots saved in: {OUTPUT_DIR}/")
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_evaluation()
