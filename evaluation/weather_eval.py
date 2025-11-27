import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Basic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 100})

COLOR_REAL = 'seagreen'
COLOR_SIM = 'crimson'
COLOR_TREND = 'black'

DATA_DIR = os.path.join("..", "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
SIM_DIR = os.path.join("..", "simulation")

FILES = {
    "sim": os.path.join(SIM_DIR, "csv_results", "weather_samples.csv"),
    "trend": os.path.join(CSV_DIR, "meteo_hourly_trend.csv"),
    "real": os.path.join(DATA_DIR, "open-meteo.csv")
}

# OUTPUT_DIR = "results_weather"
OUTPUT_DIR = os.path.join("results", "weather")

OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "weather_evaluation_report.txt")
N_SIMULATIONS = 100

# ----------------
# DATA LOADING
# ----------------


def load_data():
    print("Loading datasets...")

    try:
        df_sim = pd.read_csv(FILES["sim"])
        if 'hour' not in df_sim.columns:
            df_sim['hour'] = df_sim.index % 24
    except FileNotFoundError:
        return None, None, None, None

    try:
        df_trend = pd.read_csv(FILES["trend"])
        df_trend = df_trend.rename(columns={
            'Temp_C_Trend': 'Temp_Trend',
            'Wind_ms_Trend': 'Wind_Trend',
            'Radiation_Wm2_Trend': 'Rad_Trend'
        })
        if 'hour' not in df_trend.columns:
            df_trend['hour'] = df_trend.index
    except FileNotFoundError:
        return None, None, None, None

    try:
        df_real = pd.read_csv(FILES["real"], skiprows=2)
        df_real = df_real.rename(columns={
            'temperature_2m (°C)': 'temp',
            'wind_speed_10m (m/s)': 'wind',
            'shortwave_radiation (W/m²)': 'rad'
        })
        df_real['time'] = pd.to_datetime(df_real['time'], unit='s')
        df_real['hour'] = df_real['time'].dt.hour

        # Hourly std of real data for comparison
        df_real_hourly = df_real.groupby('hour').agg(
            temp_std=('temp', 'std'),
            wind_std=('wind', 'std'),
            rad_std=('rad', 'std')
        ).reset_index()

    except FileNotFoundError:
        return None, None, None, None

    print("Data loaded")
    return df_sim, df_trend, df_real, df_real_hourly

# ----------------
# PLOTTING
# ----------------


def save_plot_mean(df_comp, var, title, unit, ci_error):
    plt.figure()
    trend_col = f'{var.capitalize()}_Trend'
    gen_mean_col = f'gen_{var}_mean'

    if trend_col in df_comp.columns:
        plt.plot(df_comp['hour'], df_comp[trend_col], color=COLOR_TREND, ls='--', label='Historical Trend')

    plt.plot(df_comp['hour'], df_comp[gen_mean_col], color=COLOR_SIM, label='Simulated Mean')

    plt.fill_between(df_comp['hour'], df_comp[gen_mean_col] - ci_error,
                     df_comp[gen_mean_col] + ci_error, color=COLOR_SIM, alpha=0.2)

    plt.title(f"{title}: Mean Hourly Profile")
    plt.xlabel("Hour")
    plt.ylabel(f"{title} {unit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_mean.png"))
    plt.close()


def save_plot_std(df_comp, var, title, unit):
    plt.figure()

    real_std_col = f'{var}_std'
    gen_std_col = f'gen_{var}_std'

    plt.plot(df_comp['hour'], df_comp[real_std_col], color=COLOR_REAL, label='Real Std')
    plt.plot(df_comp['hour'], df_comp[gen_std_col], color=COLOR_SIM, ls='--', label='Sim Std')

    plt.title(f"{title}: Hourly Std Dev")
    plt.xlabel("Hour")
    plt.ylabel(f"Std {unit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_std.png"))
    plt.close()


def save_plot_kde(real_data, gen_data, var, title, unit):
    plt.figure()
    sns.kdeplot(real_data, fill=True, label="Real", color=COLOR_REAL, alpha=0.3)
    sns.kdeplot(gen_data, fill=True, label="Sim", color=COLOR_SIM, alpha=0.3)

    plt.title(f"{title}: Distribution")
    plt.xlabel(f"{title} {unit}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_kde.png"))
    plt.close()


def save_plot_ecdf(real_data, gen_data, var, title, unit):
    plt.figure()

    x_r = np.sort(real_data)
    y_r = np.arange(1, len(x_r) + 1) / len(x_r)
    x_g = np.sort(gen_data)
    y_g = np.arange(1, len(x_g) + 1) / len(x_g)

    plt.plot(x_r, y_r, label='Real', color=COLOR_REAL)
    plt.plot(x_g, y_g, label='Sim', color=COLOR_SIM, ls='--')

    plt.title(f"{title}: ECDF")
    plt.xlabel(f"{title} {unit}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_ecdf.png"))
    plt.close()


def save_plot_qq(gen_data, var, title):
    plt.figure()
    stats.probplot(gen_data, dist="norm", plot=plt)
    plt.title(f"{title}: QQ Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_qq.png"))
    plt.close()


def save_correlation_heatmap(df, name):
    plt.figure(figsize=(6, 5))
    cols = ['temp', 'wind', 'rad']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Correlation Matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"CorrMatrix_{name}.png"))
    plt.close()


def save_plot_hourly_boxplot(df_sim, df_real, var, title, unit):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sns.boxplot(x='hour', y=var, data=df_real, ax=ax1,
                color=COLOR_REAL, showfliers=False)
    ax1.set_title("Real Data")

    sns.boxplot(x='hour', y=var, data=df_sim, ax=ax2,
                color=COLOR_SIM, showfliers=False)
    ax2.set_title("Simulated Data")
    ax2.set_xlabel("Hour")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_hourly_box.png"))
    plt.close()

# ----------------
# METRICS
# ----------------


def calculate_metrics(df_comp, real_series, gen_series, var, trend_col):
    metrics = {}

    gen_mean_col = f'gen_{var}_mean'
    if trend_col in df_comp.columns:
        metrics['bias'] = (df_comp[gen_mean_col] - df_comp[trend_col]).abs().mean()
    else:
        metrics['bias'] = np.nan

    gen_std_col = f'gen_{var}_std'
    real_std_col = f'{var}_std'
    metrics['std_error'] = (df_comp[gen_std_col] - df_comp[real_std_col]).abs().mean()

    ks_stat, _ = stats.ks_2samp(real_series, gen_series)
    metrics['ks_stat'] = ks_stat

    cov_real = np.std(real_series) / np.mean(real_series) if np.mean(real_series) != 0 else 0
    cov_gen = np.std(gen_series) / np.mean(gen_series) if np.mean(gen_series) != 0 else 0
    metrics['cov_real'] = cov_real
    metrics['cov_gen'] = cov_gen

    metrics['p_real'] = np.percentile(real_series, [5, 50, 95, 99])
    metrics['p_gen'] = np.percentile(gen_series, [5, 50, 95, 99])

    return metrics

# ----------------
# MAIN
# ----------------


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_sim, df_trend, df_real, df_real_hourly = load_data()
    if df_sim is None:
        print("Files missing.")
        return

    save_correlation_heatmap(df_real, "RealData")
    save_correlation_heatmap(df_sim, "SimulatedData")

    vars_config = [
        ('temp', 'Temperature', '(°C)'),
        ('wind', 'Wind Speed', '(m/s)'),
        ('rad', 'Solar Radiation', '(W/m²)')
    ]

    df_sim_hourly = df_sim.groupby('hour').agg(
        gen_temp_mean=('temp', 'mean'), gen_temp_std=('temp', 'std'),
        gen_wind_mean=('wind', 'mean'), gen_wind_std=('wind', 'std'),
        gen_rad_mean=('rad', 'mean'), gen_rad_std=('rad', 'std'),
    ).reset_index()

    df_comp = pd.merge(df_sim_hourly, df_trend, on='hour', how='left')
    df_comp = pd.merge(df_comp, df_real_hourly, on='hour', how='left')

    t_val = stats.t.ppf(0.975, N_SIMULATIONS - 1)
    report = []

    report.append("WEATHER DATA PERFORMANCE REPORT\n")

    for var, title, unit in vars_config:
        real_series = df_real[var].dropna().values
        gen_series = df_sim[var].dropna().values
        one_run_series = df_sim[df_sim['run_id'] == 0][var].values

        gen_std_col = f'gen_{var}_std'
        ci_error = t_val * (df_comp[gen_std_col] / np.sqrt(N_SIMULATIONS))
        save_plot_mean(df_comp, var, title, unit, ci_error)
        save_plot_std(df_comp, var, title, unit)
        save_plot_kde(real_series, gen_series, var, title, unit)
        save_plot_ecdf(real_series, gen_series, var, title, unit)

        gen_sample = np.random.choice(gen_series, size=min(len(gen_series), 2000), replace=False)
        save_plot_qq(gen_sample, var, title)

        save_plot_hourly_boxplot(df_sim, df_real, var, title, unit)

        m = calculate_metrics(df_comp, real_series, gen_series, var, f'{var.capitalize()}_Trend')

        report.append(f"[{title.upper()}]")
        report.append(f"MEAN BIAS: {m['bias']:.4f} {unit}")
        report.append(f"STD ERROR: {m['std_error']:.4f} {unit}")
        report.append(f"CoV (Real / Sim): {m['cov_real']:.3f} / {m['cov_gen']:.3f}")
        report.append(f"KS STAT: {m['ks_stat']:.4f}")
        report.append(f"PERCENTILES (R | S):")
        report.append(f"  P05: {m['p_real'][0]:.1f} | {m['p_gen'][0]:.1f}")
        report.append(f"  P50: {m['p_real'][1]:.1f} | {m['p_gen'][1]:.1f}")
        report.append(f"  P95: {m['p_real'][2]:.1f} | {m['p_gen'][2]:.1f}")
        report.append(f"  P99: {m['p_real'][3]:.1f} | {m['p_gen'][3]:.1f}")
        report.append("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Results saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
