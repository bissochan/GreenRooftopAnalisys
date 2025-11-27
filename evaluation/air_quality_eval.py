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
    "sim": os.path.join(SIM_DIR, "csv_results", "air_quality_samples.csv"),
    "trend": os.path.join(CSV_DIR, "air_quality_hourly_trend.csv"),
    "real": os.path.join(DATA_DIR, "air_quality.csv")
}

OUTPUT_DIR = os.path.join("results", "air_quality_test")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "air_quality_evaluation_report.txt")
N_SIMULATIONS = 100

# ----------------
# DATA LOADING
# ----------------


def load_data():
    print("Loading datasets...")

    # Load simulated MC data
    try:
        df_sim = pd.read_csv(FILES["sim"])
        if 'hour' not in df_sim.columns:
            df_sim['hour'] = df_sim.index % 24
    except FileNotFoundError:
        return None, None, None, None

    # Load historical trends
    try:
        df_trend = pd.read_csv(FILES["trend"])
        df_trend = df_trend.rename(columns={
            'CO2_ppm_Trend': 'co2_Trend',
            'CO_ugm3_Trend': 'co_Trend',
            'PM10_ugm3_Trend': 'pm10_Trend',
            'PM2_5_ugm3_Trend': 'pm25_Trend'
        })
        if 'hour' not in df_trend.columns:
            df_trend['hour'] = df_trend.index
    except FileNotFoundError:
        return None, None, None, None

    # Load real hourly data
    try:
        # skip first 2 metadata lines
        df_real = pd.read_csv(FILES["real"], skiprows=2)

        # rename columns in a clean way
        df_real = df_real.rename(columns={
            'time': 'timestamp',
            'carbon_dioxide (ppm)': 'co2',
            'pm10 (μg/m³)': 'pm10',
            'pm2_5 (μg/m³)': 'pm25',
            'carbon_monoxide (μg/m³)': 'co'
        })

        # convert timestamp and extract hour
        df_real['timestamp'] = pd.to_datetime(df_real['timestamp'], errors='coerce')
        df_real['hour'] = df_real['timestamp'].dt.hour

        # aggregate hourly stats
        df_real_hourly = df_real.groupby('hour').agg(
            co2_std=('co2', 'std'),
            co_std=('co', 'std'),
            pm10_std=('pm10', 'std'),
            pm25_std=('pm25', 'std')
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
    trend_col = f'{var}_Trend'
    mean_col = f'gen_{var}_mean'

    if trend_col in df_comp.columns:
        plt.plot(df_comp['hour'], df_comp[trend_col], color=COLOR_TREND, ls='--', label='Historical Trend')

    plt.plot(df_comp['hour'], df_comp[mean_col], color=COLOR_SIM, label='Simulated Mean')

    plt.fill_between(
        df_comp['hour'],
        df_comp[mean_col] - ci_error,
        df_comp[mean_col] + ci_error,
        alpha=0.2,
        color=COLOR_SIM
    )

    plt.title(f"{title}: Mean Hourly Profile")
    plt.xlabel("Hour")
    plt.ylabel(f"{title} {unit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_mean.png"))
    plt.close()


def save_plot_std(df_comp, var, title, unit):
    plt.figure()
    real_std = f'{var}_std'
    sim_std = f'gen_{var}_std'

    plt.plot(df_comp['hour'], df_comp[real_std], color=COLOR_REAL, label="Real Std")
    plt.plot(df_comp['hour'], df_comp[sim_std], color=COLOR_SIM, ls="--", label="Sim Std")

    plt.title(f"{title}: Hourly Std Dev")
    plt.xlabel("Hour")
    plt.ylabel(f"Std {unit}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_std.png"))
    plt.close()


def save_plot_kde(real_data, gen_data, var, title, unit):
    plt.figure()
    sns.kdeplot(real_data, fill=True, color=COLOR_REAL, alpha=0.3, label="Real")
    sns.kdeplot(gen_data, fill=True, color=COLOR_SIM, alpha=0.3, label="Sim")

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

    plt.plot(x_r, y_r, color=COLOR_REAL, label="Real")
    plt.plot(x_g, y_g, color=COLOR_SIM, ls="--", label="Sim")

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
    cols = ['co2', 'co', 'pm10', 'pm25']
    corr = df[cols].dropna(subset=cols).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Correlation Matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"CorrMatrix_{name}.png"))
    plt.close()


def save_plot_hourly_boxplot(df_sim, df_real, var, title, unit):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sns.boxplot(x="hour", y=var, data=df_real, ax=ax1, color=COLOR_REAL, showfliers=False)
    ax1.set_title("Real Data")

    sns.boxplot(x="hour", y=var, data=df_sim, ax=ax2, color=COLOR_SIM, showfliers=False)
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

    mean_col = f'gen_{var}_mean'
    if trend_col in df_comp.columns:
        metrics['bias'] = (df_comp[mean_col] - df_comp[trend_col]).abs().mean()
    else:
        metrics['bias'] = np.nan

    metrics['std_error'] = (
        df_comp[f'gen_{var}_std'] - df_comp[f'{var}_std']
    ).abs().mean()

    ks_stat, _ = stats.ks_2samp(real_series, gen_series)
    metrics['ks_stat'] = ks_stat

    cov_real = np.std(real_series) / np.mean(real_series)
    cov_sim = np.std(gen_series) / np.mean(gen_series)
    metrics['cov_real'] = cov_real
    metrics['cov_gen'] = cov_sim

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
        print("Missing input files.")
        return

    save_correlation_heatmap(df_real, "RealData")
    save_correlation_heatmap(df_sim, "SimulatedData")

    vars_config = [
        ('co2', 'CO2 Concentration', '(ppm)'),
        ('co', 'CO Concentration', '(µg/m³)'),
        ('pm10', 'PM10 Level', '(µg/m³)'),
        ('pm25', 'PM2.5 Level', '(µg/m³)')
    ]

    df_sim_hourly = df_sim.groupby('hour').agg(
        gen_co2_mean=('co2', 'mean'), gen_co2_std=('co2', 'std'),
        gen_co_mean=('co', 'mean'), gen_co_std=('co', 'std'),
        gen_pm10_mean=('pm10', 'mean'), gen_pm10_std=('pm10', 'std'),
        gen_pm25_mean=('pm25', 'mean'), gen_pm25_std=('pm25', 'std'),
    ).reset_index()

    df_comp = pd.merge(df_sim_hourly, df_trend, on="hour", how="left")
    df_comp = pd.merge(df_comp, df_real_hourly, on="hour", how="left")

    t_val = stats.t.ppf(0.975, N_SIMULATIONS - 1)

    report = []
    report.append("AIR QUALITY EVALUATION REPORT\n")

    for var, title, unit in vars_config:

        real_series = df_real[var].dropna().values
        gen_series = df_sim[var].dropna().values

        ci_error = t_val * (df_comp[f'gen_{var}_std'] / np.sqrt(N_SIMULATIONS))

        save_plot_mean(df_comp, var, title, unit, ci_error)
        save_plot_std(df_comp, var, title, unit)
        save_plot_kde(real_series, gen_series, var, title, unit)
        save_plot_ecdf(real_series, gen_series, var, title, unit)

        gen_sample = np.random.choice(gen_series, size=min(len(gen_series), 2000), replace=False)
        save_plot_qq(gen_sample, var, title)
        save_plot_hourly_boxplot(df_sim, df_real, var, title, unit)

        metrics = calculate_metrics(df_comp, real_series, gen_series, var, f'{var}_Trend')

        report.append(f"[{title.upper()}]")
        report.append(f"MEAN BIAS: {metrics['bias']:.4f} {unit}")
        report.append(f"STD ERROR: {metrics['std_error']:.4f} {unit}")
        report.append(f"COV (Real / Sim): {metrics['cov_real']:.3f} / {metrics['cov_gen']:.3f}")
        report.append(f"KS STAT: {metrics['ks_stat']:.4f}")
        report.append("PERCENTILES (R | S):")
        report.append(f"  P05: {metrics['p_real'][0]:.2f} | {metrics['p_gen'][0]:.2f}")
        report.append(f"  P50: {metrics['p_real'][1]:.2f} | {metrics['p_gen'][1]:.2f}")
        report.append(f"  P95: {metrics['p_real'][2]:.2f} | {metrics['p_gen'][2]:.2f}")
        report.append(f"  P99: {metrics['p_real'][3]:.2f} | {metrics['p_gen'][3]:.2f}")
        report.append("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Air quality evaluation completed. Results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
