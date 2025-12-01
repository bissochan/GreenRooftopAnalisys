import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Global styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 100})

COLOR_REAL = 'seagreen'
COLOR_SIM = 'crimson'
COLOR_TREND = 'black'

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------


def load_data(config):
    # Load simulated data
    try:
        df_sim = pd.read_csv(config["files"]["sim"])
        if 'hour' not in df_sim.columns:
            df_sim['hour'] = df_sim.index % 24
    except FileNotFoundError:
        print(f"Missing file: {config['files']['sim']}")
        return None

    # Load trend data
    try:
        df_trend = pd.read_csv(config["files"]["trend"])
        trend_map = {f"{k}_Trend": f"{v}_Trend" for k, v in config["col_mapping"].items()}
        df_trend = df_trend.rename(columns=trend_map)

        # Special renames
        if config["name"] == "weather":
            df_trend = df_trend.rename(columns={
                'Temp_C_Trend': 'Temp_Trend',
                'Wind_ms_Trend': 'Wind_Trend',
                'Radiation_Wm2_Trend': 'Rad_Trend'
            })

        if 'hour' not in df_trend.columns:
            df_trend['hour'] = df_trend.index
    except FileNotFoundError:
        print(f"Missing file: {config['files']['trend']}")
        return None

    # Load real data
    try:
        skip = 2 if config["name"] != "generic" else 0
        df_real = pd.read_csv(config["files"]["real"], skiprows=skip)

        df_real = df_real.rename(columns=config["col_mapping"])

        # Parse timestamps
        if config.get("time_unit") == 's':
            df_real['timestamp'] = pd.to_datetime(df_real['timestamp'], unit='s')
        else:
            df_real['timestamp'] = pd.to_datetime(df_real['timestamp'], errors='coerce')

        df_real['hour'] = df_real['timestamp'].dt.hour

        agg_dict = {v[0]: ['mean', 'std'] for v in config["vars"]}
        df_real_hourly = df_real.groupby('hour').agg(agg_dict).reset_index()
        df_real_hourly.columns = ['hour'] + [f"{var}_{stat}" for (var, stat) in agg_dict.items() for stat in ['mean', 'std']]
        df_real_hourly = df_real_hourly.rename(columns={k: f"{k}_std" for k in agg_dict})
    except FileNotFoundError:
        print(f"Missing file: {config['files']['real']}")
        return None

    return df_sim, df_trend, df_real, df_real_hourly


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

def plot_mean(df_comp, var, title, unit, ci_error, output_dir):
    plt.figure()

    trend_key = f"{var}_Trend"
    mean_gen_col = f"gen_{var}_mean"
    mean_real_col = f"{var}_mean"

    if trend_key in df_comp.columns:
        plt.plot(df_comp["hour"], df_comp[trend_key], color=COLOR_TREND, ls='--')

    if mean_real_col in df_comp.columns:
        plt.plot(
            df_comp["hour"],
            df_comp[mean_real_col],
            color=COLOR_TREND,
            ls=':',
            linewidth=2
        )

    plt.plot(df_comp["hour"], df_comp[mean_gen_col], color=COLOR_SIM)
    plt.fill_between(
        df_comp["hour"],
        df_comp[mean_gen_col] - ci_error,
        df_comp[mean_gen_col] + ci_error,
        alpha=0.2,
        color=COLOR_SIM
    )

    plt.title(f"{title}: Mean Hourly Profile")
    plt.legend(["Trend", "Real Mean", "Simulated Mean", "95% CI"])
    plt.ylabel(f"{title} {unit}")
    plt.xlabel("Hour")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_mean_ci.png"))
    plt.close()


def plot_std(df_comp, var, title, unit, output_dir):
    plt.figure()
    plt.plot(df_comp["hour"], df_comp[f"{var}_std"], color=COLOR_REAL)
    plt.plot(df_comp["hour"], df_comp[f"gen_{var}_std"], color=COLOR_SIM, ls="--")
    plt.title(f"{title}: Hourly Std Dev")
    plt.ylabel(f"Std {unit}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_std.png"))
    plt.close()


def plot_hist_ecdf_qq(real_vals, gen_vals, var, title, unit, output_dir):
    # Histogram
    plt.figure()
    sns.histplot(real_vals, color=COLOR_REAL, stat='density', bins='auto', alpha=0.5)
    sns.histplot(gen_vals, color=COLOR_SIM, stat='density', bins='auto', alpha=0.5)
    plt.title(f"{title}: Histogram")
    plt.xlabel(f"{title} {unit}")
    plt.legend(["Real", "Simulated"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_hist.png"))
    plt.close()

    # ECDF
    plt.figure()
    x_r, y_r = np.sort(real_vals), np.arange(1, len(real_vals) + 1) / len(real_vals)
    x_g, y_g = np.sort(gen_vals), np.arange(1, len(gen_vals) + 1) / len(gen_vals)
    plt.plot(x_r, y_r, color=COLOR_REAL)
    plt.plot(x_g, y_g, color=COLOR_SIM, ls="--")
    plt.title(f"{title}: ECDF")
    plt.legend(["Real", "Simulated"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_ecdf.png"))
    plt.close()

    # QQ Plot
    plt.figure()
    gen_sample = np.random.choice(gen_vals, size=min(len(gen_vals), 2000), replace=False)
    stats.probplot(gen_sample, dist="norm", plot=plt)
    plt.title(f"{title}: QQ Plot")
    plt.legend(["Simulated"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_qq.png"))
    plt.close()


def plot_correlation(df, prefix, output_dir):
    plt.figure(figsize=(6, 5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix: {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CorrMatrix_{prefix}.png"))
    plt.close()


def plot_hourly_boxplot(df_sim, df_real, var, title, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    sns.boxplot(x="hour", y=var, data=df_real, ax=ax1, color=COLOR_REAL, showfliers=False)
    ax1.set_title("Real Data")
    sns.boxplot(x="hour", y=var, data=df_sim, ax=ax2, color=COLOR_SIM, showfliers=False)
    ax2.set_title("Simulated Data")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{var}_hourly_box.png"))
    plt.close()


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(df_comp, real_vals, gen_vals, var):
    metrics = {}

    trend_col = f"{var}_Trend"
    mean_col = f"gen_{var}_mean"

    if trend_col in df_comp.columns:
        metrics["bias"] = (df_comp[mean_col] - df_comp[trend_col]).abs().mean()
    else:
        metrics["bias"] = np.nan

    metrics["std_error"] = (df_comp[f"gen_{var}_std"] - df_comp[f"{var}_std"]).abs().mean()
    metrics["ks_stat"], _ = stats.ks_2samp(real_vals, gen_vals)

    mu_r, mu_g = real_vals.mean(), gen_vals.mean()
    metrics["cov_real"] = real_vals.std() / mu_r if mu_r != 0 else 0
    metrics["cov_gen"] = gen_vals.std() / mu_g if mu_g != 0 else 0

    metrics["p_real"] = np.percentile(real_vals, [5, 50, 95, 99])
    metrics["p_gen"] = np.percentile(gen_vals, [5, 50, 95, 99])

    return metrics


# ---------------------------------------------------------
# EVALUATION RUNNER
# ---------------------------------------------------------

def run_evaluation(config):
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Running evaluation: {config['name'].upper()} ---")

    loaded = load_data(config)
    if loaded is None:
        print("Skipping due to missing files.")
        return

    df_sim, df_trend, df_real, df_real_hourly = loaded

    plot_correlation(df_real, "Real", output_dir)
    plot_correlation(df_sim, "Simulated", output_dir)

    # Build aggregated simulation dataframe
    agg_def = {}
    for var, _, _ in config["vars"]:
        agg_def[f"gen_{var}_mean"] = (var, "mean")
        agg_def[f"gen_{var}_std"] = (var, "std")

    df_sim_hourly = df_sim.groupby("hour").agg(**agg_def).reset_index()

    # Merge dataframes
    df_comp = pd.merge(df_sim_hourly, df_trend, on="hour", how="left")
    df_comp = pd.merge(df_comp, df_real_hourly, on="hour", how="left")

    n_runs = df_sim['run_id'].nunique()
    t_val = stats.t.ppf(0.975, n_runs - 1)

    report_lines = [f"{config['name'].upper()} EVALUATION REPORT\n"]

    for var, title, unit in config["vars"]:
        real_vals = df_real[var].dropna().values
        gen_vals = df_sim[var].dropna().values

        ci_error = t_val * (df_comp[f"gen_{var}_std"] / np.sqrt(n_runs))

        plot_mean(df_comp, var, title, unit, ci_error, output_dir)
        plot_std(df_comp, var, title, unit, output_dir)
        plot_hist_ecdf_qq(real_vals, gen_vals, var, title, unit, output_dir)
        plot_hourly_boxplot(df_sim, df_real, var, title, output_dir)

        m = calculate_metrics(df_comp, real_vals, gen_vals, var)

        report_lines += [
            f"[{title}]",
            f"MEAN BIAS: {m['bias']:.4f} {unit}",
            f"STD ERROR: {m['std_error']:.4f} {unit}",
            f"COV (Real / Sim): {m['cov_real']:.3f} / {m['cov_gen']:.3f}",
            f"KS STAT: {m['ks_stat']:.4f}",
            "PERCENTILES (R | S):",
            f"  P05: {m['p_real'][0]:.2f} | {m['p_gen'][0]:.2f}",
            f"  P50: {m['p_real'][1]:.2f} | {m['p_gen'][1]:.2f}",
            f"  P95: {m['p_real'][2]:.2f} | {m['p_gen'][2]:.2f}",
            f"  P99: {m['p_real'][3]:.2f} | {m['p_gen'][3]:.2f}",
            ""
        ]

    report_path = os.path.join(output_dir, f"{config['name']}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Completed. Output in: {output_dir}/")


# ---------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------

DATA_DIR = os.path.join("..", "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
SIM_DIR = os.path.join("..", "simulation")

AIR_CONFIG = {
    "name": "air_quality",
    "files": {
        "sim": os.path.join(SIM_DIR, "csv_results", "air_quality_samples.csv"),
        "trend": os.path.join(CSV_DIR, "air_quality_hourly_trend.csv"),
        "real": os.path.join(DATA_DIR, "air_quality.csv")
    },
    "col_mapping": {
        'time': 'timestamp',
        'carbon_dioxide (ppm)': 'co2',
        'pm10 (μg/m³)': 'pm10',
        'pm2_5 (μg/m³)': 'pm25',
        'carbon_monoxide (μg/m³)': 'co'
    },
    "vars": [
        ('co2', 'CO2 Concentration', '(ppm)'),
        ('co', 'CO Concentration', '(µg/m³)'),
        ('pm10', 'PM10 Level', '(µg/m³)'),
        ('pm25', 'PM2.5 Level', '(µg/m³)')
    ],
    "output_dir": os.path.join("results", "air_quality")
}

WEATHER_CONFIG = {
    "name": "weather",
    "files": {
        "sim": os.path.join(SIM_DIR, "csv_results", "weather_samples.csv"),
        "trend": os.path.join(CSV_DIR, "meteo_hourly_trend.csv"),
        "real": os.path.join(DATA_DIR, "open-meteo.csv")
    },
    "col_mapping": {
        'time': 'timestamp',
        'temperature_2m (°C)': 'temp',
        'wind_speed_10m (m/s)': 'wind',
        'shortwave_radiation (W/m²)': 'rad'
    },
    "vars": [
        ('temp', 'Temperature', '(°C)'),
        ('wind', 'Wind Speed', '(m/s)'),
        ('rad', 'Solar Radiation', '(W/m²)')
    ],
    "output_dir": os.path.join("results", "weather"),
    "time_unit": 's'
}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    run_evaluation(AIR_CONFIG)
    print("\n-----------------------\n")
    run_evaluation(WEATHER_CONFIG)


if __name__ == "__main__":
    main()
