import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from math import pi

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (8, 6), 'figure.dpi': 100})

RESULT_DIR = os.path.join("..", "simulation", "csv_results")
OUTPUT_DIR = os.path.join("results", "simulation_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Variables
# -------------------------------
temp_vars = [
    "T_cement", "T_green",
    "T_air_cement", "T_air_green",
    "T_air_fac_cement", "T_air_fac_green"
]

building_vars = [
    "H_mix", "Rise_cement", "Rise_green",
    "Floors_cement", "Floors_green"
]

air_quality_vars = [
    "CO2_Removed_g", "CO_Removed_ug",
    "PM10_Removed_ug", "PM25_Removed_ug",
    "O2_Produced_g"
]

all_vars = temp_vars + building_vars

# ===============================
# FUNCTIONS
# ===============================


def range_check(df, all_vars, ranges):
    print("=== RANGE CHECK ===")
    for var in all_vars:
        min_val, max_val = df[var].min(), df[var].max()
        low_limit, high_limit = ranges.get(var, (None, None))
        print(f"{var}: min={min_val:.2f}, max={max_val:.2f}")
        if low_limit is not None and (min_val < low_limit or max_val > high_limit):
            print(f"  WARNING: values outside expected range ({low_limit} - {high_limit})")


def spike_check(df, all_vars):
    print("\n=== SPIKE CHECK (Δ hour-to-hour) ===")
    for var in all_vars:
        df['delta'] = df.groupby('run_id')[var].diff().abs()
        max_delta = df['delta'].max()
        print(f"{var}: max delta = {max_delta:.2f}")
        df.drop(columns='delta', inplace=True)


def plot_hourly_boxplots(df, all_vars, output_dir):
    for var in all_vars:
        plt.figure()
        sns.boxplot(x='hour', y=var, data=df, color='crimson', showfliers=True)
        if "Rise" in var or "H_mix" in var:
            plt.ylim(0, 8)
            plt.yticks(np.arange(0, 9, 1))
            ylabel = f"{var} [m]"
        elif "Floors" in var:
            plt.ylim(0.8, 3.2)
            plt.yticks(np.arange(1, 4, 1))
            ylabel = f"{var} [#]"
        elif "T_air_fac" in var:
            plt.ylim(-10, 45)
            ylabel = f"{var} [°C]"
        elif "T_air" in var:
            plt.ylim(-10, 50)
            ylabel = f"{var} [°C]"
        elif "T_" in var:
            plt.ylim(-10, 80)
            plt.yticks(np.arange(-10, 81, 10))
            ylabel = f"{var} [°C]"
        else:
            ylabel = var
        plt.title(f"{var} - Hourly distribution")
        plt.xlabel("Hour")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{var}_hourly_box.png"))
        plt.close()


def plot_floors_histogram(df, output_dir):
    for var in ["Floors_cement", "Floors_green"]:
        plt.figure()
        sns.countplot(x='hour', hue=var, data=df, palette='Set2')
        plt.ylabel("Count")
        plt.xlabel("Hour")
        plt.legend(title=var)
        plt.title(f"Histogram of {var} over Hours")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{var}_histogram.png"))
        plt.close()


def plot_air_quality_time(df, output_dir):
    factors = air_quality_vars
    n_runs = df['run_id'].nunique()
    t_val = stats.t.ppf(0.975, n_runs - 1)  # 95% CI

    for factor in factors:
        if factor not in df.columns:
            continue

        stats_df = df.groupby("hour")[factor].agg(["mean", "std"])
        ci_error = t_val * stats_df["std"] / np.sqrt(n_runs)

        plt.figure()
        plt.plot(stats_df.index, stats_df["mean"], color='blue', label='Mean Performance')
        plt.fill_between(stats_df.index, stats_df["mean"] - ci_error,
                         stats_df["mean"] + ci_error, color='blue', alpha=0.2, label='95% CI')
        plt.title(f"{factor} - Hourly Profile with 95% CI")
        plt.xlabel("Hour of Day")
        plt.ylabel(factor)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{factor}_time_series_CI.png"))
        plt.close()


def plot_air_quality_box(df, output_dir):
    factors = air_quality_vars
    for factor in factors:
        if factor not in df.columns:
            continue
        plt.figure()
        sns.boxplot(x="hour", y=factor, data=df, color='crimson', showfliers=True)
        plt.title(f"{factor} Distribution (Boxplot)")
        plt.xlabel("Hour")
        plt.ylabel(factor)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{factor}_boxplot.png"))
        plt.close()


def plot_correlation_matrix(df, all_vars, output_dir):
    plt.figure(figsize=(10, 8))
    corr = df[all_vars].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix - Simulation Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()


def plot_temp_ci(df, output_dir):
    for base in ["T", "T_air", "T_air_fac"]:
        plt.figure()

        for scenario, color in [("cement", "red"), ("green", "green")]:
            col = f"{base}_{scenario}"
            stats_df = df.groupby("hour")[col].agg(["mean", "count", "std"])

            ci_hi, ci_lo = [], []

            for _, row in stats_df.iterrows():
                m, c, s = row["mean"], row["count"], row["std"]
                t_val = stats.t.ppf(0.975, c - 1)
                margin = t_val * (s / np.sqrt(c))
                ci_hi.append(m + margin)
                ci_lo.append(m - margin)

            plt.plot(stats_df.index, stats_df["mean"], color=color, label=f"{scenario.capitalize()} Mean")
            plt.fill_between(stats_df.index, ci_lo, ci_hi, color=color, alpha=0.2)

        plt.title("Temperature Profile with 95% Confidence Intervals")
        plt.xlabel("Hour")
        plt.ylabel("Temperature [°C]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_confidence_intervals.png"))
        plt.close()


def plot_ecdf_comparison(df, output_dir):
    for base in ["T", "T_air", "T_air_fac"]:
        plt.figure()

        var_cement = f"{base}_cement"
        var_green = f"{base}_green"

        x_c = np.sort(df[var_cement])
        x_g = np.sort(df[var_green])
        y_c = np.arange(1, len(x_c)+1) / len(x_c)
        y_g = np.arange(1, len(x_g)+1) / len(x_g)

        plt.plot(x_c, y_c, '.', color='red', alpha=0.5, label="Cement")
        plt.plot(x_g, y_g, '.', color='green', alpha=0.5, label="Green")

        plt.title("ECDF Comparison")
        plt.xlabel("Value")
        plt.ylabel("Proportion")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ecdf_comparison_{base}.png"))
        plt.close()


def plot_convergence(df, var, output_dir):
    values = df[var].values
    running_mean = np.cumsum(values) / np.arange(1, len(values) + 1)

    plt.figure()
    plt.plot(running_mean, color='purple')
    plt.title(f"Convergence Check: {var}")
    plt.xlabel("Sample Number")
    plt.ylabel("Cumulative Mean")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"convergence_{var}.png"))
    plt.close()

def plot_diff_cement_green(df, output_dir):
    bases = ["T", "T_air", "T_air_fac", "Rise", "Floors"]
    n_runs = df['run_id'].nunique()
    t_val = stats.t.ppf(0.975, n_runs - 1)

    for base in bases:
        col_c = f"{base}_cement"
        col_g = f"{base}_green"
        if col_c not in df.columns or col_g not in df.columns:
            continue

        hourly_data = []
        for hour in range(24):
            hour_df = df[df['hour'] == hour]
            diff_series = hour_df[col_c] - hour_df[col_g]

            mean_diff = diff_series.mean()
            std_diff = diff_series.std()
            ci_margin = t_val * (std_diff / np.sqrt(n_runs))

            hourly_data.append({
                "hour": hour,
                "delta": mean_diff,
                "ci": ci_margin
            })

        df_plot = pd.DataFrame(hourly_data)

        plt.figure()
        bars = plt.bar(df_plot["hour"], df_plot["delta"], color="purple", alpha=0.6, label="Mean Δ")
        plt.errorbar(df_plot["hour"], df_plot["delta"], yerr=df_plot["ci"], fmt='none',
                     ecolor='black', capsize=3, elinewidth=1, label="95% CI")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"Difference Cement - Green ({base}) with 95% CI")
        plt.xlabel("Hour")

        if "T" in base:
            plt.ylabel("Δ [°C]")
        elif "Rise" in base:
            plt.ylabel("Δ [m]")
        elif "Floors" in base:
            plt.ylabel("Δ [# floors]")

        plt.xticks(range(24))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_delta_with_CI.png"))
        plt.close()


# ===============================
# MAIN
# ===============================

def main():
    file_path = os.path.join(RESULT_DIR, "simulation_results.csv")
    df = pd.read_csv(file_path)

    ranges = {
        "T_cement": (-20, 55), "T_green": (-20, 55),
        "T_air_cement": (-20, 55), "T_air_green": (-20, 55),
        "T_air_fac_cement": (-20, 55), "T_air_fac_green": (-20, 55),
        "H_mix": (0, 10), "Rise_cement": (0, 10), "Rise_green": (0, 10),
        "Floors_cement": (0, 5), "Floors_green": (0, 5)
    }

    range_check(df, all_vars, ranges)
    spike_check(df, all_vars)
    plot_air_quality_time(df, OUTPUT_DIR)
    plot_air_quality_box(df, OUTPUT_DIR)
    plot_temp_ci(df, OUTPUT_DIR)
    plot_hourly_boxplots(df, all_vars, OUTPUT_DIR)
    plot_correlation_matrix(df, all_vars, OUTPUT_DIR)

    plot_ecdf_comparison(df, OUTPUT_DIR)
    plot_convergence(df, "T_green", OUTPUT_DIR)
    plot_convergence(df, "T_cement", OUTPUT_DIR)
    plot_floors_histogram(df, OUTPUT_DIR)

    plot_diff_cement_green(df, OUTPUT_DIR)

    print("\nEvaluation complete. Plots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
