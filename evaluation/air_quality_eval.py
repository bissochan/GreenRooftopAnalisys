import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = os.path.join("..", "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")
SIM_DIR = os.path.join("..", "simulation")

WEATHER_SAMPLES_FILE = os.path.join(SIM_DIR, "eco_results", "weather_samples_eco.csv")
SIM_RESULTS_FILE = os.path.join(SIM_DIR, "eco_results", "simulation_results_eco.csv")
TREND_FILE = os.path.join(CSV_DIR, "air_quality_hourly_trend.csv")
REAL_DATA_FILE = os.path.join(DATA_DIR, "air_quality.csv")

OUTPUT_DIR = "results/air_quality"
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "air_quality_evaluation_report.txt")
N_SIMULATIONS = 100


def _read_real_airfile(path):
    # file contains two initial metadata lines before the real header that starts with "time,"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = None
    for i, L in enumerate(lines[:20]):
        if L.strip().lower().startswith("time,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate header line starting with 'time,' in air_quality.csv")
    df = pd.read_csv(path, skiprows=header_idx)
    return df


def load_data():
    try:
        if not os.path.exists(WEATHER_SAMPLES_FILE) or not os.path.exists(SIM_RESULTS_FILE) or not os.path.exists(TREND_FILE):
            raise FileNotFoundError("One or more required simulation/trend files are missing.")

        df_weather = pd.read_csv(WEATHER_SAMPLES_FILE)
        if "hour" not in df_weather.columns:
            df_weather["hour"] = df_weather.index % 24

        df_sim = pd.read_csv(SIM_RESULTS_FILE)
        if "hour" not in df_sim.columns:
            df_sim["hour"] = df_sim.index % 24

        df_trend = pd.read_csv(TREND_FILE)
        # ensure hour column exists in trend
        if "hour" not in df_trend.columns:
            df_trend["hour"] = df_trend.index
        # no rename necessary if columns are already CO2_Trend, PM10_Trend, PM2_5_Trend, CO_Trend

        # read real air file (robust to initial metadata lines)
        df_real = _read_real_airfile(REAL_DATA_FILE)

        # rename columns present in the provided sample
        rename_map = {}
        if "time" in df_real.columns:
            rename_map["time"] = "timestamp"
        # possible column names from file sample
        if "carbon_dioxide (ppm)" in df_real.columns:
            rename_map["carbon_dioxide (ppm)"] = "co2_real"
        if "pm10 (μg/m³)" in df_real.columns:
            rename_map["pm10 (μg/m³)"] = "pm10_real"
        if "pm2_5 (μg/m³)" in df_real.columns:
            rename_map["pm2_5 (μg/m³)"] = "pm25_real"
        if "carbon_monoxide (μg/m³)" in df_real.columns:
            rename_map["carbon_monoxide (μg/m³)"] = "co_real"

        df_real = df_real.rename(columns=rename_map)

        # parse timestamp and compute hour
        if "timestamp" in df_real.columns:
            df_real["timestamp"] = pd.to_datetime(df_real["timestamp"], errors="coerce")
            df_real["hour"] = df_real["timestamp"].dt.hour
        else:
            # fallback: create hour from order
            df_real["hour"] = np.arange(len(df_real)) % 24

        # aggregate hourly stats from real data (may contain NaNs)
        agg_map = {}
        if "co2_real" in df_real.columns:
            agg_map["real_co2_mean"] = ("co2_real", "mean")
            agg_map["real_co2_std"] = ("co2_real", "std")
        if "co_real" in df_real.columns:
            agg_map["real_co_mean"] = ("co_real", "mean")
            agg_map["real_co_std"] = ("co_real", "std")
        if "pm10_real" in df_real.columns:
            agg_map["real_pm10_mean"] = ("pm10_real", "mean")
            agg_map["real_pm10_std"] = ("pm10_real", "std")
        if "pm25_real" in df_real.columns:
            agg_map["real_pm25_mean"] = ("pm25_real", "mean")
            agg_map["real_pm25_std"] = ("pm25_real", "std")

        if agg_map:
            df_real_hourly = df_real.groupby("hour").agg(**agg_map).reset_index()
        else:
            df_real_hourly = pd.DataFrame({"hour": list(range(24))})

        return df_weather, df_sim, df_trend, df_real, df_real_hourly

    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None, None, None, None


def _ensure_outdir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot_mean(df_comp, var, title, unit, ci_error):
    plt.figure(figsize=(10, 5))
    trend_col = f"{var.upper()}_Trend"
    gen_mean_col = f"gen_{var}_mean"
    real_mean_col = f"real_{var}_mean"

    if trend_col in df_comp.columns:
        plt.plot(df_comp["hour"], df_comp[trend_col], label="Trend", color="k", linestyle="--")
    if real_mean_col in df_comp.columns:
        plt.plot(df_comp["hour"], df_comp[real_mean_col], label="Real mean", color="g")
    plt.plot(df_comp["hour"], df_comp[gen_mean_col], label="Simulated mean", color="r")
    plt.fill_between(df_comp["hour"],
                     df_comp[gen_mean_col] - ci_error,
                     df_comp[gen_mean_col] + ci_error,
                     color="r", alpha=0.2, label="95% CI")
    plt.title(f"{title} mean (hourly)")
    plt.xlabel("Hour")
    plt.ylabel(unit)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_mean.png"), dpi=150)
    plt.close()


def save_plot_std(df_comp, var, title, unit):
    plt.figure(figsize=(10, 5))
    real_std_col = f"real_{var}_std"
    gen_std_col = f"gen_{var}_std"
    if real_std_col in df_comp.columns:
        plt.plot(df_comp["hour"], df_comp[real_std_col], label="Real std", color="g")
    if gen_std_col in df_comp.columns:
        plt.plot(df_comp["hour"], df_comp[gen_std_col], label="Sim std", color="purple", linestyle="--")
    plt.title(f"{title} std dev (hourly)")
    plt.xlabel("Hour")
    plt.ylabel(unit)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_std.png"), dpi=150)
    plt.close()


def save_plot_ecdf(real_series, gen_series, var):
    if real_series is None or len(real_series) == 0:
        return
    plt.figure(figsize=(8, 5))
    xr = np.sort(real_series)
    yr = np.arange(1, len(xr) + 1) / len(xr)
    xg = np.sort(gen_series)
    yg = np.arange(1, len(xg) + 1) / len(xg)
    plt.plot(xr, yr, label="Real", color="g")
    plt.plot(xg, yg, label="Sim", color="purple", linestyle="--")
    plt.title(f"{var} ECDF")
    plt.xlabel(var)
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_ecdf.png"), dpi=150)
    plt.close()


def save_plot_qq(gen_series, var):
    if gen_series is None or len(gen_series) == 0:
        return
    plt.figure(figsize=(6, 6))
    stats.probplot(gen_series, dist="norm", plot=plt)
    plt.title(f"{var} Q-Q")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_qq.png"), dpi=150)
    plt.close()


def save_removal_plots(df_sim, var, is_co2=False):
    removal_col = f"{var.upper()}_Removed_{'g' if is_co2 else 'ug'}"
    if removal_col not in df_sim.columns:
        return
    # hourly mean ± std
    hourly = df_sim.groupby("hour")[removal_col].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(hourly["hour"], hourly["mean"], label="mean", color="b")
    plt.fill_between(hourly["hour"], hourly["mean"] - hourly["std"], hourly["mean"] + hourly["std"], alpha=0.2)
    plt.title(f"{var} removal hourly")
    plt.xlabel("Hour")
    plt.ylabel(removal_col)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_removal_hourly.png"), dpi=150)
    plt.close()

    # daily totals distribution
    daily = df_sim.groupby("run_id")[removal_col].sum()
    plt.figure(figsize=(8, 5))
    plt.hist(daily.dropna(), bins=20, color="steelblue", edgecolor="k")
    plt.axvline(daily.mean(), color="r", linestyle="--", label=f"mean {daily.mean():.2f}")
    plt.legend()
    plt.title(f"{var} daily totals distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_daily_totals.png"), dpi=150)
    plt.close()

    # cumulative mean
    mean_cumsum = df_sim.groupby("hour")[removal_col].sum().cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(mean_cumsum.index, mean_cumsum.values, "r-", linewidth=2)
    plt.title(f"{var} cumulative mean (24h)")
    plt.xlabel("Hour")
    plt.ylabel("Cumulative")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{var}_cumulative.png"), dpi=150)
    plt.close()

    return daily


def run_evaluation():
    _ensure_outdir()
    df_weather, df_sim, df_trend, df_real, df_real_hourly = load_data()
    if df_sim is None:
        print("ERROR: Could not load simulation data")
        return

    # hourly stats for simulated concentrations
    df_sim_hourly = df_sim.groupby("hour").agg(
        gen_co2_mean=("CO2", "mean"),
        gen_co2_std=("CO2", "std"),
        gen_co_mean=("CO", "mean"),
        gen_co_std=("CO", "std"),
        gen_pm10_mean=("PM10", "mean"),
        gen_pm10_std=("PM10", "std"),
        gen_pm25_mean=("PM25", "mean"),
        gen_pm25_std=("PM25", "std"),
    ).reset_index()

    df_comp = pd.merge(df_sim_hourly, df_trend, on="hour", how="left")
    df_comp = pd.merge(df_comp, df_real_hourly, on="hour", how="left")

    report_lines = []
    report_lines.append("AIR QUALITY EVALUATION")
    report_lines.append(f"Sim file: {SIM_RESULTS_FILE}")
    report_lines.append(f"Real data file: {REAL_DATA_FILE}")
    report_lines.append("")

    t_val = stats.t.ppf(0.975, max(2, N_SIMULATIONS - 1))

    variables = [
        ("co2", "CO2", "ppm", True),
        ("co", "CO", "µg/m³", False),
        ("pm10", "PM10", "µg/m³", False),
        ("pm25", "PM2.5", "µg/m³", False),
    ]

    for var, title, unit, is_co2 in variables:
        gen_mean_col = f"gen_{var}_mean"
        gen_std_col = f"gen_{var}_std"
        if gen_mean_col in df_comp.columns:
            ci_err = t_val * (df_comp[gen_std_col] / np.sqrt(N_SIMULATIONS)).fillna(0) if gen_std_col in df_comp.columns else np.zeros(len(df_comp))
            save_plot_mean(df_comp, var, title, unit, ci_err)
        if gen_std_col in df_comp.columns:
            save_plot_std(df_comp, var, title, unit)

        # ECDF and QQ using weather samples (generated concentrations)
        if var in df_weather.columns:
            gen_series = df_weather[var].dropna().values
            real_series = None
            col_real = f"{var}_real"
            if col_real in df_real.columns:
                real_series = df_real[col_real].dropna().values
            save_plot_ecdf(real_series if real_series is not None else np.array([]), gen_series, title)
            save_plot_qq(gen_series, title)

        # removal plots and daily totals
        daily_totals = save_removal_plots(df_sim, var, is_co2=is_co2)

        # report stats
        report_lines.append(f"\n[{title}]")
        # bias vs trend
        trend_col = f"{title.upper()}_Trend"
        bias = np.nan
        if trend_col in df_comp.columns and gen_mean_col in df_comp.columns:
            bias = np.nanmean(np.abs(df_comp[gen_mean_col] - df_comp[trend_col]))
        report_lines.append(f"Mean bias vs trend: {bias:.3f} {unit}")

        # compare percentiles
        if var in df_weather.columns:
            gen_ser = df_weather[var].dropna().values
            pgen = np.percentile(gen_ser, [5, 50, 95, 99]) if len(gen_ser) else [np.nan]*4
        else:
            pgen = [np.nan]*4
        if f"{var}_real" in df_real.columns:
            real_ser = df_real[f"{var}_real"].dropna().values
            preal = np.percentile(real_ser, [5, 50, 95, 99]) if len(real_ser) else [np.nan]*4
        else:
            preal = [np.nan]*4
        report_lines.append(f"Percentiles Real vs Sim (P5,P50,P95,P99): {preal} | {pgen}")

        # daily removal summary
        if daily_totals is not None:
            report_lines.append(f"Daily removal mean: {daily_totals.mean():.2f}, std: {daily_totals.std():.2f}")

    # save report and a small json summary
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    summary = {"note": "evaluation complete"}
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation complete. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_evaluation()