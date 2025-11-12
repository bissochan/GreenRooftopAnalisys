import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

# --- Roof material properties ---
alpha_cement = 0.7
alpha_green = 0.8
Q_evap_green = 100.0   # Evaporative flux (W/m^2) only for green roof
C_cement = 176000.0    # Heat capacity (J/m^2K)
C_green = 270000.0     # Heat capacity (J/m^2K)

# --- Geometry and flow parameters ---
W = 20.0               # Roof width (m)
L = 30.0               # Length along wind direction (m)
D = 30.0               # Distance roof -> facade (m)
A_eff = W * L          # Effective roof area (m^2)
h_floor = 3.0          # Height per building floor (m)

rho = 1.2              # Air density (kg/m^3)
cp = 1005.0            # Air heat capacity (J/kg*K)
wind_eps = 0.05        # Minimal wind speed (m/s)

# --- Mixing and buoyancy parameters ---
Kz = 0.3               # Eddy diffusivity constant
H_min = 0.2
H_max = 6.0
g = 9.81               # Gravity (m/s^2)
T0_K = 273.15          # Reference (K)
gamma_damping = 0.45   # Buoyancy damping factor

# --- Numerical parameters ---
k = 0.7                # Sensor coupling factor
dt = 3600.0            # Time step (s)
n_sub = 60             # Sub-steps per hour
dt_sub = dt / n_sub
h_c_max = 100.0        # Max convective coefficient
max_step_roof = 30.0   # Max ΔT per step (°C)
SPIN_CYCLES = 2

# --- Initial conditions ---
T_cement_initial = 11.0
T_green_initial = 11.0

# --- Input path ---
CSV_PATH = "data/csv/hourly_trend.csv"

# ============================================================
# Load CSV input
# ============================================================

df = pd.read_csv(CSV_PATH)
radiation = df["Rad_Trend"].values
T_env_series = df["Temp_Trend"].values
wind_series = df["Wind_Trend"].values

# ============================================================
# Helper functions
# ============================================================

def compute_mixing_height_kz(wind, L_val=L, Kz_val=Kz, Hmin=H_min, Hmax=H_max):
    """Compute dynamic mixing height from wind and eddy diffusivity."""
    tau = L_val / max(wind, wind_eps)
    H = np.sqrt(Kz_val * tau)
    return float(np.clip(H, Hmin, Hmax))


def compute_air_over_roof(T_surface, T_env, wind, H_mix):
    """Compute air temperature above the roof using convective balance."""
    wind_use = max(wind, wind_eps)
    h_c = min(10.0 + 5.0 * wind_use, h_c_max)
    A_cross = W * H_mix
    m_dot = rho * wind_use * A_cross
    deltaT = (h_c * (A_eff * 2.0) / (m_dot * cp)) * (T_surface - T_env)
    return T_env + deltaT


def update_roof_euler(T_surface, T_air_local, alpha, G, Q_evap, C, wind, dt_local):
    """Update roof surface temperature via energy balance (explicit Euler)."""
    h_c = min(10.0 + 5.0 * wind, h_c_max)
    Q_net = alpha * G - Q_evap - h_c * (T_surface - T_air_local)
    return T_surface + (dt_local / C) * Q_net


def postprocess_facade_deltaT(T_air_end_cement, T_air_end_green, T_env, wind, H_mix):
    """Compute air temperature at facade, buoyant rise height, and affected floors."""
    tau = D / np.maximum(wind, wind_eps)

    # Relaxation factor (set to 0 -> direct propagation)
    relax_factor = 0.0

    T_air_local_cement = T_air_end_cement * (1 - relax_factor) + T_env * relax_factor
    T_air_local_green = T_air_end_green * (1 - relax_factor) + T_env * relax_factor

    # Buoyant rise estimation
    a_z_c = g * np.maximum(T_air_local_cement - T_env, 0.0) / (T_env + T0_K)
    a_z_g = g * np.maximum(T_air_local_green - T_env, 0.0) / (T_env + T0_K)

    rise_ideal_c = 0.5 * a_z_c * (tau ** 2)
    rise_ideal_g = 0.5 * a_z_g * (tau ** 2)

    Rise_cement = gamma_damping * rise_ideal_c + H_mix
    Rise_green = gamma_damping * rise_ideal_g + H_mix

    Rise_cement = np.maximum(Rise_cement, H_mix * 0.1)
    Rise_green = np.maximum(Rise_green, H_mix * 0.1)

    Floors_affected_cement = np.ceil(Rise_cement / h_floor)
    Floors_affected_green = np.ceil(Rise_green / h_floor)

    return T_air_local_cement, T_air_local_green, Rise_cement, Rise_green, Floors_affected_cement, Floors_affected_green


# ============================================================
# Core simulation
# ============================================================

def run_simulation_singlepass(T_c_init, T_g_init, radiation_arr, T_env_arr, wind_arr):
    """Run full simulation for concrete and green roof with steady convection."""
    N = len(radiation_arr)
    T_c_arr = np.zeros(N)
    T_g_arr = np.zeros(N)
    sensor_c_arr = np.zeros(N)
    sensor_g_arr = np.zeros(N)
    H_mix_arr = np.zeros(N)
    T_air_end_c_arr = np.zeros(N)
    T_air_end_g_arr = np.zeros(N)

    T_c_arr[0] = T_c_init
    T_g_arr[0] = T_g_init
    sensor_c_arr[0] = T_env_arr[0] + k * (T_c_init - T_env_arr[0])
    sensor_g_arr[0] = T_env_arr[0] + k * (T_g_init - T_env_arr[0])

    for t in range(1, N):
        T_env = T_env_arr[t]
        wind = max(wind_arr[t], wind_eps)
        G = radiation_arr[t]

        H_mix = compute_mixing_height_kz(wind)
        H_mix_arr[t] = H_mix

        T_c = T_c_arr[t-1]
        T_g = T_g_arr[t-1]

        for _ in range(n_sub):
            T_air_c = compute_air_over_roof(T_c, T_env, wind, H_mix)
            T_air_g = compute_air_over_roof(T_g, T_env, wind, H_mix)

            T_c_new = update_roof_euler(T_c, T_air_c, alpha_cement, G, 0.0, C_cement, wind, dt_sub)
            T_g_new = update_roof_euler(T_g, T_air_g, alpha_green, G, Q_evap_green, C_green, wind, dt_sub)

            T_c += np.clip(T_c_new - T_c, -max_step_roof, max_step_roof)
            T_g += np.clip(T_g_new - T_g, -max_step_roof, max_step_roof)

        T_c_arr[t] = T_c
        T_g_arr[t] = T_g
        sensor_c_arr[t] = T_env + k * (T_c - T_env)
        sensor_g_arr[t] = T_env + k * (T_g - T_env)
        T_air_end_c_arr[t] = compute_air_over_roof(T_c, T_env, wind, H_mix)
        T_air_end_g_arr[t] = compute_air_over_roof(T_g, T_env, wind, H_mix)

    H_mix_arr[0] = compute_mixing_height_kz(wind_arr[0])
    T_air_end_c_arr[0] = compute_air_over_roof(T_c_init, T_env_arr[0], wind_arr[0], H_mix_arr[0])
    T_air_end_g_arr[0] = compute_air_over_roof(T_g_init, T_env_arr[0], wind_arr[0], H_mix_arr[0])
    return T_c_arr, T_g_arr, H_mix_arr, sensor_c_arr, sensor_g_arr, T_air_end_c_arr, T_air_end_g_arr


# ============================================================
# Spin-up
# ============================================================

def spin_up_procedure(spin_cycles=SPIN_CYCLES):
    """Spin-up loop to reach thermal equilibrium."""
    T_c_guess, T_g_guess = T_env_series[0], T_env_series[0]
    for _ in range(spin_cycles):
        T_c_arr, T_g_arr, *_ = run_simulation_singlepass(
            T_c_guess, T_g_guess, radiation, T_env_series, wind_series
        )
        T_c_guess, T_g_guess = T_c_arr[-1], T_g_arr[-1]
    return T_c_guess, T_g_guess


# ============================================================
# Run simulation
# ============================================================

T_c0, T_g0 = spin_up_procedure()

T_cement_arr, T_green_arr, H_mix_arr, sensor_c_arr, sensor_g_arr, T_air_end_cement, T_air_end_green = \
    run_simulation_singlepass(T_c0, T_g0, radiation, T_env_series, wind_series)

T_air_local_cement, T_air_local_green, Rise_cement, Rise_green, Floors_affected_cement, Floors_affected_green = \
    postprocess_facade_deltaT(T_air_end_cement, T_air_end_green, T_env_series, wind_series, H_mix_arr)

# ============================================================
# Output
# ============================================================

df_out = df.copy()
df_out["T_cement"] = T_cement_arr
df_out["T_green"] = T_green_arr
df_out["Sensor_near_cement"] = sensor_c_arr
df_out["Sensor_near_green"] = sensor_g_arr
df_out["T_air_end_cement"] = T_air_end_cement
df_out["T_air_end_green"] = T_air_end_green
df_out["T_air_local_cement"] = T_air_local_cement
df_out["T_air_local_green"] = T_air_local_green
df_out["H_mix"] = H_mix_arr
df_out["Rise_cement"] = Rise_cement
df_out["Rise_green"] = Rise_green
df_out["Floors_affected_cement"] = Floors_affected_cement
df_out["Floors_affected_green"] = Floors_affected_green

out_csv = "roof_simulation_output_convective.csv"
df_out.to_csv(out_csv, index=False)
print(f"✅ Simulation completed. Output saved to {out_csv}")

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(sensor_c_arr, color='tomato', label="Sensor near Concrete Roof")
plt.plot(sensor_g_arr, color='seagreen', label="Sensor near Green Roof")
plt.plot(T_air_local_cement, '-.', color='orangered', label="Facade Air (Concrete)")
plt.plot(T_air_local_green, '-.', color='limegreen', label="Facade Air (Green)")
plt.plot(T_env_series, '--', color='gray', label="Ambient Air")
plt.plot(T_air_end_cement, ':', color='darkred', label="Air End of Roof (Concrete)")
plt.plot(T_air_end_green, ':', color='darkgreen', label="Air End of Roof (Green)")
plt.xlabel("Hour")
plt.ylabel("Temperature (°C)")
plt.title("Sensor and Air Temperature Evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sensor_facade_temperatures_convective.png")

plt.figure(figsize=(10, 5))
plt.plot(Rise_cement, color='tomato', label="Rise Concrete (m)")
plt.plot(Rise_green, color='seagreen', label="Rise Green (m)")
plt.plot(Floors_affected_cement, '--', color='orangered', label="Floors Affected Concrete")
plt.plot(Floors_affected_green, '--', color='limegreen', label="Floors Affected Green")
plt.xlabel("Hour")
plt.ylabel("Height (m) / Floors")
plt.title("Air Parcel Rise and Floors Affected")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rise_floors_convective.png")

plt.show()
