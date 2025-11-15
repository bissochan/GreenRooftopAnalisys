import pandas as pd
import matplotlib.pyplot as plt
from model import parameters as p
from model import roof as roof_model
from model import aerodynamics as aero_model
from model import facade as facade_model
from blocks.roof_block import RoofBlock
from blocks.aerodynamics_block import AerodynamicsBlock
from blocks.facade_block import FacadeBlock
from blocks.master import Master

# ============================================================
# Load input CSV
# ============================================================
CSV_PATH = "data/csv/hourly_trend.csv"
df = pd.read_csv(CSV_PATH)
radiation_series = df["Rad_Trend"].values
T_env_series = df["Temp_Trend"].values
wind_series = df["Wind_Trend"].values

# ============================================================
# Initialize blocks
# ============================================================
roof_cement_block = RoofBlock(
    T_init=T_env_series[0],
    alpha=p.alpha_cement,
    Q_evap=0.0,
    C=p.C_cement,
    roof_funcs=roof_model,
    params=p
)

roof_green_block = RoofBlock(
    T_init=T_env_series[0],
    alpha=p.alpha_green,
    Q_evap=p.Q_evap_green,
    C=p.C_green,
    roof_funcs=roof_model,
    params=p
)

aero_block = AerodynamicsBlock(aero_model)
facade_block = FacadeBlock(facade_model)

master = Master(roof_cement_block, roof_green_block, aero_block, facade_block, p)

# ============================================================
# Spin-up procedure
# ============================================================
print("🔄 Starting spin-up to reach thermal equilibrium...")
T_c_guess, T_g_guess = T_env_series[0], T_env_series[0]

for _ in range(p.SPIN_CYCLES):
    for t in range(len(T_env_series)):
        master.do_step(T_env_series[t], radiation_series[t], wind_series[t])
    T_c_guess = roof_cement_block.T
    T_g_guess = roof_green_block.T

roof_cement_block.T = T_c_guess
roof_green_block.T = T_g_guess
print(f"✅ Spin-up complete. Initial T_cement={T_c_guess:.2f}, T_green={T_g_guess:.2f}")

# ============================================================
# Run main simulation
# ============================================================
N = len(T_env_series)
T_c_arr = []
T_g_arr = []
H_mix_arr = []
sensor_c_arr = []
sensor_g_arr = []
T_air_end_c_arr = []
T_air_end_g_arr = []
facade_outputs = []

print("🚀 Running simulation...")
for t in range(N):
    T_env = T_env_series[t]
    G = radiation_series[t]
    wind = wind_series[t]

    T_c, T_g, H_mix, T_air_c, T_air_g, facade_out = master.do_step(T_env, G, wind)

    # Store outputs
    T_c_arr.append(T_c)
    T_g_arr.append(T_g)
    H_mix_arr.append(H_mix)
    T_air_end_c_arr.append(T_air_c)
    T_air_end_g_arr.append(T_air_g)
    sensor_c_arr.append(T_env + p.k * (T_c - T_env))
    sensor_g_arr.append(T_env + p.k * (T_g - T_env))
    facade_outputs.append(facade_out)

# Unpack facade outputs
T_air_local_cement = [f[0] for f in facade_outputs]
T_air_local_green = [f[1] for f in facade_outputs]
Rise_cement = [f[2] for f in facade_outputs]
Rise_green = [f[3] for f in facade_outputs]
Floors_affected_cement = [f[4] for f in facade_outputs]
Floors_affected_green = [f[5] for f in facade_outputs]

# ============================================================
# Save outputs
# ============================================================
df_out = df.copy()
df_out["T_cement"] = T_c_arr
df_out["T_green"] = T_g_arr
df_out["Sensor_near_cement"] = sensor_c_arr
df_out["Sensor_near_green"] = sensor_g_arr
df_out["T_air_end_cement"] = T_air_end_c_arr
df_out["T_air_end_green"] = T_air_end_g_arr
df_out["T_air_local_cement"] = T_air_local_cement
df_out["T_air_local_green"] = T_air_local_green
df_out["H_mix"] = H_mix_arr
df_out["Rise_cement"] = Rise_cement
df_out["Rise_green"] = Rise_green
df_out["Floors_affected_cement"] = Floors_affected_cement
df_out["Floors_affected_green"] = Floors_affected_green

out_csv = "roof_simulation_output.csv"
df_out.to_csv(out_csv, index=False)
print(f"✅ Simulation completed. Output saved to {out_csv}")

# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(sensor_c_arr, color='tomato', label="Sensor near Concrete Roof")
plt.plot(sensor_g_arr, color='seagreen', label="Sensor near Green Roof")
plt.plot(T_air_local_cement, '-.', color='orangered', label="Facade Air (Concrete)")
plt.plot(T_air_local_green, '-.', color='limegreen', label="Facade Air (Green)")
plt.plot(T_env_series, '--', color='gray', label="Ambient Air")
plt.plot(T_air_end_c_arr, ':', color='darkred', label="Air End of Roof (Concrete)")
plt.plot(T_air_end_g_arr, ':', color='darkgreen', label="Air End of Roof (Green)")
plt.xlabel("Hour")
plt.ylabel("Temperature (°C)")
plt.title("Sensor and Air Temperature Evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sensor_facade_temperatures.png")

plt.figure(figsize=(10,5))
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
plt.savefig("rise_floors.png")

plt.show()
