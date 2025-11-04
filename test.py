import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters (edit these values to adapt the model)
# ============================================================

alpha_cement = 0.7       # Solar absorptivity of concrete roof
alpha_green = 0.8        # Solar absorptivity of green roof
Q_evap_green = 100       # Evaporative cooling effect for green roof (W/m^2)

C_cement = 176000        # Thermal inertia of concrete (J/m^2*K)
C_green = 270000         # Thermal inertia of green roof (J/m^2*K)

k = 0.7                  # Coupling factor between roof & sensor air
dt = 3600                # Time step (seconds) -> 3600s = 1 hour

# Initial temperatures of roofs (°C)
T_cement_initial = 11
T_green_initial = 11

# ============================================================
# Load CSV input
# The CSV must contain: Radiation (W/m2), T_air (°C), Wind (m/s)
# ============================================================

CSV_PATH = "data/csv/hourly_trend.csv"
df = pd.read_csv(CSV_PATH)

radiation = df["Rad_Trend"].values
T_air_series = df["Temp_Trend"].values
wind_series = df["Wind_Trend"].values

hours = len(df)

# ============================================================
# Simulation storage
# ============================================================

T_cement = np.zeros(hours)
T_green = np.zeros(hours)

sensor_cement = np.zeros(hours)
sensor_green = np.zeros(hours)

# Initial conditions
T_cement[0] = T_air_series[0]
T_green[0] = T_air_series[0]

# Initial sensor temperatures
sensor_cement[0] = T_air_series[0] + k * (T_cement[0] - T_air_series[0])
sensor_green[0]  = T_air_series[0] + k * (T_green[0]  - T_air_series[0])

# ============================================================
# Model (hour by hour)
# ============================================================

def heat_exchange(T_surface, T_air, alpha, G, Q_evap, C, wind, dt):
    """
    Update rule for roof surface temperature with thermal inertia.
    """
    h_c = 10 + 5 * wind  # convective heat loss increases with wind speed
    Q_net = alpha * G - Q_evap - h_c * (T_surface - T_air)
    T_new = T_surface + (dt / C) * Q_net
    return T_new

for t in range(1, hours):
    T_cement[t] = heat_exchange(T_cement[t-1], T_air_series[t], alpha_cement, radiation[t], 0, C_cement, wind_series[t], dt)
    T_green[t]  = heat_exchange(T_green[t-1],  T_air_series[t], alpha_green,  radiation[t], Q_evap_green, C_green, wind_series[t], dt)

    # Sensor temperature near roof (simple linear coupling)
    sensor_cement[t] = T_air_series[t] + k * (T_cement[t] - T_air_series[t])
    sensor_green[t]  = T_air_series[t] + k * (T_green[t]  - T_air_series[t])

# ============================================================
# Store results in DataFrame
# ============================================================

df["T_cement"] = T_cement
df["T_green"] = T_green
df["Sensor_near_cement"] = sensor_cement
df["Sensor_near_green"] = sensor_green

df.to_csv("roof_simulation_output.csv", index=False)
print("✅ Simulation completed. Results saved to roof_simulation_output.csv")

# ============================================================
# Plot results
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(df["Sensor_near_cement"], label="Sensor near Concrete Roof")
plt.plot(df["Sensor_near_green"], label="Sensor near Green Roof")
plt.plot(T_air_series, label="Ambient Air Temperature", linestyle='--', color='gray')
plt.xlabel("Hour")
plt.ylabel("Sensor Temperature (°C)")
plt.title("Sensor Temperature Over Time")
plt.legend()
plt.grid(True)
plt.savefig("roof_sensor_temperatures.png")
plt.show()
