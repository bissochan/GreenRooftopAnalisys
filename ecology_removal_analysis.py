import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

GREEN_ROOF_AREA = 600.0
df = pd.read_csv('simulation\\csv_results\\simulation_results.csv')

# Convert CO2 ppm → µg/m³
MOLAR_MASS_CO2 = 44.01  # g/mol
MOLAR_VOLUME = 22.4     # L/mol at 0°C
AIR_DENSITY = 1.2       # kg/m³
df['CO2_Initial_ugm3'] = df['CO2_ppm'] * (MOLAR_MASS_CO2 / MOLAR_VOLUME) * 1000 * AIR_DENSITY
df['CO2_Final_ugm3'] = df['CO2_Initial_ugm3'] - (df['CO2_Removed_g'] * 1e6 / (GREEN_ROOF_AREA * df['H_mix']))

# Aggregate ALL data
hourly = df.groupby("hour").agg({
    "CO2_Initial_ugm3": ["mean", "std"],
    "CO2_Final_ugm3": ["mean", "std"],
    "CO_Initial_ugm3": ["mean", "std"],
    "CO_Final_ugm3": ["mean", "std"],
    "PM10_Initial_ugm3": ["mean", "std"],
    "PM10_Final_ugm3": ["mean", "std"],
    "PM25_Initial_ugm3": ["mean", "std"],
    "PM25_Final_ugm3": ["mean", "std"],
}).round(3)

hourly.columns = ['_'.join(col).strip() for col in hourly.columns]
hours = hourly.index

# 4 PLOTS - 1 per pollutant
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

pollutants = [
    ("CO2", "#34495e", "CO₂"),
    ("CO", "#e74c3c", "CO"),
    ("PM10", "#9b59b6", "PM10"),
    ("PM25", "#e67e22", "PM2.5")
]

for i, (pol, color, label) in enumerate(pollutants):
    row, col = divmod(i, 2)
    axs[row, col].errorbar(hours, hourly[f"{pol}_Initial_ugm3_mean"], 
                          yerr=hourly[f"{pol}_Initial_ugm3_std"], 
                          fmt='-o', color=color, label="Without green roof", 
                          linewidth=3, capsize=6)
    axs[row, col].errorbar(hours, hourly[f"{pol}_Final_ugm3_mean"], 
                          yerr=hourly[f"{pol}_Final_ugm3_std"], 
                          fmt='-s', color="#27ae60", label="With green roof", 
                          linewidth=3, capsize=6)
    axs[row, col].set_title(f"{label}: Hourly Average Concentration", fontweight='bold', fontsize=14)
    axs[row, col].set_xlabel("Hour of the day")
    axs[row, col].set_ylabel(f"{label} (µg/m³)")
    axs[row, col].legend()
    axs[row, col].grid(True, alpha=0.3)
    axs[row, col].set_ylim(0, None)

plt.suptitle("Green Roof Effect: All Pollutants Comparison\n"
             "(Without vs With Green Roof - Monte Carlo 100 simulations)", 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('green_roof_ALL_pollutants_EN.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nDAILY AVERAGE REDUCTIONS:")
for pol in ["CO2", "CO", "PM10", "PM25"]:
    init_col, final_col = f"{pol}_Initial_ugm3_mean", f"{pol}_Final_ugm3_mean"
    red = ((hourly[init_col] - hourly[final_col]) / hourly[init_col] * 100).mean()
    print(f"{pol}: {red:.2f}% ↓")
