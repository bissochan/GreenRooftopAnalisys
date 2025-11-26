"""
model/ecology.py

Funzioni utili per calcoli ecologici:
- rimozione PM per deposizione secca (g)
- sequestro di carbonio (kgC, kgCO2)
- conversione Q_evap (W/m2) -> energia (kWh) e massa d'acqua evaporata (kg)
"""

from .parameters import L_v, seconds_per_year
import numpy as np

# ECOLOGICAL COEFFICIENTS & PARAMETERS (configurable)
CO2_UPTAKE_RATE = 0.35        # g CO2 removed per m² per hour per (W/m² / 100)
BASAL_CO2_UPTAKE_G_PER_M2 = 0.005   # g CO2 / m² / h (basal night uptake)
CO2_CONC_FACTOR = 1e-3              # factor to add small conc-dependent uptake
O2_PRODUCTION_FACTOR = 0.72         # g O2 produced per g CO2 fixed
GREEN_ROOF_AREA = 600.0             # m²

# Deposition parameters for particulates and gas
Vd_PM10 = 2e-3        # m/s
Vd_PM25 = 1.2e-3      # m/s
Vd_CO = 5e-5          # m/s
LAI_FACTOR = 2.0      # leaf area index multiplier
SECONDS_PER_HOUR = 3600.0

def pm_removed_g(area_m2, conc_ug_m3, Vd_m_s, dt_s):
    """
    Rimuove particolato per deposizione secca su area_m2 durante dt_s.
    conc_ug_m3: concentrazione media durante il passo (µg/m³)
    Vd_m_s: deposition velocity (m/s)
    Restituisce: removed_mass_g (grammi)
    Formula:
      flux (µg/m2/s) = Vd * C
      annual/integrated = flux * dt_s * area_m2  (µg) -> /1e6 -> g
    """
    flux_ug_m2_s = Vd_m_s * conc_ug_m3  # µg / m2 / s
    total_ug = flux_ug_m2_s * dt_s * area_m2
    total_g = total_ug / 1e6
    return float(total_g)

def carbon_sequestered_kg(area_m2, gC_m2_yr, dt_s):
    """
    Sequestro di carbonio per un passo dt_s.
    gC_m2_yr: gC per m2 per anno
    Restituisce: kgC, kgCO2 per il passo
    """
    gC_total = gC_m2_yr * area_m2 * (dt_s / seconds_per_year)
    kgC = gC_total / 1000.0
    kgCO2 = kgC * 44.0 / 12.0
    return float(kgC), float(kgCO2)

def evap_energy_and_mass(Q_evap_Wm2, area_m2, dt_s):
    """
    Converte Q_evap (W/m2) su area e tempo in:
      - energia kWh
      - massa di acqua evaporata (kg)
    Q_evap_Wm2: potenza per metro quadro (W/m²)
    area_m2: area (m²)
    dt_s: durata passo (s)
    """
    # Energia totale in J
    energy_J = Q_evap_Wm2 * area_m2 * dt_s
    kWh = energy_J / (3600.0 * 1000.0)
    # massa d'acqua evaporata (kg) = energia_J / L_v
    mass_kg = energy_J / L_v
    return float(kWh), float(mass_kg)

def aggregate_scenarios(low, typ, high):
    """Utility: ritorna diz con low/typ/high"""
    return {"low": low, "typical": typ, "high": high}

def compute_ecology(radiation, co2, co, pm10, pm25):
    """
    Compute ecological removals for one hour step.

    Returns:
      co2_removed_g, co_removed_ug, pm10_removed_ug, pm25_removed_ug, o2_produced_g
    Notes:
      - CO2: photosynthesis-driven (radiation), plus basal and small conc-dependent term
      - PM/CO: deposition flux = Vd * C * Area -> µg/s -> *3600 -> µg/h
      - Handles NaN/None inputs gracefully.
    """
    # CO2 uptake (g/h)
    co2_from_light = CO2_UPTAKE_RATE * (max(0.0, radiation) / 100.0) * GREEN_ROOF_AREA
    co2_basal = BASAL_CO2_UPTAKE_G_PER_M2 * GREEN_ROOF_AREA
    co2_conc_term = 0.0
    if co2 is not None and not np.isnan(co2):
        co2_conc_term = CO2_CONC_FACTOR * float(co2) * GREEN_ROOF_AREA
    co2_removed_g = max(0.0, co2_from_light + co2_basal + co2_conc_term)

    # PM10 removal (µg/h)
    pm10_c = 0.0 if pm10 is None or np.isnan(pm10) else float(pm10)
    pm10_flux_ug_s = Vd_PM10 * LAI_FACTOR * pm10_c * GREEN_ROOF_AREA
    pm10_removed_ug = max(0.0, pm10_flux_ug_s * SECONDS_PER_HOUR)

    # PM2.5 removal (µg/h)
    pm25_c = 0.0 if pm25 is None or np.isnan(pm25) else float(pm25)
    pm25_flux_ug_s = Vd_PM25 * LAI_FACTOR * pm25_c * GREEN_ROOF_AREA
    pm25_removed_ug = max(0.0, pm25_flux_ug_s * SECONDS_PER_HOUR)

    # CO removal (µg/h)
    co_c = 0.0 if co is None or np.isnan(co) else float(co)
    co_flux_ug_s = Vd_CO * LAI_FACTOR * co_c * GREEN_ROOF_AREA
    co_removed_ug = max(0.0, co_flux_ug_s * SECONDS_PER_HOUR)

    # O2 production from CO2 fixation (g)
    o2_produced_g = co2_removed_g * O2_PRODUCTION_FACTOR

    return co2_removed_g, co_removed_ug, pm10_removed_ug, pm25_removed_ug, o2_produced_g
