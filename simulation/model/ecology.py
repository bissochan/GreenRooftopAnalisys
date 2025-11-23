"""
model/ecology.py

Funzioni utili per calcoli ecologici:
- rimozione PM per deposizione secca (g)
- sequestro di carbonio (kgC, kgCO2)
- conversione Q_evap (W/m2) -> energia (kWh) e massa d'acqua evaporata (kg)
"""

from .parameters import L_v, seconds_per_year
import numpy as np

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
