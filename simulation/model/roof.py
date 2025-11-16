import numpy as np
from .parameters import *

def compute_convection_coeff(wind):
    """Compute convective heat transfer coefficient."""
    return min(10.0 + 5.0 * wind, h_c_max)

def compute_air_over_roof(T_surface, T_env, wind, H_mix):
    """Compute convective air temperature above the roof."""
    wind_use = max(wind, wind_eps)
    h_c = compute_convection_coeff(wind_use)
    A_cross = W * H_mix
    m_dot = rho * wind_use * A_cross
    deltaT = (h_c * A_eff / (m_dot * cp)) * (T_surface - T_env) * 2.0
    return T_env + deltaT

def compute_evaporation(K_evap, G, wind):
    """Compute evaporation heat loss for green roof."""
    if K_evap == 0.0:
        return 0.0
    else:
        return K_evap * G * (1 + 0.3 * wind)

def update_roof_euler(T_surface, T_air_local, alpha, G, K_evap, C, wind, dt_local):
    """Explicit Euler update for roof surface temperature."""
    h_c = compute_convection_coeff(wind)
    Q_evap = compute_evaporation(K_evap, G, wind)
    Q_net = alpha * G - Q_evap - h_c * (T_surface - T_air_local)
    return T_surface + (dt_local / C) * Q_net
