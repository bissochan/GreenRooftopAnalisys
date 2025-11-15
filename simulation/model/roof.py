import numpy as np
from .parameters import *

def compute_air_over_roof(T_surface, T_env, wind, H_mix):
    """Compute convective air temperature above the roof."""
    wind_use = max(wind, wind_eps)
    h_c = min(10.0 + 5.0 * wind_use, h_c_max)
    A_cross = W * H_mix
    m_dot = rho * wind_use * A_cross
    deltaT = (h_c * (A_eff * 2.0) / (m_dot * cp)) * (T_surface - T_env)
    return T_env + deltaT

def update_roof_euler(T_surface, T_air_local, alpha, G, Q_evap, C, wind, dt_local):
    """Explicit Euler update for roof surface temperature."""
    h_c = min(10.0 + 5.0 * wind, h_c_max)
    Q_net = alpha * G - Q_evap - h_c * (T_surface - T_air_local)
    return T_surface + (dt_local / C) * Q_net
