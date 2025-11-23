import numpy as np
from .parameters import *


def compute_facade_effects(T_air_end_c, T_air_end_g, T_env, wind, H_mix):
    """Compute facade air temperature and buoyant rise."""
    tau = D / max(wind, wind_eps)
    relax_factor = 0.0

    T_air_fac_c = T_air_end_c * (1 - relax_factor) + T_env * relax_factor
    T_air_fac_g = T_air_end_g * (1 - relax_factor) + T_env * relax_factor

    a_z_c = g * max(T_air_fac_c - T_env, 0.0) / (T_env + T0_K)
    a_z_g = g * max(T_air_fac_g - T_env, 0.0) / (T_env + T0_K)

    rise_c = gamma_damping * (0.5 * a_z_c * tau ** 2) + H_mix
    rise_g = gamma_damping * (0.5 * a_z_g * tau ** 2) + H_mix

    rise_c = max(rise_c, H_mix * 0.1)
    rise_g = max(rise_g, H_mix * 0.1)

    floors_c = np.ceil(rise_c / h_floor)
    floors_g = np.ceil(rise_g / h_floor)

    return {
        "T_air_fac_c": T_air_fac_c,
        "T_air_fac_g": T_air_fac_g,
        "rise_c": rise_c,
        "rise_g": rise_g,
        "floors_c": floors_c,
        "floors_g": floors_g
    }
