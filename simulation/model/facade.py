import numpy as np
from .parameters import *


def compute_facade_effects_dynamic(T_air_end_c, T_air_end_g, T_env, wind, H_mix):
    """
    Compute facade temperatures, buoyant rise and floors using a time-resolved
    advection + relaxation model for the air parcel traveling from roof -> facade.

    Parameters:
      - T_air_end_c, T_air_end_g: air parcel temperature leaving roof (°C)
      - T_env: ambient temperature (°C)
      - wind: wind speed (m/s)
      - H_mix: initial mixing height (m)
      - D: distance roof->facade (m)
      - gamma_damping, h_floor, wind_eps, g, T0_K: physical params from params module
      - max_steps: limit on integration steps
      - k_relax_base: base relaxation rate (s^-1) when wind = 0
      - k_relax_wind_factor: multiplier to increase k_relax with wind
    Returns dict with keys like before.
    """
    max_steps=3600
    k_relax_base=0.01,
    k_relax_wind_factor=0.5

    # protect wind and compute transit time (seconds)
    U = max(wind, wind_eps)
    tau = max(D / U, 0.0)

    if tau <= 1.0:
        # short transit: single-step analytical relaxation
        n_steps = 1
        dt = tau if tau > 0 else 1.0
    else:
        n_steps = int(np.ceil(min(tau, max_steps)))
        dt = tau / n_steps

    k_relax = k_relax_base + k_relax_wind_factor * U
    k_relax = float(np.clip(k_relax, 1e-4, 1.0))    # ensure reasonable bounds

    # integration initial conditions
    z_c = 0.0  # initial vertical position above roof reference (m)
    z_g = 0.0
    v_c = 0.0  # vertical velocity (m/s)
    v_g = 0.0

    T_c = float(T_air_end_c)
    T_g = float(T_air_end_g)
    T_env_K = T_env + T0_K  # used in buoyancy denom (Kelvin offset)

    # integrate in time
    for _ in range(n_steps):
        # relax toward environment: exponential Euler step
        T_c = T_c - k_relax * (T_c - T_env) * dt
        T_g = T_g - k_relax * (T_g - T_env) * dt

        a_c = g * max(T_c - T_env, 0.0) / T_env_K
        a_g = g * max(T_g - T_env, 0.0) / T_env_K

        v_c += a_c * dt
        v_g += a_g * dt

        z_c += v_c * dt
        z_g += v_g * dt

        if z_c < -1e3 or z_g < -1e3:
            z_c = max(z_c, -1e3)
            z_g = max(z_g, -1e3)
            v_c = np.sign(v_c) * min(abs(v_c), 1e3)
            v_g = np.sign(v_g) * min(abs(v_g), 1e3)

    rise_c = gamma_damping * z_c + H_mix
    rise_g = gamma_damping * z_g + H_mix

    # enforce minimum fraction of mixing height
    rise_c = max(rise_c, H_mix * 0.1)
    rise_g = max(rise_g, H_mix * 0.1)

    floors_c = int(np.ceil(rise_c / h_floor))
    floors_g = int(np.ceil(rise_g / h_floor))

    # final facade air temp: parcel has mixed toward env during flight
    T_air_fac_c = T_c
    T_air_fac_g = T_g

    return {
        "T_air_fac_c": T_air_fac_c,
        "T_air_fac_g": T_air_fac_g,
        "rise_c": float(rise_c),
        "rise_g": float(rise_g),
        "floors_c": floors_c,
        "floors_g": floors_g,
    }
