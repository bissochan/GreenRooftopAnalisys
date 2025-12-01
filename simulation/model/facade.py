import numpy as np
from .parameters import *

def compute_facade_effects_dynamic(T_air_end_c, T_air_end_g, T_env, wind, H_mix):
    dt = 1.0
    t_mid = 10.0
    alpha_temp = 0.2
    T0_rise = 18.0
    beta_rise = 1.0
    max_rise_rate = 10.0  # scale factor for rise intensity

    U = max(wind, wind_eps)
    tau = max(D / U, 0.1)   # transit time (s)

    # decreasing temperature sigmoid over time
    def temp_sigmoid_decreasing(T_start, t):
        return T_env + (T_start - T_env) / (1.0 + np.exp(alpha_temp * (t - t_mid)))

    # temperature gating (0..1)
    def rise_gate(T_air):
        return 1.0 / (1.0 + np.exp(-beta_rise * (T_air - T0_rise)))

    T_env_K = T_env + T0_K
    # rise increment per step
    def dz_increment(T_air):
        delta_T = max(T_air - T_env, 0.0)
        gate = rise_gate(T_air)
        delta_scaling = delta_T / T_env_K
        return max_rise_rate * gate * delta_scaling * dt

    n_steps = int(max(1, round(tau / dt)))

    # initial heights
    rise_c = H_mix
    rise_g = H_mix

    # time stepping
    for i in range(n_steps):
        t = i * dt
        T_c = temp_sigmoid_decreasing(T_air_end_c, t)
        T_g = temp_sigmoid_decreasing(T_air_end_g, t)

        rise_c += dz_increment(T_c)
        rise_g += dz_increment(T_g)

    # convert height to floors
    floors_c = int(np.ceil(max(rise_c, H_mix * 0.1) / h_floor))
    floors_g = int(np.ceil(max(rise_g, H_mix * 0.1) / h_floor))

    return {
        "T_air_fac_c": temp_sigmoid_decreasing(T_air_end_c, tau),
        "T_air_fac_g": temp_sigmoid_decreasing(T_air_end_g, tau),
        "rise_c": float(rise_c),
        "rise_g": float(rise_g),
        "floors_c": floors_c,
        "floors_g": floors_g,
    }
