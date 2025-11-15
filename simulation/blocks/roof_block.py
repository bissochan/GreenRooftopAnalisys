import numpy as np

class RoofBlock:
    """Block for roof temperature evolution."""

    def __init__(self, T_init, alpha, Q_evap, C, roof_funcs, params):
        self.T = T_init
        self.alpha = alpha
        self.Q_evap = Q_evap
        self.C = C
        self.f = roof_funcs
        self.p = params

        self.T_air_over = None

    def set_inputs(self, T_env, G, wind, H_mix):
        self.T_env = T_env
        self.G = G
        self.wind = wind
        self.H_mix = H_mix

    def do_step(self):
        T_air = self.f.compute_air_over_roof(
            self.T, self.T_env, self.wind, self.H_mix
        )

        T_new = self.f.update_roof_euler(
            self.T, T_air, self.alpha, self.G,
            self.Q_evap, self.C, self.wind, self.p.dt_sub
        )

        dT = np.clip(T_new - self.T, -self.p.max_step_roof, self.p.max_step_roof)
        self.T += dT

        self.T_air_over = T_air

    def get_outputs(self):
        return self.T, self.T_air_over
