class FacadeBlock:
    """Facade thermal and buoyancy block."""

    def __init__(self, facade_funcs):
        self.f = facade_funcs
        self.outputs = None

    def set_inputs(self, T_air_end_c, T_air_end_g, T_env, wind, H_mix):
        self.T_air_end_c = T_air_end_c
        self.T_air_end_g = T_air_end_g
        self.T_env = T_env
        self.wind = wind
        self.H_mix = H_mix

    def do_step(self):
        self.outputs = self.f.compute_facade_effects_dynamic(
            self.T_air_end_c, self.T_air_end_g,
            self.T_env, self.wind, self.H_mix
        )
    def get_outputs(self):
        return self.outputs
