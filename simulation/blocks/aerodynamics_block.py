class AerodynamicsBlock:
    """Computes H_mix each timestep."""

    def __init__(self, aero_funcs):
        self.f = aero_funcs
        self.H_mix = None

    def set_inputs(self, wind):
        self.wind = wind

    def do_step(self):
        self.H_mix = self.f.compute_mixing_height_kz(self.wind)

    def get_outputs(self):
        return self.H_mix
