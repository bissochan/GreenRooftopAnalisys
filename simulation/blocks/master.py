class Master:
    """Master algorithm coordinating all blocks."""

    def __init__(self, roof_cement, roof_green, aero, facade, params):
        self.roof_c = roof_cement
        self.roof_g = roof_green
        self.aero = aero
        self.facade = facade
        self.p = params

    def do_step(self, T_env, G, wind):
        # Aerodynamic step ===============
        self.aero.set_inputs(wind)
        self.aero.do_step()
        H_mix = self.aero.get_outputs()

        # Roofs step ===============
        self.roof_c.set_inputs(T_env, G, wind, H_mix)
        self.roof_g.set_inputs(T_env, G, wind, H_mix)

        for _ in range(self.p.n_sub):
            self.roof_c.do_step()
            self.roof_g.do_step()

        T_c, T_air_c = self.roof_c.get_outputs()
        T_g, T_air_g = self.roof_g.get_outputs()

        # Facade step ===============
        self.facade.set_inputs(T_air_c, T_air_g, T_env, wind, H_mix)
        self.facade.do_step()
        facade_out = self.facade.get_outputs()

        return {
            "T_c": T_c,
            "T_g": T_g,
            "H_mix": H_mix,
            "T_air_c": T_air_c,
            "T_air_g": T_air_g,
            "T_air_fac_c": facade_out["T_air_fac_c"],
            "T_air_fac_g": facade_out["T_air_fac_g"],
            "Rise_c": facade_out["rise_c"],
            "Rise_g": facade_out["rise_g"],
            "Floors_c": facade_out["floors_c"],
            "Floors_g": facade_out["floors_g"],
        }
