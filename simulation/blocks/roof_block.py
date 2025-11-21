import numpy as np
from model import roof as roof_funcs  # notare import verso package model
from model.parameters import L_v  # opzionale se vuoi usarlo qui

class RoofBlock:
    """Block for roof temperature evolution."""

    def __init__(self, T_init, alpha, K_evap, C, roof_funcs, params):
        self.T = T_init
        self.alpha = alpha
        self.K_evap = K_evap
        self.C = C
        self.f = roof_funcs
        self.p = params

        self.T_air_over = None
        # Nuove variabili per output ecologico
        self.Q_evap_Wm2 = 0.0  # potenza di evaporazione (W/m2)
        self.et_mass_kg_per_m2_s = 0.0  # massa evaporata per m2 per s

    def set_inputs(self, T_env, G, wind, H_mix):
        self.T_env = T_env
        self.G = G
        self.wind = wind
        self.H_mix = H_mix

    def do_step(self):
        T_air = self.f.compute_air_over_roof(
            self.T, self.T_env, self.wind, self.H_mix
        )

        # Calcolo evaporazione (il roof model fornisce Q_evap in unità coerenti)
        Q_evap = self.f.compute_evaporation(self.K_evap, self.G, self.wind)
        self.Q_evap_Wm2 = float(Q_evap)

        # Convertiamo Q_evap (W/m2) in massa evaporata per m2/s: m_dot = Q / L_v (kg/s per m2)
        # se L_v non è disponibile qui, possiamo tenerlo nel module model.ecology; ma è comodo averlo:
        try:
            from model.parameters import L_v as _L_v
        except Exception:
            _L_v = L_v
        self.et_mass_kg_per_m2_s = float(self.Q_evap_Wm2 / _L_v) if _L_v > 0 else 0.0

        T_new = self.f.update_roof_euler(
            self.T, T_air, self.alpha, self.G,
            self.K_evap, self.C, self.wind, self.p.dt_sub
        )

        dT = np.clip(T_new - self.T, -self.p.max_step_roof, self.p.max_step_roof)
        self.T += dT

        self.T_air_over = T_air

    def get_outputs(self):
        # Manteniamo la firma originale per compatibilità (T, T_air_over)
        return self.T, self.T_air_over

    def get_evaporation(self):
        """Ritorna Q_evap_Wm2 e et_mass_kg_per_m2_s"""
        return self.Q_evap_Wm2, self.et_mass_kg_per_m2_s
