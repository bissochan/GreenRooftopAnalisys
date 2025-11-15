import numpy as np
from .parameters import *

def compute_mixing_height_kz(wind):
    """Compute dynamic mixing height from eddy diffusivity."""
    tau = L / max(wind, wind_eps)
    H = np.sqrt(Kz * tau)
    return float(np.clip(H, H_min, H_max))
