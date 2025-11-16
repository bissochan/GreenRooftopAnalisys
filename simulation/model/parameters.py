# Physical and numerical parameters for the blocks

alpha_cement = 0.7        # Cement roof solar absorptivity
alpha_green = 0.8         # Green roof solar absorptivity

k_evap_green = 0.2        # Green roof evaporation coefficient

C_cement = 176000.0       # Cement roof heat capacity (J/m²·K)
C_green = 270000.0        # Green roof heat capacity (J/m²·K)

W = 20.0                  # Roof width (m)
L = 30.0                  # Roof length (m)
D = 30.0                  # Building depth (m)
A_eff = W * L             # Effective roof area (m²)

h_floor = 3.0             # Floor height (m)

rho = 1.2                 # Air density (kg/m³)
cp = 1005.0               # Air specific heat (J/kg·K)
wind_eps = 0.05           # Min wind speed to avoid zeros

Kz = 0.3                  # Eddy diffusivity (m²/s)
H_min = 0.2               # Min mixing height (m)
H_max = 6.0               # Max mixing height (m)

g = 9.81                  # Gravity (m/s²)
T0_K = 273.15             # °C to Kelvin offset
gamma_damping = 0.45      # Stabilization damping factor

k = 0.7                   # Coupling factor for internal air mixing

dt = 3600.0               # Main timestep (s)
n_sub = 60                # Substeps per hour
dt_sub = dt / n_sub       # Substep duration (s)

max_step_roof = 30.0      # Max solver step for roof (s)
h_c_max = 100.0           # Max convective coefficient

SPIN_CYCLES = 2           # Warm-up cycles for model stabilization
