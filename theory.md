# Urban Roof Convective Model — Physical Description and Simulation Principles

This model simulates **the thermal behavior of roof surfaces** and the **air temperature dynamics above and near a building facade**, accounting for solar heating, evaporative cooling (for green roofs), convective heat transfer, wind-driven mixing, and buoyancy-induced vertical transport of air. The simulation provides insights into how different roof types affect the surrounding microclimate and how air parcels rise and interact with the building facade.

---

## 1. Scenario and Assumptions

The model considers two roof types:

1. **Concrete roof**: Standard, non-vegetated roof.  
   - Solar absorption drives heating.  
   - No evaporative cooling occurs.

2. **Green roof**: Vegetated roof with soil layer.  
   - Absorbs solar radiation.  
   - Provides evaporative cooling flux from transpiration and water evaporation.

**Key assumptions:**

- The roofs are flat, with defined **width (W)** and **length (L)**.  
- Air above the roof is heated via **convective transfer** from the roof surface.  
- Wind induces **mixing**, represented by a simple **eddy diffusivity model**.  
- Buoyancy effects lift air parcels vertically depending on the temperature difference relative to ambient air.  
- No heat loss during the horizontal transport from the roof to the facade (`relax_factor = 0`).  
- Time integration is performed with an **explicit Euler scheme**, subdividing each hourly time step into smaller sub-steps for numerical stability.

---

## 2. Roof Energy Balance

The roof surface temperature evolves according to an **energy balance**:

$$
T_{\text{roof}}(t+\Delta t) = T_{\text{roof}}(t) + \frac{\Delta t}{C} \left[\alpha G - Q_{\text{evap}} - h_c \left(T_{\text{roof}}(t) - T_{\text{air}}(t)\right)\right]
$$

Where:

| Symbol | Meaning | Units | Interpretation |
|--------|---------|-------|----------------|
| $T_{\text{roof}}$ | Roof surface temperature | °C | State variable updated over time |
| $T_{\text{air}}$ | Local air temperature above roof | °C | Air receives convective heat from roof |
| $\alpha$ | Solar absorptivity | – | Fraction of incoming solar radiation absorbed by the roof |
| $G$ | Solar radiation (irradiance) | W/m² | Energy flux from the sun |
| $Q_{\text{evap}}$ | Evaporative flux | W/m² | Only for green roofs, cools the surface |
| $h_c$ | Convective heat transfer coefficient | W/(m²·K) | Represents wind-enhanced heat exchange |
| $C$ | Thermal capacity per unit area | J/(m²·K) | Roof inertia: larger C → slower temperature response |
| $\Delta t$ | Time step | s | Simulation timestep (3600 s) |

### 2.1 Physical Interpretation

- **Solar Heating** ($\alpha G$): Increases roof temperature proportionally to absorptivity and irradiance.  
- **Evaporative Cooling** ($-Q_{\text{evap}}$): Active only for green roofs, reduces temperature via latent heat loss.  
- **Convective Heat Exchange** ($-h_c(T_{\text{roof}}-T_{\text{air}})$): Drives heat transfer between roof and air.  
  - Higher wind → larger $h_c$ → faster thermal equilibration.  
- **Thermal inertia** ($C$): Determines how quickly the roof responds to energy fluxes.  

---

## 3. Convective Air Temperature Above Roof

Air temperature directly above the roof is estimated using a **steady convective balance**:

$$
T_{\text{air}} = T_{\text{env}} + \frac{h_c A_{\text{roof}}}{\dot{m} c_p} \left(T_{\text{roof}} - T_{\text{env}}\right)
$$

Where:

| Symbol | Meaning | Units | Interpretation |
|--------|---------|-------|----------------|
| $T_{\text{air}}$ | Air temperature above roof | °C | Resulting from convective exchange |
| $T_{\text{env}}$ | Ambient air temperature | °C | Base environmental temperature |
| $A_{\text{roof}}$ | Roof area | m² | Effective area for heat transfer |
| $\dot{m}$ | Mass flux of air | kg/s | Wind-driven air movement above roof |
| $c_p$ | Specific heat capacity of air | J/(kg·K) | Air’s ability to store heat |
| $h_c$ | Convective coefficient | W/(m²·K) | Dependent on wind |

**Interpretation:**  
The warmer the roof relative to ambient, the more heat is transferred to the air. Wind increases air flux and reduces temperature difference, smoothing the air heating effect.

---

## 4. Mixing Layer Estimation

The **mixing height** $H_{\text{mix}}$ represents the vertical extent over which wind-driven turbulent mixing distributes heat:

$$
H_{\text{mix}} = \text{clip}(\sqrt{K_z \tau}, H_{\text{min}}, H_{\text{max}})
$$

- $K_z$: Eddy diffusivity constant.  
- $\tau = L / \text{max}(U, \epsilon)$: Air residence time over roof along wind direction.  
- Limits: $H_{\text{min}}$ and $H_{\text{max}}$ ensure physically reasonable bounds.  

**Interpretation:**  
- Faster wind → shorter residence time → smaller vertical mixing.  
- Slower wind → longer residence → higher effective mixing height.

---

## 5. Buoyant Rise and Facade Interaction

Air leaving the roof rises due to **buoyancy** if warmer than ambient:

$$
a_z = g \frac{\max(T_{\text{air}} - T_{\text{env}}, 0)}{T_{\text{env}} + T_0}
$$

$$
\text{Rise} = \gamma_{\text{damping}} \cdot \frac{1}{2} a_z \tau^2 + H_{\text{mix}}
$$

Where:

| Symbol | Meaning | Units | Interpretation |
|--------|---------|-------|----------------|
| $a_z$ | Buoyant vertical acceleration | m/s² | Positive only if air is warmer than environment |
| $\tau$ | Transit time roof → facade | s | Time available for vertical lift |
| $\gamma_{\text{damping}}$ | Buoyancy damping factor | – | Accounts for turbulent dissipation and drag |
| $H_{\text{mix}}$ | Mixing height | m | Initial vertical extent of air parcel |

The **number of affected floors** is calculated by dividing total rise by floor height:

$$
\text{Floors affected} = \lceil \frac{\text{Rise}}{h_{\text{floor}}} \rceil
$$

**Interpretation:**  
- Air parcels from warmer roofs rise higher.  
- Green roofs produce slightly lower air temperatures → reduced rise.  
- Mixing layer sets a minimum rise even for neutral or slightly cooler air.

---

## 6. Numerical Implementation

- **Time stepping**: Hourly steps subdivided into `n_sub` sub-steps for stability.  
- **Explicit Euler update**: Surface temperature incrementally updated.  
- **Roof-to-air convective calculation**: Uses instantaneous wind and roof temperature.  
- **Spin-up procedure**: Simulation iterates for a few cycles to reach thermal equilibrium before recording output.

---

## 7. Outputs

The simulation produces:

1. **Roof surface temperatures** (`T_cement`, `T_green`)  
2. **Estimated sensor readings near roofs** (`Sensor_near_cement`, `Sensor_near_green`)  
3. **Air temperature at roof exit** (`T_air_end_cement`, `T_air_end_green`)  
4. **Air temperature at facade** (`T_air_local_cement`, `T_air_local_green`)  
5. **Mixing height** (`H_mix`)  
6. **Buoyant rise and floors affected** (`Rise_*`, `Floors_affected_*`)  

Plots are generated for:

- Roof and facade temperatures over time.  
- Air parcel rise and corresponding floors affected.

---

## 8. Practical Relevance

This model allows urban planners and engineers to:

- Compare **microclimate effects** of different roof types.  
- Predict **thermal comfort** near building facades.  
- Assess the influence of **wind and solar radiation** on roof and air temperatures.  
- Quantify **buoyant air movement**, which affects ventilation and pollutant dispersion.

By combining **roof energy balance**, **convective air heating**, **mixing height estimation**, and **buoyant rise calculation**, the model provides a **physically consistent yet computationally efficient framework** for simulating roof-air interactions in urban environments.
