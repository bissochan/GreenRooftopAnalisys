# Integrated Urban Roof Model: Physical and Ecological Framework

This document details the **deterministic physical and chemical principles** used in the simulation. It unifies the thermodynamic modeling of the roof-facade interaction with the ecological modeling of pollutant removal.

The framework consists of two coupled domains:
1.  **Thermo-Aerodynamic Domain:** Solves heat transfer, wind-driven mixing, and buoyant plume rise.
2.  **Ecological Domain:** Computes dry deposition fluxes and carbon sequestration rates based on biological parameters.

---

# PART I: Thermo-Aerodynamic Model

This section describes the simulation of roof surface thermal behavior, convective heat exchange, and the resulting air parcel dynamics towards the building facade.

## 1. Scenario and Assumptions

The model considers two roof types:

1.  **Concrete roof**: Standard, non-vegetated roof.
    * Solar absorption drives heating.
    * No evaporative cooling occurs.

2.  **Green roof**: Vegetated roof with soil layer.
    * Absorbs solar radiation.
    * Provides evaporative cooling flux from transpiration and water evaporation.
    * Provides ecosystem services (PM removal, CO₂ uptake).

**Key Assumptions:**

* The roofs are flat, with defined **width (W)** and **length (L)**.
* Air above the roof is heated via **convective transfer** from the roof surface.
* Wind induces **mixing**, represented by a simple **eddy diffusivity model**.
* **Dynamic Transport & Decay**: Air parcels move from the roof to the facade. During this transport, they lose heat to the environment (non-adiabatic).
* **Gated Buoyancy**: Vertical rise is not automatic; it requires the air parcel to exceed a specific temperature threshold.
* Time integration is performed with an **explicit Euler scheme** for the roof and a separate time-stepping loop for the facade transport.

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
| $\Delta t$ | Time step | s | Simulation substep |

### 2.1 Physical Interpretation

* **Solar Heating** ($\alpha G$): Increases roof temperature proportionally to absorptivity and irradiance.
* **Evaporative Cooling** ($-Q_{\text{evap}}$): Active only for green roofs, reduces temperature via latent heat loss.
* **Convective Heat Exchange** ($-h_c(T_{\text{roof}}-T_{\text{air}})$): Drives heat transfer between roof and air.
    * Higher wind → larger $h_c$ → faster thermal equilibration.
* **Thermal inertia** ($C$): Determines how quickly the roof responds to energy fluxes.

---

## 3. Convective Air Temperature Above Roof

Air temperature directly above the roof is estimated using a **steady convective balance**, modified by an enhancement factor:

$$
T_{\text{air}} = T_{\text{env}} + 2.0 \cdot \frac{h_c A_{\text{roof}}}{\dot{m} c_p} \left(T_{\text{roof}} - T_{\text{env}}\right)
$$

Where:

| Symbol | Meaning | Units | Interpretation |
|--------|---------|-------|----------------|
| $T_{\text{air}}$ | Air temperature above roof | °C | Resulting from convective exchange |
| $T_{\text{env}}$ | Ambient air temperature | °C | Base environmental temperature |
| $A_{\text{roof}}$ | Roof area | m² | Effective area for heat transfer |
| $\dot{m}$ | Mass flux of air | kg/s | Wind-driven air movement above roof |
| $2.0$ | Enhancement Factor | - | Accounts for heat accumulation along the fetch |
| $h_c$ | Convective coefficient | W/(m²·K) | Dependent on wind |

**Interpretation:**
The warmer the roof relative to ambient, the more heat is transferred to the air. The factor **2.0** emphasizes that as air moves across the roof, it accumulates heat, often making the effective temperature difference at the downwind edge higher than a simple instantaneous mix would suggest.

---

## 4. Mixing Layer Estimation

The **mixing height** $H_{\text{mix}}$ represents the vertical extent over which wind-driven turbulent mixing distributes heat:

$$
H_{\text{mix}} = \text{clip}(\sqrt{K_z \tau}, H_{\text{min}}, H_{\text{max}})
$$

* $K_z$: Eddy diffusivity constant.
* $\tau = L / \text{max}(U, \epsilon)$: Air residence time over roof along wind direction.
* Limits: $H_{\text{min}}$ and $H_{\text{max}}$ ensure physically reasonable bounds.

**Interpretation:**
* Faster wind → shorter residence time → smaller vertical mixing.
* Slower wind → longer residence → higher effective mixing height.

---

## 5. Dynamic Facade Interaction (Plume Rise)

Unlike simple kinematic models, this simulation uses a **step-wise dynamic integration** to model the air parcel as it moves away from the roof.

### 5.1 Temperature Decay
As the air parcel leaves the roof, its temperature ($T_{parcel}$) relaxes towards the environmental temperature ($T_{env}$) following a **decreasing sigmoid function**:

$$
T_{parcel}(t) = T_{env} + \frac{T_{start} - T_{env}}{1 + e^{\alpha_{temp}(t - t_{mid})}}
$$

### 5.2 Gated Buoyancy Integration
The vertical rise is calculated by integrating an incremental rise ($dz$) over the transit time. The rise is "gated" by a logistic function to ensure significant rising only occurs when the temperature exceeds a threshold ($T_{0,rise} \approx 18^\circ C$).

$$
dz = \text{max\_rate} \cdot \text{Gate}(T) \cdot \frac{T_{parcel} - T_{env}}{T_{env}} \cdot dt
$$

$$
\text{Total Rise} = H_{mix} + \sum dz
$$

**Interpretation:**
* **Decay**: Warm air doesn't stay warm forever. As it travels from the roof edge to the facade wall, it mixes with cool ambient air. If the wind is slow, it cools down significantly before hitting the facade.
* **Gating**: Buoyancy isn't linear. Small temperature differences might not overcome atmospheric stratification. The "gate" ensures that only significantly warm parcels create a strong upward plume.

---

# PART II: Ecological Model

This section details the biological and chemical formulas used to calculate the environmental benefits of the green roof (Green Roof only).

## 6. Carbon Dioxide Sequestration

CO₂ removal is modeled as the sum of three distinct components:

$$
\text{CO}_2^{\text{removed}} = \text{CO}_2^{\text{light}} + \text{CO}_2^{\text{basal}} + \text{CO}_2^{\text{conc}}
$$

### 6.1 The Components
1.  **Light-driven (Photosynthesis):** Dominant term, proportional to solar radiation ($R$).
    $$\text{CO}_2^{\text{light}} = k_{\text{photo}} \cdot \frac{R}{100} \cdot A_{\text{roof}}$$
2.  **Basal uptake:** Constant background absorption (dark respiration balance).
    $$\text{CO}_2^{\text{basal}} = k_{\text{basal}} \cdot A_{\text{roof}}$$
3.  **Concentration-dependent:** Slight increase in uptake with higher ambient CO₂.
    $$\text{CO}_2^{\text{conc}} = k_{\text{conc}} \cdot C_{\text{CO}_2} \cdot A_{\text{roof}}$$

**Parameters:**
* $k_{\text{photo}} = 0.35$ g/(m²·h) per unit radiation.
* $k_{\text{basal}} = 0.005$ g/(m²·h).
* $A_{\text{roof}} = 600$ m².

**Oxygen Production:**
Oxygen generation is derived stoichiometrically from the fixed carbon mass:
$$
\text{O}_2^{\text{prod}} = \text{CO}_2^{\text{removed}} \times 0.72
$$

---

## 7. Particulate and Gas Deposition

Removal of PM10, PM2.5, and CO is calculated using a **dry deposition** model. The flux depends on the **Deposition Velocity ($V_d$)**, the **Leaf Area Index (LAI)**, and the ambient concentration.

$$
\text{Flux}_{[\mu g/s]} = V_d \cdot \text{LAI}_{\text{factor}} \cdot C_{\text{pollutant}} \cdot A_{\text{roof}}
$$

To get the hourly removal:
$$
\text{Removal}_{[\mu g/h]} = \text{Flux} \cdot 3600
$$

### 7.1 Deposition Parameters

| Pollutant | Base $V_d$ (m/s) | LAI Factor | Effective $V_d$ | Interpretation |
|-----------|------------------|------------|-----------------|----------------|
| **PM10** | $0.002$          | $2.0$      | $0.004$         | Coarse particles settle via gravity/impaction. |
| **PM2.5** | $0.0012$         | $2.0$      | $0.0024$        | Fine particles settle slower. |
| **CO** | $0.00005$        | $2.0$      | $0.0001$        | Gas uptake is diffusion-limited. |

**Interpretation:**
Unlike CO₂, which depends on light, particulate removal happens 24/7 but is most effective when pollution concentrations are high (linear dependency).

---

# PART III: Implementation and Output

## 8. Numerical Implementation

* **Roof Solver**: Explicit Euler with 60 substeps per hour to handle fast thermal dynamics.
* **Facade Solver**: A nested loop that integrates the parcel trajectory over its specific transit time.
* **Ecological Solver**: Calculated at each hourly step based on the current environmental state.
* **Coupling**: The roof output ($T_{air}$) becomes the initial condition for the facade input.
* **Spin-up procedure**: Simulation iterates for a few cycles to reach thermal equilibrium before recording output.

## 9. Outputs

The simulation produces:

1.  **Roof surface temperatures** (`T_cement`, `T_green`)
2.  **Air temperature at roof exit** (`T_air_end_cement`, `T_air_end_green`)
3.  **Air temperature at facade** (`T_air_fac_cement`, `T_air_fac_green`): The temperature after transit decay.
4.  **Mixing height** (`H_mix`)
5.  **Buoyant rise and floors affected** (`Rise_*`, `Floors_affected_*`)
6.  **Ecological Metrics** (Green Roof only): Mass of PM10/PM2.5/CO removed, CO2 sequestered, O2 produced.

## 10. Practical Relevance

This integrated model allows urban planners and engineers to:

* Compare **microclimate effects** of different roof types.
* Predict **thermal comfort** near building facades.
* Assess the influence of **wind and solar radiation** on roof and air temperatures.
* Quantify **buoyant air movement**, which affects ventilation and pollutant dispersion.
* Quantify **ecosystem services** provided by vegetation in terms of air purification.