# Roof Surface Temperature Model — Explanation and Interpretation

This model describes **how the temperature of a roof surface changes over time** when it is exposed to solar radiation, interacts with the air, and in some cases includes evaporative cooling (as with a green roof).

The core update equation used in the simulation is:

$$
T_{\text{roof}}(t + \Delta t) =
T_{\text{roof}}(t) +
\frac{\Delta t}{C}
\left[
\alpha G
- Q_{\text{evap}}
- h_c \left(T_{\text{roof}}(t) - T_{\text{air}}(t)\right)
\right]
$$

---

## Components of the Formula

| Symbol | Meaning | Units | Interpretation |
|-------|---------|--------|----------------|
| $ T_{\text{roof}} $ | Roof surface temperature | °C | What we are computing over time |
| $ T_{\text{air}} $ | Ambient air temperature | °C | Outside temperature influencing the roof |
| $ \alpha $ | Solar absorptivity of roof material | – (0–1) | How much sunlight the roof absorbs |
| $ G $ | Solar radiation (irradiance) | W/m² | Sun energy arriving per unit area |
| $ Q_{\text{evap}} $ | Evaporative cooling (for green roofs) | W/m² | Heat loss due to plant transpiration and water evaporation |
| $ h_c $ | Convective heat transfer coefficient | W/(m²·K) | Heat exchange efficiency between roof and air, increases with wind |
| $ C $ | Thermal inertia (heat capacity per unit area) | J/(m²·K) | How slowly the roof temperature changes when energy is added/lost |
| $ \Delta t $ | Time step | s | Length of each simulation update (e.g., 3600s per hour) |

---

## Physical Interpretation

The **net heat flux** influencing the roof is:

$$
Q_{\text{net}} = \alpha G - Q_{\text{evap}} - h_c (T_{\text{roof}} - T_{\text{air}})
$$

This can be broken down as:

### 1. **Solar Heating**: $ \alpha G $
- Sunlight provides energy.
- Roofs with darker surfaces (higher $ \alpha $) absorb more heat.
- Green roofs often have slightly higher $ \alpha $, but…

### 2. **Evaporative Cooling**: $ -Q_{\text{evap}} $
- Only present for green/vegetated roofs.
- Plants and water absorb heat and release it through evaporation (like sweating).
- This process **cools the roof**, lowering temperature.

### 3. **Convective Heat Exchange**: $ - h_c (T_{\text{roof}} - T_{\text{air}}) $
- If the roof is **hotter** than the air → it loses heat to the air.
- If the roof is **cooler** → it gains heat from the air.
- Higher wind → higher $ h_c $ → **faster heat equalization**.

---

## Why the Heat is Divided by the Thermal Inertia $ C $

$$
\frac{\Delta t}{C}
$$

This term determines **how fast the roof’s temperature changes**.

- **Large \(C\)** (concrete → heavy, dense, thick):  
  The roof **heats and cools slowly** → temperature changes are smooth.

- **Smaller \(C\)** (thin metal, low mass roofs):  
  The roof **responds quickly** to sunlight and air temperature.

Green roofs usually have **higher C** than bare concrete due to soil moisture → they change temperature **more slowly**, leading to **more stable temperature profiles**.

---

## Summary of Interactions

| Effect | Concrete Roof | Green Roof |
|-------|---------------|------------|
| Solar absorption | Heats surface | Heats surface |
| Evaporative cooling | **No** cooling (0 W/m²) | **Significant** cooling effect |
| Thermal inertia | Medium/High | High → smoother temperature curve |
| Convective exchange | Depends on wind | Depends on wind, same behavior |

Thus:

- A **concrete roof** *typically gets hotter at midday* and *retains heat longer into the night*.  
- A **green roof** stays *cooler during the day* and releases heat more gradually due to water evaporation and higher thermal mass.

---

## Practical Relevance

This formula captures the **dominant drivers** of roof temperature behavior in a **simple but physically meaningful way**, allowing us to:

- Simulate differences between roof materials
- Estimate heat released into air around buildings
- Understand temperature differences affecting nearby sensors or microclimate

This foundational model remains computationally light while preserving realistic physical behavior, which makes it well-suited for **urban energy balance studies** and **microclimate simulations**.
