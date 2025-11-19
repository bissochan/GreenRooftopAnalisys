# Synthetic Environmental Data Simulation
This document explains how the daily and hourly synthetic data are generated, the meaning of each introduced parameter, and the motivations behind the modeling choices.

---

## 1. Overview
The script produces realistic synthetic hourly values for **temperature**, **wind speed**, and **solar radiation**.
The process combines three components:

1. **A deterministic hourly trend**
2. **A stochastic residual process modeled as AR(1)**
3. **A scenario-level bias** applied once per simulated day

Together, these components allow the simulator to reproduce realistic variability while staying consistent with observed seasonal and hourly dynamics.

---

## 2. Deterministic Trend Component
A CSV file (`hourly_trend.csv`) provides the baseline hourly profiles:

- `Temp_Trend`
- `Wind_Trend`
- `Rad_Trend`

These represent the **mean expected value** for each hour based on historical data or domain knowledge. They capture systematic patterns such as:

- higher temperatures in mid-afternoon
- typical diurnal wind behaviour
- zero solar radiation during the night

The trend is the backbone of the simulation; stochastic components only introduce deviations around it.

---

## 3. Stochastic Innovation via GMM Models
For each environmental variable, a **Gaussian Mixture Model (GMM)** is loaded from disk.
These models describe the distribution of the underlying hourly stochastic innovations.

When simulating a value, the script samples a single innovation from the corresponding GMM:

- This allows **multi-modal distributions**, key for variables such as wind or radiation.
- It ensures innovations remain physically plausible, because they follow distributions extracted from real data.

GMM outputs feed into the AR(1) residual process.

---

## 4. AR(1) Residual Process
Each variable has an autoregressive structure:

$$
r_t = \phi \, r_{t-1} + \sigma \, \varepsilon_t
$$

Where:

- $ r_t $ is the residual at time *t*
- $ \phi $ controls the persistence (memory)
- $ \sigma $ scales the GMM innovation
- $ \varepsilon_t $ is a sample from the fitted GMM

### Meaning of parameters
| Variable | φ (phi) | σ (sigma) | Purpose |
|---------|---------|-----------|---------|
| temp | 0.9947 | 0.1033 | High persistence: temperature evolves smoothly |
| wind | 0.8921 | 0.4519 | Lower persistence, greater volatility |
| rad  | 0.9337 | 0.3581 | Persistent when daylight exists |

**phi** controls how similar consecutive hours are.
**sigma** controls the amplitude of the random hourly deviation.

### Error capping
Temperature and wind have a `max_err` value to prevent unrealistic jumps.
Radiation does not, because nighttime behavior already resets the residual to zero.

---

## 5. Scenario-Level Bias
Before simulating the hours, the model draws one “scenario bias” per variable:

- **Temperature bias**: Normal(0, 7.16)
  - Models warm or cold days

- **Wind bias**: Normal(0, 1.10)
  - Models breezy or calm days

- **Radiation gain**: Normal(1.0, 0.3), clipped to [0.4, 1.5]
  - Multiplies the daily radiation curve to simulate cloudiness or exceptionally clear days

This bias is added *once* and stays constant for the whole simulated day.
It ensures the synthetic day is internally coherent:
a cloudy day stays cloudy, a warm day stays warm, etc.

---

## 6. Radiation Night Reset
Solar radiation behaves fundamentally differently from temperature and wind.
During nighttime (`Rad_Trend < 10`), the simulator forces:

- radiation = 0
- residual radiation = 0

This ensures physically correct behavior (no radiation at night, no accumulation of errors).

---

## 7. Hourly Simulation Loop
For each hour:

1. Sample a GMM innovation for each variable
2. Update the AR(1) residual
3. Apply caps if necessary
4. Combine:
   - deterministic trend
   - scenario bias
   - residual
5. Enforce physical constraints:
   - wind ≥ 0
   - radiation ≥ 0
   - radiation = 0 at night

The output is a realistic synthetic dataset maintaining both structure and randomness.

---

## 8. Output
The simulator writes to:

```
data/csv/synthetic_data_scenario.csv
```

containing:

- hourly timestamp
- trend values
- synthetic values

This file is suitable for further analysis, model testing, or scenario generation.

---

## 9. Motivations Behind the Design Choices

### ✔ Separate trend + stochastic process
This keeps long-term structure and short-term noise independent, mimicking real climate behaviour.

### ✔ AR(1) residuals
Environmental variables show temporal autocorrelation; AR(1) reliably captures this without overfitting.

### ✔ GMM innovations
Real environmental data is rarely Gaussian. GMMs allow flexible, realistic sampling.

### ✔ Scenario-level bias
Introduces day-to-day variability while preserving hourly coherence.

### ✔ Night radiation reset
Ensures physical correctness and stabilizes model behaviour.

---

This structure gives a robust balance between realism, interpretability, and flexibility for future extensions such as climate scenario scaling, extreme event injection, or multi-day simulation.
