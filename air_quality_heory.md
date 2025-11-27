# Air Quality Synthetic Data Simulation

This document explains how the daily and hourly synthetic air quality data are generated, the meaning of each introduced parameter, the ecological removal modeling, and the motivations behind the choices.

---

## 1. Overview

The air quality simulation pipeline consists of five stages:

1. **Data Analysis** (analyze_data.py) — extract trends, residuals, and tuning parameters from raw observations
2. **Model Building** (build_model.py) — fit GMM models to residuals for stochastic innovation sampling
3. **Stochastic Day Generation** (`simulation/generate_stochastic_day_eco()`) — create realistic synthetic hourly concentrations
4. **Ecological Removal Computation** (ecology.py) — calculate pollutant removal via green roof processes
5. **Evaluation & Validation** (air_quality_eval.py) — compare simulated vs. real data and assess model performance

Together, these components allow the simulator to:
- Reproduce observed pollutant dynamics (concentration, variability, temporal structure)
- Quantify ecosystem services (CO₂ uptake, PM/gas removal)
- Support scenario analysis and green infrastructure design

---

## 2. Data Analysis Stage (`analyze_data.py`)

### 2.1 Input
Raw air quality observations from air_quality.csv:
- **Variables**: CO₂ (ppm), PM10 (µg/m³), PM2.5 (µg/m³), CO (µg/m³)
- **Format**: Hourly timestamps with occasional NaN values
- **Time coverage**: Historical period (e.g., 1 year of data)

### 2.2 Trend Extraction
For each variable, compute the **hourly mean profile**:

$$
\text{Trend}_h = \frac{1}{N_{\text{days}}} \sum_{d=1}^{N_{\text{days}}} C(d, h)
$$

Where $C(d,h)$ is the concentration for day $d$ at hour $h$.

**Saved as**: air_quality_hourly_trend.csv

**Columns**: 
- `CO2_Trend`, `PM10_Trend`, `PM2_5_Trend`, `CO_Trend`

**Interpretation**: 
- These represent the **typical diurnal cycle** (e.g., CO peaks during rush hours, PM settles overnight)
- Used as the deterministic backbone for synthetic generation

### 2.3 Residual Extraction
Compute residuals as deviations from the trend:

$$
\text{Residual}_t = C_t - \text{Trend}_{h(t)}
$$

Where $t$ is hour index and $h(t)$ is the hour of day for time $t$.

**Saved as**: air_quality_residuals.csv

**Purpose**: 
- Capture day-to-day and hour-to-hour variability not explained by the diurnal pattern
- Feed into AR(1) models and GMM fitting

### 2.4 AR(1) Tuning Parameter Extraction
For each variable, fit a first-order autoregressive model:

$$
r_t = \phi \, r_{t-1} + \sigma \, \varepsilon_t
$$

Using Yule-Walker or MLE to estimate:
- **φ (phi)**: autocorrelation lag-1 coefficient (persistence)
- **σ (sigma)**: noise scale
- **bias**: daily scenario-level bias standard deviation

**Saved as**: air_quality_tuning_params.csv

**Example values** (from your data):

| Variable | φ | σ | bias |
|----------|---|---|------|
| CO₂ | 0.9753 | 0.2208 | 19.5587 |
| PM10 | 0.9733 | 0.2296 | 8.4735 |
| PM2.5 | 0.9732 | 0.2299 | 6.1579 |
| CO | 0.9677 | 0.2520 | 77.8994 |

**Interpretation**:
- High φ (0.96–0.97) → concentrations persist across hours (air mass inertia)
- σ scales the stochastic shocks
- bias represents warm/cold or polluted/clean day scenarios

---

## 3. GMM Model Building (`build_model.py`)

### 3.1 Rationale
Raw residuals from real air quality data are rarely Gaussian. A **Gaussian Mixture Model** allows:
- Multi-modal distributions (e.g., rush-hour CO peak vs. evening decay)
- Capture of tail behavior (extreme pollution events)
- Physical plausibility when sampled

### 3.2 Process
For each pollutant residual series:

1. **Component selection**: Fit GMMs with k=1,2,...,6 components; select k using BIC
2. **Parameter estimation**: Maximum likelihood (sklearn.mixture.GaussianMixture)
3. **Serialization**: Save fitted model to `data/pkl_models/{var}_gmm.pkl`

**Outputs**: `co2_gmm.pkl`, `pm10_gmm.pkl`, `pm2_5_gmm.pkl`, `co_gmm.pkl`

### 3.3 BIC Selection
BIC balances model fit vs. complexity:

$$
\text{BIC}(k) = -2 \ln \hat{L}(k) + k \log(N)
$$

The script produces diagnostic plots (`co2_bic.png`, etc.) showing which k minimizes BIC.

**Example**: If PM10 residuals show two distinct modes (background + traffic spikes), BIC may select k=2.

---

## 4. Stochastic Day Generation (`generate_stochastic_day_eco()`)

### 4.1 Structure (analogous to weather)
Each simulated day combines:
1. **Deterministic hourly trend** (from analyze_data)
2. **Scenario-level bias** (day-to-day variability)
3. **AR(1) residual process** (hour-to-hour dynamics)

### 4.2 Scenario Bias
Before the hourly loop, sample **one bias per variable**:

$$
\text{bias}_{\text{CO}_2} \sim N(0, \sigma_{\text{bias,CO}_2}^2) = N(0, 19.56^2)
$$

Similarly for PM10, PM2.5, CO.

**Purpose**: 
- Models polluted vs. clean days
- Ensures internal coherence (whole day is shifted up/down consistently)

### 4.3 Hourly Loop
For each hour $h$ (0–23):

1. **Sample GMM innovation** $\varepsilon_h$ from the fitted GMM model
2. **Update AR(1) residual**:
   $$r_h = \phi \, r_{h-1} + \sigma \, \varepsilon_h$$
3. **Clip if necessary**: Cap residuals to $[\text{−max\_err}, \text{+max\_err}]$ (e.g., ±50 ppm for CO₂)
4. **Combine components**:
   $$C_h = \text{Trend}_h + \text{bias} + r_h$$
5. **Enforce constraints**:
   - CO₂: clip to [300, 800] ppm (realistic range)
   - PM10, PM2.5: clip to ≥ 0.1 µg/m³
   - CO: clip to ≥ 0.1 µg/m³

### 4.4 Output
For each Monte Carlo run (100 runs):
- 24 hourly records per variable (CO₂, CO, PM10, PM2.5)
- Saved to weather_samples_eco.csv

---

## 5. Ecological Removal Modeling (`model/ecology.py`)

### 5.1 Rationale
Green roofs remove pollutants via distinct mechanisms:

| Pollutant | Mechanism | Dependency |
|-----------|-----------|-----------|
| CO₂ | Photosynthesis | Radiation (light), ambient concentration |
| PM10, PM2.5 | Dry deposition on leaves | Ambient concentration, LAI, wind |
| CO | Stomatal uptake + microbial oxidation | Ambient concentration, LAI |

### 5.2 CO₂ Removal (Photosynthesis)

Three components sum to total removal:

$$
\text{CO}_2^{\text{removed}} = \text{CO}_2^{\text{light}} + \text{CO}_2^{\text{basal}} + \text{CO}_2^{\text{conc}}
$$

**Light-driven term** (primary):
$$
\text{CO}_2^{\text{light}} = k_{\text{photo}} \cdot \frac{R}{100} \cdot A
$$
- $k_{\text{photo}} = 0.35$ g CO₂/(m²·h) per (W/m²)/100
- $R$ = solar radiation (W/m²)
- $A = 600$ m² = green roof area

**Basal (nighttime) term**:
$$
\text{CO}_2^{\text{basal}} = k_{\text{basal}} \cdot A = 0.005 \text{ g/(m}^2\text{·h)} \cdot 600 \text{ m}^2 = 3 \text{ g/h}
$$
- Accounts for respiration-driven uptake (minimal but non-zero)

**Concentration-dependent term** (optional):
$$
\text{CO}_2^{\text{conc}} = k_{\text{conc}} \cdot C_{\text{CO}_2} \cdot A
$$
- $k_{\text{conc}} = 1 \times 10^{-3}$
- $C_{\text{CO}_2}$ in ppm
- Small feedback: higher ambient CO₂ → slightly higher uptake

**O₂ Production**:
$$
\text{O}_2^{\text{prod}} = \text{CO}_2^{\text{removed}} \times 0.72
$$
(stoichiometric: ~0.72 g O₂ per g CO₂ fixed)

### 5.3 Particulate Matter Removal (Deposition)

Uses a **deposition velocity** model:

$$
\text{Flux} = V_d \cdot C \cdot A \quad [\mu\text{g/s}]
$$

Where:
- $V_d$ = deposition velocity (m/s)
- $C$ = ambient concentration (µg/m³)
- $A$ = area (m²)

**Conversion to hourly**:
$$
\text{Removal}_{[\mu\text{g/h}]} = V_d \cdot C \cdot A \cdot 3600
$$

#### PM10
$$
V_{d,\text{PM10}} = 2 \times 10^{-3} \text{ m/s}
$$
- Adjusted by LAI factor (2.0): $V_d^{\text{eff}} = 2 \times 10^{-3} \times 2.0 = 4 \times 10^{-3}$ m/s

#### PM2.5
$$
V_{d,\text{PM25}} = 1.2 \times 10^{-3} \text{ m/s}
$$
- Smaller than PM10 (finer particles deposit slower)

**Example calculation** (PM10 at C=25 µg/m³):
$$
\text{Flux} = 4 \times 10^{-3} \times 25 \times 600 = 60 \text{ µg/s} = 216{,}000 \text{ µg/h}
$$

#### CO (Gas)
$$
V_{d,\text{CO}} = 5 \times 10^{-5} \text{ m/s}
$$
- Much slower than particles (gaseous, diffusion-limited)
- Removal is minimal but non-zero

### 5.4 Handling Missing Data
All functions check for NaN/None:
```python
if co2 is None or np.isnan(co2):
    co2_conc_term = 0.0
```
Ensures robust handling of incomplete sensor records.

---

## 6. Monte Carlo Simulation Loop (`run_simulation_eco.py`)

### 6.1 Process
Repeat N=100 times:

1. Generate one stochastic day (`generate_stochastic_day_eco()`)
2. For each hour, compute removals (`compute_ecology()`)
3. Record: concentrations, removals, cumulative loads

### 6.2 Output Files

**`weather_samples_eco.csv`**:
- 2,400 rows (100 runs × 24 hours)
- Columns: `run_id`, `hour`, `CO2`, `CO`, `PM10`, `PM25`, `Radiation`

**`simulation_results_eco.csv`**:
- 2,400 rows
- Columns: `run_id`, `hour`, `CO2`, `CO`, `PM10`, `PM25`, `CO2_Removed_g`, `CO_Removed_ug`, `PM10_Removed_ug`, `PM25_Removed_ug`, `O2_Produced_g`

**`ecology_summary_mc.json`**:
- Summary statistics (mean, std, min, max for daily totals)

### 6.3 Why Monte Carlo?
- Uncertainty quantification: spreads across 100 plausible days
- Captures full distribution of removal services
- Supports risk/robustness analysis (e.g., "worst case removal", "typical scenario")

---

## 7. Evaluation & Validation (`air_quality_eval.py`)

### 7.1 Data Loading
Loads and aligns:
- Simulated concentrations (`weather_samples_eco.csv`)
- Simulated removals (`simulation_results_eco.csv`)
- Real air quality data (air_quality.csv)
- Historical trends (air_quality_hourly_trend.csv)

### 7.2 Diagnostic Plots

#### `{var}_mean.png`
- **X-axis**: hour of day
- **Y-axis**: concentration (ppm or µg/m³)
- **Lines**: trend (black dashed), real mean (green), simulated mean (red ± 95% CI)
- **Use**: Detect systematic bias between simulated and observed hourly profiles

#### `{var}_std.png`
- **Y-axis**: standard deviation
- **Lines**: real std (green), simulated std (purple dashed)
- **Use**: Assess whether model captures observed variability; under-dispersion → too smooth; over-dispersion → too noisy

#### `{var}_ecdf.png`
- **X-axis**: concentration values
- **Y-axis**: cumulative probability (0–1)
- **Lines**: empirical CDF for real (green) vs. simulated (purple)
- **Use**: Compare overall distributions; divergence in tails indicates poor extreme event modeling

#### `{var}_qq.png`
- **X-axis**: theoretical quantiles (standard normal)
- **Y-axis**: sample quantiles (simulated data)
- **Use**: Check normality assumption; S-shaped deviations indicate heavy tails or skewness

#### `{var}_removal_hourly.png`
- **Y-axis**: removal rate (g/h for CO₂, µg/h for others)
- **Lines**: mean removal ± 1 std by hour
- **Use**: Visualize when ecosystem services are strongest (e.g., CO₂ removal peaks during daylight)

#### `{var}_daily_totals.png`
- **X-axis**: daily total removal (g/day or µg/day)
- **Y-axis**: frequency across 100 MC runs
- **Statistics**: mean, median lines
- **Use**: Assess variability and range of removal services; wide distribution → high uncertainty

#### `{var}_cumulative.png`
- **X-axis**: hour of day
- **Y-axis**: cumulative removal (g or µg)
- **Lines**: 30 sample runs (faint gray) + mean (red bold)
- **Use**: Show accumulation pattern; e.g., CO₂ removal rises during daylight, plateaus at night

### 7.3 Numerical Metrics (in report)

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Mean Bias | $\frac{1}{24} \sum_h (\text{sim}_h - \text{trend}_h)$ | Signed systematic offset |
| RMSE | $\sqrt{\frac{1}{24} \sum_h (\text{sim}_h - \text{real}_h)^2}$ | Overall fit error |
| CV (Real vs Sim) | $\frac{\sigma}{\mu}$ for each | Coefficient of variation (relative spread) |
| KS statistic | Kolmogorov–Smirnov test | Distance between empirical CDFs; p-value < 0.05 → significant difference |
| Percentiles (P05, P50, P95, P99) | Quantile comparison | Distribution shape; mismatches in tails flag poor extreme modeling |

### 7.4 Common Issues & Fixes

**Issue**: Simulated mean is consistently higher/lower than real by a fixed amount
- **Cause**: Bias term in AR(1) tuning or trend offset
- **Fix**: Inspect `air_quality_tuning_params.csv` bias column; recalibrate or apply post-hoc correction

**Issue**: Simulated std much smaller than real
- **Cause**: σ parameter too small, or GMM not capturing multi-modal behavior
- **Fix**: Refit GMM with more components; increase σ

**Issue**: ECDF diverges in tails (P99 too high/low)
- **Cause**: AR(1) max_err cap too aggressive, or GMM doesn't sample extremes
- **Fix**: Relax max_err; refit GMM allowing heavier tails

**Issue**: Removal rates constant (no variability)
- **Cause**: compute_ecology using fixed coefficients instead of concentration-dependent terms
- **Fix**: Verify that Vd and LAI scaling are active; check that conc is passed to function

---


## 8. Extension Points for Future Work

| Extension | Impact | Difficulty |
|-----------|--------|------------|
| Wind speed dependency for Vd | More realistic PM removal | Medium |
| Wet deposition (rain events) | Capture scavenging episodes | Medium |
| Seasonal LAI variation | Account for dormancy/leafout | Low |
| Multi-day correlation | Simulate pollution episodes | High |
| Green roof type/design parameters | Species-specific efficiency | Low–Medium |
| Urban heat island feedback | Temperature–pollution coupling | High |

---

## 9. Summary

The air quality simulation pipeline transforms raw observations into:

1. **Stochastic synthetic days** with realistic temporal structure and day-to-day coherence
2. **Quantified ecosystem services** (CO₂ sequestration, particulate removal, O₂ production)
3. **Uncertainty estimates** via Monte Carlo sampling
4. **Validated predictions** with comprehensive diagnostic plots

This foundation supports:
- Green infrastructure design and optimization
- Climate scenario analysis
- Ecosystem service quantification
- Decision support for urban planners and policymakers