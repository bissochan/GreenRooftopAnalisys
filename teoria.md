# Scenario
C'è un tetto piano che può essere di cemento o con uno strato d'erba (green roof). Il tetto ha un'area rettangolare definita da larghezza W e lunghezza L.
Alla fine del tetto, a una certa distanza D, c'è una facciata verticale di un altro edificio.

Un flusso d'aria entra da sinistra, passa sopra il tetto sempre nella stessa direzione e arriva alla facciata.
L'obiettivo è capire se l'aria che arriva alla facciata sarà più calda se il tetto è di cemento o green roof.


```
                              ____________
                              |
                              |
                       ↗  ↗  | f
     ---->   --->    ↗  ↗  ↗ | a
  ---->   ---->  --->  ↗ ↗   | c
   ________________           | a
   |     roof     |           | d
   |              |           | e
   |              |           |
   |              |           |
   |              |           |
   |              |           |
------------------------------------------
   |------L-------|
                  |-----D-----|

```

L'intero sistema è modellato a blocchi: **tetto**, **aria sopra il tetto**, **trasporto orizzontale dell'aria** e **facciata verticale**.

---

## Concetti fisici applicati

### Il tetto
Il tetto assorbe radiazione solare, che lo riscalda.
- La radiazione solare G viene moltiplicata per l'assorbanza α del materiale del tetto (cemento o erba).

Dal guadagno di energia **si sottrae**:
1. **Perdita per convezione** verso l'aria soprastante:
   - Dipende dalla differenza di temperatura tra tetto e aria.
   - Il coefficiente di scambio termico è h = 10 + 5*v, dove v è la velocità del vento.

2. **Perdita per evaporazione** (solo per green roof):
   - Dipende dalla radiazione solare G, dalla velocità del vento e dal coefficiente K_evap.
   - Formula: Q_evap = K_evap * G * (1 + 0.3 * v).

La variazione di temperatura dipende dall'energia netta divisa per la capacità termica C del tetto, che dipende dal materiale.

L'evoluzione della temperatura del tetto è modellata con un'**equazione differenziale ordinaria (ODE)** risolta tramite **Eulero esplicito**.

$$
T_{\text{roof}}(t+\Delta t) = T_{\text{roof}}(t) + \frac{\Delta t}{C} \left[\alpha G - Q_{\text{evap}} - h_c \left(T_{\text{roof}}(t) - T_{\text{air}}(t)\right)\right]
$$

- **Comportamento atteso:**
  - Tetto di cemento → temperature più alte durante il giorno, raffreddamento rapido di notte (dovuto alla capacità termica inferiore)
  - Green roof → temperature più basse di giorno, rilascio di calore più lento di notte.

---

### Aria sopra il tetto
L'aria sopra il tetto viene riscaldata per **convezione** dal tetto stesso.

- La temperatura dell'aria è considerata **omogenea** fino a una certa altezza, detta **altezza di miscelazione H_mix**.
- H_mix viene calcolata come: H_mix = sqrt(Kz * tau), dove:
  - Kz = costante di diffusività turbolenta verticale
  - tau = tempo che l'aria impiega a passare sopra il tetto = L / v

**Interpretazione semplice:**
- Vento forte → aria scorre veloce → H_mix più basso.
- Vento debole → aria ha più tempo per mescolarsi → H_mix più alto.

La temperatura dell'aria sopra il tetto si calcola aggiungendo un ΔT alla temperatura ambiente:

ΔT = (h * A_eff) / (m_dot * cp) * (T_surface - T_env) * 2

dove:
- h = coefficiente di scambio termico
- A_eff = area effettiva di scambio del tetto (W * L)
- m_dot = portata di massa dell'aria sopra il tetto = rho * v * A_cross, dove:
  - rho = densità dell'aria
  - v = velocità del vento
  - A_cross = sezione attraversata dal vento = W * H_mix
- cp = capacità termica dell'aria
- T_surface = temperatura del tetto
- T_env = temperatura ambiente
- 2 = fattore empirico per amplificare l'effetto della convezione

---

### Trasporto orizzontale e facciata
Dopo il tetto, l'aria si muove orizzontalmente fino alla facciata a distanza D.

- L'aria può **salire per effetto della spinta di Archimede (buoyancy)** se è più calda dell'ambiente:
  - a_z = g * max(T_air - T_env, 0) / (T_env + T0)
  - Altezza totale di salita: Rise = γ * 0.5 * a_z * tau^2 + H_mix

- τ = tempo che l'aria impiega a raggiungere la facciata = D / v
- γ = fattore di smorzamento che rappresenta dissipazione e attrito
- Viene calcolato anche il numero di piani “colpiti” dall’aria calda: Floors_affected = ceil(Rise / h_floor)

**Interpretazione semplice:**
- Aria più calda → Rise maggiore → più piani influenzati.
- Green roof → aria meno calda → Rise leggermente inferiore.
- H_mix garantisce un’altezza minima anche se l’aria è neutra o leggermente più fredda dell’ambiente.

---

### Sintesi del modello
- Blocchi principali: **tetto**, **aria sopra il tetto**, **trasporto orizzontale**, **facciata verticale**
- Fenomeni fisici considerati:
  - Riscaldamento solare
  - Evaporazione (green roof)
  - Convezione verso aria sovrastante
  - Miscelazione verticale (H_mix)
  - Salita per buoyancy
- Metodo numerico: **Eulero esplicito con sub-step per stabilità**
