# District Heating Optimization Problem

## Goal

Minimize the total operational cost of meeting the district heating network's thermal demand by optimally dispatching three generation/conversion assets (Heat Pump, Condensing Boiler, CHP) and a Thermal Storage, while participating in the electricity day-ahead (DA) market.

The objective function includes:
- **Gas costs** for Boiler and CHP (fuel input × gas price + CO2 costs)
- **Electricity costs** for the Heat Pump (electricity consumption × DA price)
- **Electricity revenue** from CHP (electricity production × DA price)

---

## Inputs

### Time-Series Data (Exogenous)

| Input | Unit | Source |
|---|---|---|
| Heat demand (thermal power to be supplied) | MW_th | Forecasted |
| Day-ahead electricity price (DA Price) | €/MWh_el | Market data (given) |

### Fixed Parameters

#### Gas Grid
| Parameter | Value |
|---|---|
| Gas Price | 35 €/MWh_Hs |
| CO2 Factor | 0.201 tCO2/MWh_Hs |
| CO2 Price | 60 €/tCO2 |

#### Thermal Storage
| Parameter | Value |
|---|---|
| Storage Capacity | 200 MWh_th |
| Max. Charging Power | 15 MW_th |
| Max. Discharging Power | 15 MW_th |
| Loss | 0.000125 MWh_th per 15-min interval |
| Initial State of Charge | 200 MWh_th |

#### Large-Scale Heat Pump
| Parameter | Value |
|---|---|
| Rated Electrical Power | 8 MW_el |
| Min. Electrical Power | 1 MW_el |
| COP (Coefficient of Performance) | 3.5 MW_th/MW_el |

#### Condensing Boiler
| Parameter | Value |
|---|---|
| Rated Thermal Power | 12 MW_th |
| Min. Load | 2 MW_th |
| Efficiency | 0.97 MW_th/MW_Hs |
| Minimum Run Time | 1 hour |
| Minimum Downtime | 1 hour |

#### CHP (Combined Heat and Power)
| Parameter | Value |
|---|---|
| Electrical Power max. | 6 MW_el |
| Electrical Power min. | 2 MW_el |
| Thermal Power max. | 7.2 MW_th (*) |
| Electrical Efficiency | 0.4 MW_el/MW_Hs |
| Thermal Efficiency | 0.48 MW_th/MW_Hs |
| Overall Efficiency | 0.88 |
| Minimum Run Time | 2 hours (8 intervals) |
| Minimum Downtime | 2 hours (8 intervals) |
| Start-up Costs | 600 € |

(*) The drawio diagram lists Thermal Power max. as 7 MW_th. We use 7.2 MW_th here because it is the value derived from the efficiency ratio: `η_th / η_el × P_el_max = 0.48 / 0.4 × 6 = 7.2 MW_th`. The diagram value appears to be rounded. This should be reconciled with the drawio as the canonical source of truth.

---

## Constraints

### Time Resolution & Horizon

- **Time resolution**: 15-minute intervals, `dt = 0.25 h`
- **Optimization horizon**: 24 hours = 96 intervals, `t ∈ {1, 2, ..., 96}`
- **Demand input**: Hourly forecast (24 values), held constant across the four 15-min intervals within each hour
- **DA electricity price**: Hourly (24 values), held constant across the four 15-min intervals within each hour

### Sets & Indices

- `t ∈ {1, 2, ..., T}` where `T = 96` (time intervals)
- `dt = 0.25 h` (interval duration)

### Decision Variables

| Variable | Type | Unit | Description |
|---|---|---|---|
| `z_hp(t)` | Binary | {0,1} | Heat pump on/off |
| `z_boiler(t)` | Binary | {0,1} | Boiler on/off |
| `z_chp(t)` | Binary | {0,1} | CHP on/off |
| `s_chp(t)` | Binary | {0,1} | CHP startup indicator |
| `d_chp(t)` | Binary | {0,1} | CHP shutdown indicator |
| `s_boiler(t)` | Binary | {0,1} | Boiler startup indicator |
| `d_boiler(t)` | Binary | {0,1} | Boiler shutdown indicator |
| `y_sto(t)` | Binary | {0,1} | Storage mode: 1 = charging, 0 = discharging |
| `P_hp_el(t)` | Continuous | MW_el | Heat pump electrical input |
| `Q_hp(t)` | Continuous | MW_th | Heat pump thermal output |
| `Q_boiler(t)` | Continuous | MW_th | Boiler thermal output |
| `F_boiler(t)` | Continuous | MW_Hs | Boiler fuel input |
| `P_chp_el(t)` | Continuous | MW_el | CHP electrical output |
| `Q_chp(t)` | Continuous | MW_th | CHP thermal output |
| `F_chp(t)` | Continuous | MW_Hs | CHP fuel input |
| `Q_charge(t)` | Continuous | MW_th | Storage charging power |
| `Q_discharge(t)` | Continuous | MW_th | Storage discharging power |
| `SoC(t)` | Continuous | MWh_th | Storage state of charge |

### Initial Conditions

For the **first optimization run**, all units are assumed off:
```
z_hp(0) = 0,  z_boiler(0) = 0,  z_chp(0) = 0
```
For **subsequent runs** (rolling horizon), initial states and durations-in-current-state are carried over from the previous solution.

---

### 1. Heat Demand (Hard Constraint)

Heat demand must be met at every time step `t`:

```
Q_hp(t) + Q_boiler(t) + Q_chp(t) + Q_discharge(t) - Q_charge(t) = Q_demand(t)    ∀t
```

All assets can operate in any combination (not mutually exclusive).

---

### 2. Thermal Storage

**Bounds:**
```
0 ≤ SoC(t) ≤ 200 MWh_th                          ∀t
0 ≤ Q_charge(t) ≤ 15 · y_sto(t) MW_th             ∀t
0 ≤ Q_discharge(t) ≤ 15 · (1 - y_sto(t)) MW_th    ∀t
```

The binary variable `y_sto(t)` prevents simultaneous charging and discharging: when `y_sto(t) = 1`, only charging is possible; when `y_sto(t) = 0`, only discharging is possible.

**State of charge dynamics:**
```
SoC(t) = SoC(t-1) + Q_charge(t) · 0.25 - Q_discharge(t) · 0.25 - 0.000125    ∀t
```

- `0.25` converts power (MW) to energy (MWh) per 15-min interval
- `0.000125 MWh_th` is the constant thermal loss per 15-min interval

**Initial condition:**
```
SoC(0) = 200 MWh_th
```

**No terminal constraint** — the optimizer is free to end the day at any SoC. For rolling horizon operation, the final SoC becomes the initial condition for the next day.

---

### 3. Heat Pump

**On/off logic (big-M formulation):**
```
z_hp(t) · 1 ≤ P_hp_el(t) ≤ z_hp(t) · 8          ∀t
```

When `z_hp(t) = 0`: `P_hp_el(t) = 0` (forced off).
When `z_hp(t) = 1`: `1 MW_el ≤ P_hp_el(t) ≤ 8 MW_el`.

**Thermal output (fixed COP):**
```
Q_hp(t) = 3.5 · P_hp_el(t)                        ∀t
```

When ON: `3.5 MW_th ≤ Q_hp(t) ≤ 28 MW_th`.

**No minimum run time or downtime** — the heat pump can switch freely between intervals.

**Cost:**
```
C_hp(t) = P_hp_el(t) · DA_price(t) · dt            ∀t
```

---

### 4. Condensing Boiler

**On/off logic (big-M formulation):**
```
z_boiler(t) · 2 ≤ Q_boiler(t) ≤ z_boiler(t) · 12    ∀t
```

When `z_boiler(t) = 0`: `Q_boiler(t) = 0`.
When `z_boiler(t) = 1`: `2 MW_th ≤ Q_boiler(t) ≤ 12 MW_th`.

**Fuel consumption:**
```
F_boiler(t) = Q_boiler(t) / 0.97                      ∀t
```

**Startup and shutdown indicators:**
```
s_boiler(t) ≥ z_boiler(t) - z_boiler(t-1)             ∀t
d_boiler(t) ≥ z_boiler(t-1) - z_boiler(t)             ∀t
s_boiler(t), d_boiler(t) ∈ {0, 1}
```

`s_boiler(t) = 1` when the boiler turns on at `t`. `d_boiler(t) = 1` when it turns off at `t`.

**Minimum run time — 1 hour (4 intervals):**
```
∑_{i=max(1,t-3)}^{t} s_boiler(i) ≤ z_boiler(t)       ∀t
```

Meaning: if the boiler started at any point in the last 4 intervals (or fewer at the start of the horizon), it must still be on now.

**Minimum downtime — 1 hour (4 intervals):**
```
∑_{i=max(1,t-3)}^{t} d_boiler(i) ≤ 1 - z_boiler(t)   ∀t
```

Meaning: if the boiler shut down at any point in the last 4 intervals (or fewer at the start of the horizon), it must still be off now.

**Cost:**
```
C_boiler(t) = F_boiler(t) · (35 + 0.201 × 60) · dt = F_boiler(t) · 47.06 · dt    ∀t
```

No startup cost specified for the boiler (assumed zero).

---

### 5. CHP (Combined Heat and Power)

**On/off logic (big-M formulation):**
```
z_chp(t) · 2 ≤ P_chp_el(t) ≤ z_chp(t) · 6           ∀t
```

When `z_chp(t) = 0`: `P_chp_el(t) = 0`.
When `z_chp(t) = 1`: `2 MW_el ≤ P_chp_el(t) ≤ 6 MW_el`.

**Thermal output (fixed back-pressure coupling):**
```
Q_chp(t) = (η_th / η_el) · P_chp_el(t) = 1.2 · P_chp_el(t)    ∀t
```

When ON: `2.4 MW_th ≤ Q_chp(t) ≤ 7.2 MW_th`.

**Fuel consumption:**
```
F_chp(t) = P_chp_el(t) / 0.4                          ∀t
```

Cross-check at full load: `F = 6/0.4 = 15 MW_Hs`, `(P_el + Q_th) / F = (6 + 7.2) / 15 = 0.88`. ✓

**Startup and shutdown indicators:**
```
s_chp(t) ≥ z_chp(t) - z_chp(t-1)                     ∀t
d_chp(t) ≥ z_chp(t-1) - z_chp(t)                     ∀t
s_chp(t), d_chp(t) ∈ {0, 1}
```

**Minimum run time — 2 hours (8 intervals):**
```
∑_{i=max(1,t-7)}^{t} s_chp(i) ≤ z_chp(t)             ∀t
```

Meaning: if the CHP started at any point in the last 8 intervals (or fewer at the start of the horizon), it must still be on now.

**Minimum downtime — 2 hours (8 intervals):**
```
∑_{i=max(1,t-7)}^{t} d_chp(i) ≤ 1 - z_chp(t)         ∀t
```

Meaning: if the CHP shut down at any point in the last 8 intervals (or fewer at the start of the horizon), it must still be off now.

For subsequent rolling horizon runs, the carried-over state and duration determine how many intervals of the minimum up/downtime constraint remain binding at the start of the horizon.

**Startup cost:**
```
C_chp_start(t) = 600 · s_chp(t)                       ∀t
```

**Operating cost:**
```
C_chp_fuel(t) = F_chp(t) · 47.06 · dt                 ∀t
```

**Revenue:**
```
R_chp(t) = P_chp_el(t) · DA_price(t) · dt             ∀t
```

---

### 6. Grid Connection

- The Heat Pump consumes electricity from the grid: priced at `DA_price(t)`
- The CHP produces electricity sold to the grid: priced at `DA_price(t)`
- **No grid connection capacity limit** — the grid can supply/absorb any amount
- Gas consumed by Boiler and CHP is priced at the gas grid price plus CO2 costs: `35 + 0.201 × 60 = 47.06 €/MWh_Hs`

---

### 7. Objective Function

Minimize total daily cost:

```
min ∑_{t=1}^{96} [ C_hp(t) + C_boiler(t) + C_chp_fuel(t) + C_chp_start(t) - R_chp(t) ]
```

Expanded:
```
min ∑_{t=1}^{96} [
    P_hp_el(t) · DA_price(t) · 0.25                          (heat pump electricity cost)
  + (Q_boiler(t) / 0.97) · 47.06 · 0.25                      (boiler fuel + CO2 cost)
  + (P_chp_el(t) / 0.4) · 47.06 · 0.25                       (CHP fuel + CO2 cost)
  + 600 · s_chp(t)                                            (CHP startup cost)
  - P_chp_el(t) · DA_price(t) · 0.25                          (CHP electricity revenue)
]
```

---

## Notes

- The problem is a **Mixed-Integer Linear Program (MILP)** due to binary on/off decisions, minimum up/down time constraints, and startup cost modeling
- Heat demand is a **forecasted** input — forecast quality directly impacts optimization results
- DA prices are hourly (24 values/day) — each value is held constant across 4 intervals within the hour
- The demand forecast is hourly (24 values/day) — same treatment as DA prices
- For rolling horizon operation: final states (SoC, unit on/off status, time-in-state) become initial conditions for the next day's optimization
