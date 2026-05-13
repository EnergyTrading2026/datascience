# District Heating Optimization Problem

> **Note:** This document defines the MILP formulation (decision variables, constraints, objective, parameters). The production system runs an hourly MPC variant of this problem — see [`hourly_mpc.md`](hourly_mpc.md) for the architecture, rolling-horizon mechanics, and orchestration. Where the two documents disagree on horizon length or rolling-horizon cadence, `hourly_mpc.md` is current.

> **Plant config (since `feat/modular-assets`):** The numeric values listed below describe the *legacy* 1-of-each plant. They live in code as `PlantConfig.legacy_default()` (`src/optimization/config.py`) and are the default when no `--config-file` is passed. Per-asset modularity is enabled in code: a plant may carry 0..N of each family (heat pumps, boilers, CHPs, storages), each with its own `id` and physics. Runtime input is `plant_config.json`, generated via `optimization-write-default-config <path>` and then hand-edited by the operator. The MILP formulation below generalizes to multi-asset by indexing every variable and constraint over the per-family asset set.

MILP for 24 h dispatch of Heat Pump, Condensing Boiler, CHP, and Thermal Storage against day-ahead (DA) electricity prices. Minimizes gas + CO2 + electricity cost net of CHP electricity revenue.

## Time Structure

- Step: 15 min, `dt = 0.25 h`
- Horizon: 24 h = 96 steps, `t = 1..96`
- `demand_th(t)` and `price_el(t)`: hourly forecasts (24 values), held constant across the 4 steps within each hour

## Inputs

### Time series

| Name | Unit | Notes |
|---|---|---|
| `demand_th(t)` | MW_th | hourly forecast, broadcast to 4 × 15-min steps |
| `price_el(t)`  | EUR/MWh_el | hourly DA price, broadcast to 4 × 15-min steps |

### Gas + CO2

| Param | Value |
|---|---|
| Gas price | 35 EUR/MWh_Hs |
| CO2 factor | 0.201 tCO2/MWh_Hs |
| CO2 price | 60 EUR/tCO2 |
| **Effective fuel cost** | `35 + 0.201 × 60 = 47.06 EUR/MWh_Hs` |

### Thermal storage

| Param | Value |
|---|---|
| Capacity | 200 MWh_th |
| Hard floor (every `t`) | 50 MWh_th |
| Max charge / discharge | 15 / 15 MW_th |
| Loss | 0.000125 MWh_th per 15-min step |
| Initial SoC | 200 MWh_th |

### Heat pump

| Param | Value |
|---|---|
| Electrical power | 1–8 MW_el |
| COP | 3.5 MW_th / MW_el |
| Min run / downtime | none |

### Condensing boiler

| Param | Value |
|---|---|
| Thermal output | 2–12 MW_th |
| Efficiency | 0.97 MW_th / MW_Hs |
| Min run / downtime | 1 h / 1 h (4 steps each) |
| Startup cost | none |

### CHP

| Param | Value |
|---|---|
| Electrical output | 2–6 MW_el |
| Thermal output | 2.4–7.2 MW_th (= 1.2 × `chp_el_out`) |
| η_el / η_th | 0.40 / 0.48 |
| Min run / downtime | 2 h / 2 h (8 steps each) |
| Startup cost | 600 EUR |

CHP thermal max derived as `(η_th / η_el) · P_el_max = 0.48 / 0.40 · 6 = 7.2 MW_th`. Sanity check at full load: `(P_el + Q_th) / F = (6 + 7.2) / (6 / 0.4) = 13.2 / 15 = 0.88` overall efficiency. ✓

## Decision Variables

### Binary
- `hp_on(t)`, `boiler_on(t)`, `chp_on(t)`
- `boiler_start(t)`, `boiler_stop(t)`
- `chp_start(t)`, `chp_stop(t)`
- `sto_mode_charge(t)` — 1 = charging, 0 = discharging

### Continuous (≥ 0)
- HP: `hp_el_in(t)`, `hp_th_out(t)`
- Boiler: `boiler_th_out(t)`, `boiler_fuel_in(t)`
- CHP: `chp_el_out(t)`, `chp_th_out(t)`, `chp_fuel_in(t)`
- Storage: `sto_th_charge(t)`, `sto_th_discharge(t)`, `sto_th_soc(t)`

## Initial Conditions

Cold start (first run):
```
hp_on(0) = boiler_on(0) = chp_on(0) = 0
sto_th_soc(0) = 200
```

Rolling horizon: previous-day final SoC, on/off states, and time-in-current-state are carried over as initial conditions for the next day. Time-in-state matters because partial min-run / min-down obligations from the previous day must be honored at the start of the new horizon.

## Constraints

### 1. Heat balance (every `t`)
```
hp_th_out(t) + boiler_th_out(t) + chp_th_out(t)
  + sto_th_discharge(t) - sto_th_charge(t)  =  demand_th(t)
```

All assets can run in any combination.

### 2. Storage

Bounds:
```
50 <= sto_th_soc(t)       <= 200
 0 <= sto_th_charge(t)    <= 15 * sto_mode_charge(t)
 0 <= sto_th_discharge(t) <= 15 * (1 - sto_mode_charge(t))
```

`sto_mode_charge` blocks simultaneous charge / discharge.

Dynamics:
```
sto_th_soc(t) = sto_th_soc(t-1)
              + 0.25 * sto_th_charge(t)
              - 0.25 * sto_th_discharge(t)
              - 0.000125
```

**No terminal constraint and no soft penalty.** The `>= 50` floor at every `t` already prevents end-of-horizon draining; ablation in `all_in_one_evaluation_notebook.ipynb` showed the soft-shortfall term adds no value once the hard floor is in place.

### 3. Heat pump
```
hp_on(t) * 1 <= hp_el_in(t) <= hp_on(t) * 8
hp_th_out(t) = 3.5 * hp_el_in(t)
```

No min run / downtime — HP can switch every step.

### 4. Boiler
```
boiler_on(t) * 2 <= boiler_th_out(t) <= boiler_on(t) * 12
boiler_fuel_in(t) = boiler_th_out(t) / 0.97
```

Startup / shutdown indicators:
```
boiler_start(t) >= boiler_on(t)   - boiler_on(t-1)
boiler_stop(t)  >= boiler_on(t-1) - boiler_on(t)
```

Min run (1 h = 4 steps) and min down (1 h = 4 steps):
```
sum_{i=max(1, t-3)}^{t} boiler_start(i) <= boiler_on(t)
sum_{i=max(1, t-3)}^{t} boiler_stop(i)  <= 1 - boiler_on(t)
```

No startup cost.

### 5. CHP
```
chp_on(t) * 2 <= chp_el_out(t) <= chp_on(t) * 6
chp_th_out(t)  = 1.2 * chp_el_out(t)
chp_fuel_in(t) = chp_el_out(t) / 0.4
```

Startup / shutdown indicators:
```
chp_start(t) >= chp_on(t)   - chp_on(t-1)
chp_stop(t)  >= chp_on(t-1) - chp_on(t)
```

Min run (2 h = 8 steps) and min down (2 h = 8 steps):
```
sum_{i=max(1, t-7)}^{t} chp_start(i) <= chp_on(t)
sum_{i=max(1, t-7)}^{t} chp_stop(i)  <= 1 - chp_on(t)
```

### 6. Grid

- HP draws from the grid at `price_el(t)`.
- CHP sells to the grid at `price_el(t)`.
- No grid capacity limit.
- Boiler and CHP fuel priced at `47.06 EUR/MWh_Hs` (gas + CO2).

## Objective

```
min sum_t [
    hp_el_in(t)        * price_el(t) * 0.25      # HP electricity cost
  + boiler_fuel_in(t)  * 47.06       * 0.25      # boiler fuel + CO2
  + chp_fuel_in(t)     * 47.06       * 0.25      # CHP fuel + CO2
  + 600 * chp_start(t)                            # CHP startup cost
  - chp_el_out(t)      * price_el(t) * 0.25      # CHP electricity revenue
]
```

## Notes

- **MILP.** Integrality from unit on/off, min up/down windows, and storage charge/discharge mode.
- Demand and DA price are hourly inputs broadcast across 4 × 15-min steps each hour.
- Rolling horizon: end-of-day SoC + unit on/off states + time-in-state seed the next day's solve.
- Hard SoC floor of 50 at every `t` replaces the earlier soft-penalty + terminal-target formulation. The `use_hard_soc_min` flag in `all_in_one_evaluation_notebook.ipynb` exists but is currently not gating the constraint — the floor is always on.
