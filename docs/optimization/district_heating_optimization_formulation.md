# District Heating Optimization Problem

## Goal

Minimize total operational cost of meeting district heating demand by optimally dispatching:

- Heat Pump (HP)
- Condensing Boiler
- CHP
- Thermal Storage

while participating in the electricity day-ahead (DA) market.

The objective includes:

- Gas + CO2 costs (Boiler, CHP)
- Electricity cost (Heat Pump)
- Electricity revenue (CHP)
- Terminal storage penalty to prevent end-of-horizon draining


## Inputs

### Time Series

- `demand_th(t)` [MW_th] - heat demand (forecast)
- `price_el(t)` [EUR/MWh_el] - day-ahead electricity price

### Constants

#### Gas and CO2

- Gas price: 35 EUR/MWh_Hs
- CO2 factor: 0.201 tCO2/MWh_Hs
- CO2 price: 60 EUR/tCO2
- Effective fuel cost: **47.06 EUR/MWh_Hs**

#### Thermal Storage

- Capacity: 200 MWh_th
- Max charge: 15 MW_th
- Max discharge: 15 MW_th
- Loss: 0.000125 MWh_th per interval
- Initial SoC: 200 MWh_th

#### Heat Pump

- Power: 1-8 MW_el
- COP: 3.5

#### Boiler

- Power: 2-12 MW_th
- Efficiency: 0.97
- Min run time: 1 h
- Min downtime: 1 h

#### CHP

- Electrical: 2-6 MW_el
- Thermal: 2.4-7.2 MW_th
- eta_el = 0.40
- eta_th = 0.48
- Min run time: 2 h
- Min downtime: 2 h
- Startup cost: 600 EUR

---

## Time Structure

- Time step: 15 min (`dt = 0.25 h`)
- Horizon: 24 h = 96 steps
- Index: `t = 1,...,96`



## Decision Variables

### Binary

- `hp_on(t)`
- `boiler_on(t)`
- `chp_on(t)`
- `boiler_start(t)`, `boiler_stop(t)`
- `chp_start(t)`, `chp_stop(t)`
- `sto_mode_charge(t)` with `1 = charge`, `0 = discharge`

### Continuous

#### Heat Pump

- `hp_el_in(t)`
- `hp_th_out(t)`

#### Boiler

- `boiler_th_out(t)`
- `boiler_fuel_in(t)`

#### CHP

- `chp_el_out(t)`
- `chp_th_out(t)`
- `chp_fuel_in(t)`

#### Storage

- `sto_th_charge(t)`
- `sto_th_discharge(t)`
- `sto_th_soc(t)`

#### Terminal

- `sto_th_shortfall >= 0`



## Initial Conditions

```text
hp_on(0) = boiler_on(0) = chp_on(0) = 0
sto_th_soc(0) = 200
```

Rolling horizon:

- States carried over between days

---

## Constraints

### 1. Heat Balance

```text
hp_th_out(t)
+ boiler_th_out(t)
+ chp_th_out(t)
+ sto_th_discharge(t)
- sto_th_charge(t)
= demand_th(t)
```

### 2. Thermal Storage

#### Bounds

```text
0 <= sto_th_soc(t) <= 200
0 <= sto_th_charge(t) <= 15 * sto_mode_charge(t)
0 <= sto_th_discharge(t) <= 15 * (1 - sto_mode_charge(t))
```

#### Dynamics

```text
sto_th_soc(t) =
  sto_th_soc(t-1)
+ 0.25 * sto_th_charge(t)
- 0.25 * sto_th_discharge(t)
- 0.000125
```

#### End-of-Horizon Protection

Soft target:

```text
sto_th_shortfall >= SoC_target - sto_th_soc(T)
sto_th_shortfall >= 0
```

Hard constraint:

```text
sto_th_soc(T) >= SoC_min
```

### 3. Heat Pump

```text
hp_on(t) * 1 <= hp_el_in(t) <= hp_on(t) * 8
hp_th_out(t) = 3.5 * hp_el_in(t)
```

### 4. Boiler

```text
boiler_on(t) * 2 <= boiler_th_out(t) <= boiler_on(t) * 12
boiler_fuel_in(t) = boiler_th_out(t) / 0.97
```

Startup / shutdown:

```text
boiler_start(t) >= boiler_on(t) - boiler_on(t-1)
boiler_stop(t) >= boiler_on(t-1) - boiler_on(t)
```

Minimum run time:

```text
sum_{i=t-3}^{t} boiler_start(i) <= boiler_on(t)
```

Minimum downtime:

```text
sum_{i=t-3}^{t} boiler_stop(i) <= 1 - boiler_on(t)
```

### 5. CHP

```text
chp_on(t) * 2 <= chp_el_out(t) <= chp_on(t) * 6
chp_th_out(t) = 1.2 * chp_el_out(t)
chp_fuel_in(t) = chp_el_out(t) / 0.4
```

Startup / shutdown:

```text
chp_start(t) >= chp_on(t) - chp_on(t-1)
chp_stop(t) >= chp_on(t-1) - chp_on(t)
```

Minimum run time:

```text
sum_{i=t-7}^{t} chp_start(i) <= chp_on(t)
```

Minimum downtime:

```text
sum_{i=t-7}^{t} chp_stop(i) <= 1 - chp_on(t)
```

---

## Objective Function

```text
min sum_t [
    hp_el_in(t) * price_el(t) * 0.25
  + boiler_fuel_in(t) * 47.06 * 0.25
  + chp_fuel_in(t) * 47.06 * 0.25
  + 600 * chp_start(t)
  - chp_el_out(t) * price_el(t) * 0.25
] + lambda_term * sto_th_shortfall
```

---
