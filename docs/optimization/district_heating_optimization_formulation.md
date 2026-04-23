# District Heating Optimization Problem

## Goal

We want to operate a district heating system over one day at minimum cost.

At each 15-minute step, the optimizer decides how to combine:

- Heat Pump (HP)
- Condensing Boiler
- CHP
- Thermal Storage

to satisfy heat demand while interacting with the electricity day-ahead market.

The cost to minimize includes:

- Electricity purchase for the heat pump
- Gas + CO2 cost for boiler and CHP
- CHP startup cost
- Electricity sales revenue from CHP
- A terminal storage penalty so the optimizer does not empty the tank at the end of the horizon just to look artificially cheap

---

## 1. System Intuition

The physical system is simple:

- The heat pump converts electricity into heat.
- The boiler converts gas into heat.
- The CHP consumes gas and produces both electricity and heat.
- The storage can shift heat over time by charging in low-cost periods and discharging in high-cost periods.

At every time step, total heat supplied to the network must equal heat demand.

This already suggests the core optimization tradeoff:

- Use the heat pump when electricity is cheap enough.
- Use boiler or CHP when gas-based heat is more attractive.
- Use the storage to move heat across time.
- Avoid myopic storage depletion near the end of the horizon.

Because the boiler and CHP have on/off behavior and minimum run/downtime limits, the problem is a mixed-integer linear program (MILP).

---

## 2. Naming Convention

All variables follow:

```text
<asset><energy><role>
```

Examples:

- `hp_el_in` = electricity into heat pump
- `chp_th_out` = heat output from CHP
- `sto_th_soc` = storage state of charge

This naming keeps the formulation readable:

- `asset`: `hp`, `boiler`, `chp`, `sto`
- `energy`: `el`, `th`, `fuel`
- `role`: `in`, `out`, `charge`, `discharge`, `soc`

---

## 3. Time Structure

We optimize one day ahead:

- Time step: 15 min (`dt = 0.25 h`)
- Horizon: 24 h = 96 steps
- Index: `t = 1,...,96`

Demand and electricity prices are assumed hourly and applied to the four 15-minute steps within each hour.

---

## 4. Inputs and Parameters

### Time Series Inputs

- `demand_th(t)` [MW_th] - heat demand forecast
- `price_el(t)` [EUR/MWh_el] - day-ahead electricity price

### Fuel and Emissions

- Gas price: 35 EUR/MWh_Hs
- CO2 factor: 0.201 tCO2/MWh_Hs
- CO2 price: 60 EUR/tCO2

Therefore the effective fuel cost is:

```text
fuel_cost = 35 + 0.201 * 60 = 47.06 EUR/MWh_Hs
```

### Thermal Storage

- Capacity: 200 MWh_th
- Max charge: 15 MW_th
- Max discharge: 15 MW_th
- Loss: 0.000125 MWh_th per interval
- Initial SoC: 200 MWh_th

### Heat Pump

- Electrical input range: 1-8 MW_el when on
- COP: 3.5

### Boiler

- Thermal output range: 2-12 MW_th when on
- Efficiency: 0.97
- Minimum run time: 1 h
- Minimum downtime: 1 h

### CHP

- Electrical output range: 2-6 MW_el when on
- Thermal output range: 2.4-7.2 MW_th
- `eta_el = 0.40`
- `eta_th = 0.48`
- Minimum run time: 2 h
- Minimum downtime: 2 h
- Startup cost: 600 EUR

---

## 5. Decision Variables

The optimizer chooses both continuous operating levels and binary on/off decisions.

### Binary Variables

- `hp_on(t)`
- `boiler_on(t)`
- `chp_on(t)`
- `boiler_start(t)`, `boiler_stop(t)`
- `chp_start(t)`, `chp_stop(t)`
- `sto_mode_charge(t)` with `1 = charge`, `0 = discharge`

### Continuous Variables

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

#### Terminal Protection

- `sto_th_shortfall >= 0`

---

## 6. Initial Conditions

For the first optimization run:

```text
hp_on(0) = boiler_on(0) = chp_on(0) = 0
sto_th_soc(0) = 200
```

In rolling-horizon operation, unit states and storage state are passed from one day to the next.

---

## 7. Building the Model Step by Step

### 7.1 Heat Must Balance

The most fundamental constraint is the heat balance.

Heat can come from:

- the heat pump
- the boiler
- the CHP
- storage discharge

Heat can also be absorbed into storage through charging.

So at every time step:

```text
hp_th_out(t)
+ boiler_th_out(t)
+ chp_th_out(t)
+ sto_th_discharge(t)
- sto_th_charge(t)
= demand_th(t)
```

This is the backbone of the model.

### 7.2 Storage Shifts Heat Across Time

The storage has three key pieces:

1. Its state of charge must stay within physical bounds.
2. It cannot charge and discharge at the same time.
3. Its state evolves from one step to the next.

#### Storage Bounds

```text
0 <= sto_th_soc(t) <= 200
0 <= sto_th_charge(t) <= 15 * sto_mode_charge(t)
0 <= sto_th_discharge(t) <= 15 * (1 - sto_mode_charge(t))
```

The binary variable `sto_mode_charge(t)` prevents simultaneous charging and discharging.

#### Storage Dynamics

```text
sto_th_soc(t) =
  sto_th_soc(t-1)
+ 0.25 * sto_th_charge(t)
- 0.25 * sto_th_discharge(t)
- 0.000125
```

The factor `0.25` converts MW into MWh over a 15-minute interval.

### 7.3 End-of-Horizon Protection

Without any terminal condition, the optimizer tends to drain the storage at the end of the day because the next day is invisible.

To discourage that, we introduce a soft terminal target:

```text
sto_th_shortfall >= SoC_target - sto_th_soc(T)
sto_th_shortfall >= 0
```

and penalize `sto_th_shortfall` in the objective.

Optionally, a hard terminal constraint can be used instead:

```text
sto_th_soc(T) >= SoC_min
```

### 7.4 Heat Pump Model

The heat pump is the simplest controllable unit.

If it is on, it must operate between minimum and maximum electrical input:

```text
hp_on(t) * 1 <= hp_el_in(t) <= hp_on(t) * 8
```

Its thermal output follows directly from the COP:

```text
hp_th_out(t) = 3.5 * hp_el_in(t)
```

### 7.5 Boiler Model

The boiler produces only heat, but it has unit commitment logic.

#### Output and Fuel Consumption

```text
boiler_on(t) * 2 <= boiler_th_out(t) <= boiler_on(t) * 12
boiler_fuel_in(t) = boiler_th_out(t) / 0.97
```

#### Startup and Shutdown Logic

```text
boiler_start(t) >= boiler_on(t) - boiler_on(t-1)
boiler_stop(t) >= boiler_on(t-1) - boiler_on(t)
```

#### Minimum Run Time

Since the minimum run time is 1 hour and each step is 15 minutes, the boiler must stay on for 4 steps after startup:

```text
sum_{i=t-3}^{t} boiler_start(i) <= boiler_on(t)
```

#### Minimum Downtime

Likewise, after shutdown it must remain off for 4 steps:

```text
sum_{i=t-3}^{t} boiler_stop(i) <= 1 - boiler_on(t)
```

### 7.6 CHP Model

The CHP is similar to the boiler, but it co-produces electricity and heat.

#### Electrical Output, Thermal Coupling, Fuel Consumption

```text
chp_on(t) * 2 <= chp_el_out(t) <= chp_on(t) * 6
chp_th_out(t) = 1.2 * chp_el_out(t)
chp_fuel_in(t) = chp_el_out(t) / 0.4
```

The relation `chp_th_out(t) = 1.2 * chp_el_out(t)` comes from:

```text
eta_th / eta_el = 0.48 / 0.40 = 1.2
```

#### Startup and Shutdown Logic

```text
chp_start(t) >= chp_on(t) - chp_on(t-1)
chp_stop(t) >= chp_on(t-1) - chp_on(t)
```

#### Minimum Run Time

The CHP has a 2-hour minimum run time, which corresponds to 8 time steps:

```text
sum_{i=t-7}^{t} chp_start(i) <= chp_on(t)
```

#### Minimum Downtime

It must also remain off for 8 steps after shutdown:

```text
sum_{i=t-7}^{t} chp_stop(i) <= 1 - chp_on(t)
```

---

## 8. Cost Structure

Now that the physical model is defined, the objective becomes intuitive.

### Heat Pump Electricity Cost

```text
hp_el_in(t) * price_el(t) * 0.25
```

### Boiler Fuel Cost

```text
boiler_fuel_in(t) * 47.06 * 0.25
```

### CHP Fuel Cost

```text
chp_fuel_in(t) * 47.06 * 0.25
```

### CHP Startup Cost

```text
600 * chp_start(t)
```

### CHP Electricity Revenue

```text
chp_el_out(t) * price_el(t) * 0.25
```

This term is revenue, so it is subtracted in the minimization objective.

---

## 9. Full Objective Function

Putting everything together:

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

## 10. Final MILP Summary

The complete MILP consists of:

- Binary unit commitment variables for HP, boiler, CHP, and storage charge mode
- Continuous dispatch variables for power, heat, fuel, and storage state
- Heat balance constraints
- Storage bounds and storage dynamics
- Asset operating constraints
- Boiler and CHP minimum run/downtime constraints
- A linear cost objective with a terminal storage penalty

---

## 11. Notes

- The formulation is a MILP because of binary on/off and startup variables.
- Demand and prices are hourly inputs mapped to 15-minute steps.
- The terminal storage term prevents systematic end-of-horizon emptying.
- In rolling-horizon operation, storage state and unit states are passed forward.
- Forecast quality strongly affects solution quality.

---

## 12. Optional Extensions

- MPC with a multi-day horizon
- Learned storage value function
- Dynamic end-of-horizon SoC targets based on forecasts
