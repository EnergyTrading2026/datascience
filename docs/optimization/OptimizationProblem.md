# Compact Summary

## Costs / Revenues

- HP electricity purchase: DA price
- Boiler gas price: 35 EUR/MWh_Hs
- CO2 factor: 0.201 tCO2/MWh_Hs
- CO2 price: 60 EUR/tCO2
- Effective gas + CO2 cost: 47.06 EUR/MWh_Hs
- CHP startup cost: 600 EUR
- CHP electricity sales revenue: DA price

## Core Constraints

- Heat demand must always be met

### Storage

- Capacity: 200 MWh_th
- Max charge: 15 MW_th
- Max discharge: 15 MW_th
- Initial SOC: 200 MWh_th
- Loss: 0.000125 MWh_th per 15 min

### Heat Pump

- Electrical power: 1 to 8 MW_el
- Heat output via COP = 3.5

### Boiler

- Thermal power: 2 to 12 MW_th
- Efficiency: 0.97
- Min runtime: 1 h
- Min downtime: 1 h

### CHP

- Electrical power: 2 to 6 MW_el
- Thermal power: up to 7 MW_th
- Electrical efficiency: 0.4
- Thermal efficiency: 0.48
- Overall efficiency: 0.88
- Min runtime: 8 intervals
- Min downtime: 8 intervals

## MILP Formulation

We consider a time-indexed MILP on 15-minute intervals.

### Sets and Indices

- `t in T = {1, ..., N}`: time intervals

### Parameters

- `Delta = 0.25 h`: interval length
- `H_t`: heat demand in interval `t` [MW_th]
- `pi_t`: day-ahead electricity price in interval `t` [EUR/MWh_el]
- `c_gas_co2 = 47.06`: effective gas plus CO2 cost [EUR/MWh_Hs]
- `c_start_CHP = 600`: CHP startup cost [EUR/start]
- `COP_HP = 3.5`
- `eta_B = 0.97`: boiler efficiency
- `eta_CHP_el = 0.4`
- `eta_CHP_th = 0.48`
- `S_max = 200`: storage capacity [MWh_th]
- `P_ch_max = 15`: max storage charge power [MW_th]
- `P_dis_max = 15`: max storage discharge power [MW_th]
- `S_0 = 200`: initial storage state of charge [MWh_th]
- `loss = 0.000125`: storage loss per interval [MWh_th]
- `UT_B = 4`, `DT_B = 4`: boiler min up/down time in intervals
- `UT_CHP = 8`, `DT_CHP = 8`: CHP min up/down time in intervals

### Decision Variables

- `p_HP_t >= 0`: heat pump electrical power [MW_el]
- `q_HP_t >= 0`: heat pump heat output [MW_th]
- `u_HP_t in {0,1}`: heat pump on/off
- `q_B_t >= 0`: boiler heat output [MW_th]
- `g_B_t >= 0`: boiler fuel input [MW_Hs]
- `u_B_t in {0,1}`: boiler on/off
- `v_B_t in {0,1}`: boiler startup
- `w_B_t in {0,1}`: boiler shutdown
- `p_CHP_t >= 0`: CHP electrical power [MW_el]
- `q_CHP_t >= 0`: CHP thermal power [MW_th]
- `g_CHP_t >= 0`: CHP fuel input [MW_Hs]
- `u_CHP_t in {0,1}`: CHP on/off
- `v_CHP_t in {0,1}`: CHP startup
- `w_CHP_t in {0,1}`: CHP shutdown
- `c_t >= 0`: storage charging power [MW_th]
- `d_t >= 0`: storage discharging power [MW_th]
- `SOC_t >= 0`: storage state of charge [MWh_th]

### Objective

Minimize total operating cost:

```text
min sum_t Delta * (pi_t * p_HP_t + c_gas_co2 * g_B_t + c_gas_co2 * g_CHP_t - pi_t * p_CHP_t)
    + sum_t c_start_CHP * v_CHP_t
```

This is equivalent to maximizing profit from CHP electricity sales minus electricity purchase, fuel, CO2, and startup costs.

### Constraints

#### 1. Heat Balance

For all `t in T`:

```text
q_HP_t + q_B_t + q_CHP_t + d_t - c_t = H_t
```

This enforces that heat demand is always met, while allowing the storage to absorb or release heat.

#### 2. Storage Dynamics

For all `t in T`:

```text
SOC_t = SOC_{t-1} + Delta * c_t - Delta * d_t - loss
0 <= SOC_t <= S_max
0 <= c_t <= P_ch_max
0 <= d_t <= P_dis_max
SOC_0 = 200
```

If simultaneous charge and discharge should be forbidden explicitly, add a binary storage mode variable. Without it, simultaneous charging/discharging is typically avoided by the objective unless prices or penalties create degeneracy.

#### 3. Heat Pump

For all `t in T`:

```text
q_HP_t = COP_HP * p_HP_t
1 * u_HP_t <= p_HP_t <= 8 * u_HP_t
```

This means the heat pump is either off or operates between 1 and 8 MW_el.

#### 4. Boiler

For all `t in T`:

```text
2 * u_B_t <= q_B_t <= 12 * u_B_t
q_B_t <= eta_B * g_B_t
u_B_t - u_B_{t-1} = v_B_t - w_B_t
```

Minimum up/down times:

For all `t` with `t >= UT_B`:

```text
sum_{tau=t-UT_B+1}^t v_B_tau <= u_B_t
```

For all `t` with `t >= DT_B`:

```text
sum_{tau=t-DT_B+1}^t w_B_tau <= 1 - u_B_t
```

#### 5. CHP

For all `t in T`:

```text
2 * u_CHP_t <= p_CHP_t
p_CHP_t <= 6 * u_CHP_t
0 <= q_CHP_t <= 7 * u_CHP_t
p_CHP_t <= eta_CHP_el * g_CHP_t
q_CHP_t <= eta_CHP_th * g_CHP_t
p_CHP_t + q_CHP_t <= 0.88 * g_CHP_t
u_CHP_t - u_CHP_{t-1} = v_CHP_t - w_CHP_t
```

Minimum up/down times:

For all `t` with `t >= UT_CHP`:

```text
sum_{tau=t-UT_CHP+1}^t v_CHP_tau <= u_CHP_t
```

For all `t` with `t >= DT_CHP`:

```text
sum_{tau=t-DT_CHP+1}^t w_CHP_tau <= 1 - u_CHP_t
```

### Initial Conditions

To solve the MILP completely, the initial commitment states must be specified:

- `u_HP_0`, `u_B_0`, `u_CHP_0`
- if needed, the elapsed up/down time before `t = 1`

These are required for a correct treatment of minimum up/down constraints near the start of the horizon.

### Modeling Note on CHP Data Consistency

The CHP data contains a small inconsistency:

- with `eta_CHP_el = 0.4` and `eta_CHP_th = 0.48`, a fixed-efficiency CHP implies `q_CHP_t = 1.2 * p_CHP_t`
- together with `q_CHP_t <= 7`, this implies `p_CHP_t <= 5.83`

So the stated upper bound `p_CHP_t <= 6` can only be reached if at least one of the following is relaxed:

- the thermal cap `7 MW_th`
- the fixed efficiency interpretation
- the exact efficiency values

The formulation above keeps the efficiencies and thermal cap as primary and therefore may never use the full 6 MW_el in the optimum.
