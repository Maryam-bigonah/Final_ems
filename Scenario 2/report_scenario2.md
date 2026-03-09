# Scenario 2 – PV + BESS + Flexible Demand (Full Technical Note)

This note documents from zero to 100 how Scenario 2 is implemented, which bugs were fixed, how the optimization model works, and how the KPIs and plots are generated. You can use this text directly in your thesis or to explain the implementation to your professor.

---

## 1. Scenario Overview

Scenario 2 ("PV + BESS + Flexible Demand") extends the previous scenarios:

- **S0 – Baseline:** Building without PV/BESS, no flexibility.
- **S1-PV-Only:** Building with PV only, no battery.
- **S1-PV+BESS:** Building with PV + battery storage, no flexible demand.
- **S2-PV+BESS+Flex (this scenario):** Same PV + BESS as S1, but add MILP-based scheduling of:
  - Household appliances: Dishwasher (DW), Washing Machine (WM), Dryer (DR),
  - EV charging for selected archetypes.

The goal of Scenario 2 is to **minimize annual electricity cost** while:

- Preserving occupant **comfort** via a daily comfort window (06:00–23:00) for noisy appliances.
- Respecting **presence-based EV charging windows** (weekday evenings/nights, plus weekend rules).
- Obeying all physical constraints: battery capacity and power limits, PV balance, and a global grid power cap.
- **Conserving service**: the total annual energy of each shiftable appliance type is the same as in the baseline (no artificial reduction of energy consumption).

All the main logic is implemented in `run_scenario2.py` and plotting is handled in `plot_scenario2.py`.

---

## 2. Configuration and Physical Parameters

At the top of `run_scenario2.py` we define scenario settings and physical constants:

```python
CONFIG = {
    # Shiftable appliance set:
    # "MAIN" = {DW, WM, DR}, "EXTENDED" can add others like Iron/Vacuum
    "SHIFTABLE_SET": "MAIN", 
    
    # "SAME_DAY" = shifts within the baseline day
    # "DEFERRAL_24H" = allows deferral up to +24 hours
    "SHIFT_MODE": "SAME_DAY", 
    
    # Comfort window for noisy appliances (local time)
    "COMFORT_START_HOUR": 6,   # 06:00
    "COMFORT_END_HOUR": 23,    # 23:00 (finish before 23:00)
    
    # Building grid connection limit [kW]
    "P_CAP_GRID": 50.0 
}

# Battery physical parameters
EB      = 20.0   # Battery capacity [kWh]
P_CH_MAX = 5.0   # Max charge power [kW]
P_DIS_MAX = 5.0  # Max discharge power [kW]
SOC_MIN  = 1.0   # Min SOC [kWh]
SOC_MAX  = 19.0  # Max SOC [kWh]
EFF_CH   = 0.95  # Charge efficiency
EFF_DIS  = 0.95  # Discharge efficiency

# Building composition
NK = 5  # Apartments per archetype
ARCHETYPES = ["CoupleWorking", "Family1Child", "Family3Children", "RetiredCouple"]
```

**How to explain it:**

- Each archetype represents **5 identical apartments** (NK = 5), so the building has 20 apartments.
- Appliances can move only inside the comfort window; they never run overnight in F3 during weekdays.
- The battery has 20 kWh useable capacity and charge/discharge power capped at 5 kW.

---

## 3. Data Preparation

All data is prepared in `load_data_s2()`.

### 3.1. Electricity Prices and PV Forecast

We first load hourly electricity prices (including TOU bands) and PV generation forecasts for year 2025:

```python
def load_data_s2():
    el_prices = load_electricity_price(DATASET_DIR / "arera_fixed_prices_2025.csv")
    ref_idx = el_prices.index  # 8760 hourly steps in 2025

    pv_df = pd.read_csv(DATASET_DIR / "pv_prediction_2025.csv")
    pv_df["dt_end"] = pd.to_datetime(pv_df["dt_end"])
    pv_df.set_index("dt_end", inplace=True)

    # Align PV forecast to the same hourly index and ensure kW units
    P_pv_series = pv_df.reindex(ref_idx).fillna(0)["P_pred"]
    if P_pv_series.max() > 100:
        P_pv_series /= 1000.0  # convert W to kW if necessary
```

This ensures that all time series (price, PV, loads) share a **common DateTimeIndex** with 8,760 hours.

### 3.2. Extracting Shiftable Appliance Tasks

For each archetype, we load its sub-metered electricity CSV and automatically detect appliance cycles (tasks):

```python
arch_files = {
    "CoupleWorking":   "couple working.Electricity.csv",
    "Family1Child":    "Family _1_ child, 1 at work.Electricity.csv",
    "Family3Children": "Family, 3 children.Electricity.csv",
    "RetiredCouple":   "Retired Couple, no work.Electricity.csv",
}

load_fixed_k = {}
shiftable_labels = get_shiftable_labels()  # maps DW/WM/DR to column name patterns
shiftable_tasks = {k: {app: [] for app in shiftable_labels} for k in ARCHETYPES}

from src.loader import _find_ts_col, _parse_time, _year_shift_and_trim, _resample_minute_to_hourly
print("\n--- Extracting Shiftable Tasks (Set = {}) ---".format(CONFIG["SHIFTABLE_SET"]))

for arch in ARCHETYPES:
    df = pd.read_csv(DATASET_DIR / arch_files[arch], sep=";", low_memory=False)
    ts_col = _find_ts_col(df.columns)
    df["_ts"] = _parse_time(df[ts_col])
    df = df.set_index("_ts").sort_index()

    numeric_df = df.select_dtypes(include=np.number)
    numeric_df = numeric_df.drop(columns=[c for c in numeric_df.columns if "timestep" in c.lower()],
                                 errors="ignore")

    # If resolution is minute-level, resample to hourly by summation
    if pd.Series(numeric_df.index).diff().dropna().median() <= pd.Timedelta(minutes=2):
        numeric_df = _resample_minute_to_hourly(numeric_df)

    numeric_df = _year_shift_and_trim(numeric_df, 2025).reindex(ref_idx).fillna(0.0)
    tot_base = numeric_df.sum(axis=1)

    # Detect DW / WM / DR columns by keywords
    found_cols = {app: [] for app in shiftable_labels}
    for c in numeric_df.columns:
        for app, keywords in shiftable_labels.items():
            if any(kw.lower() in c.lower() for kw in keywords):
                found_cols[app].append(c)

    print(f"  [{arch}] Detected shiftables:")
    for app, cols in found_cols.items():
        print(f"    - {app}: {len(cols)} column(s)")

    # Build tasks per appliance
    eps = 0.001
    for app, cols in found_cols.items():
        if not cols:
            continue
        app_series = numeric_df[cols].sum(axis=1)

        in_task = False
        task_start_idx = None
        task_profile = []

        for i, val in enumerate(app_series):
            if val > eps:
                if not in_task:
                    in_task = True
                    task_start_idx = i
                task_profile.append(val)
            else:
                if in_task:
                    shiftable_tasks[arch][app].append({
                        "baseline_start": task_start_idx,
                        "baseline_day": app_series.index[task_start_idx].date(),
                        "duration": len(task_profile),
                        "profile": task_profile.copy(),
                    })
                    in_task = False
                    task_profile = []

        # subtract shiftable demand from total to get fixed load
        tot_base -= app_series

    # Fixed load (kW) at building level (×NK apartments)
    load_fixed_k[arch] = np.maximum(tot_base.values * NK, 0.0)
```

**Interpretation:**

- A **task** is a continuous block of non-zero power for DW/WM/DR.
- Each task has:
  - `baseline_start` (hour index),
  - `baseline_day` (date),
  - `duration` (number of hours),
  - `profile` (list of hourly power values).
- `load_fixed_k[arch]` represents “everything else” that is **not shifted** (lighting, cooking, etc.), scaled to building-level by `NK`.

### 3.3. EV Charging Requirements and Time Windows

We read baseline EV demand and derive per-day energy requirements and allowed windows:

```python
print("\n--- Extracting EV Charging Requirements ---")
ev_df = pd.read_csv(DATASET_DIR / "EV.csv", low_memory=False)
ts_col = _find_ts_col(ev_df.columns)
ev_df["_ts"] = _parse_time(ev_df[ts_col])
ev_df = ev_df.set_index("_ts").sort_index().reindex(ref_idx).fillna(0.0)

ev_req_d = {k: {} for k in ARCHETYPES}
ev_base_series = {}
ev_map = {"CoupleWorking": "Couple Family", "Family1Child": "Family_1_child"}

for arch in ARCHETYPES:
    if arch in ev_map:
        colname = [c for c in ev_df.columns if ev_map[arch] in c][0]
        val_series = ev_df[colname].copy() * NK  # building-level EV demand
        ev_base_series[arch] = val_series

        # Daily energy requirement, with window ending at 08:00
        target_sum = val_series.resample("24h", offset="8h").sum()
        ev_req_d[arch] = target_sum

        # Diagnostics: baseline energy outside the chosen window
        w_c, w_f = get_ev_windows(el_prices)
        w_mask = w_c if arch == "CoupleWorking" else w_f
        out_of_bounds = val_series.values[~w_mask].sum()
        print(f"  ↳ {arch} total EV = {target_sum.sum():.2f} kWh")
        print(f"    ↳ Baseline charge outside target mapped window: {out_of_bounds:.2f} kWh")
    else:
        ev_base_series[arch] = pd.Series(0.0, index=ref_idx)
```

Allowed EV charging windows are defined in `get_ev_windows(el_prices)`, based on day-of-week and holidays, with different evening start times for **CoupleWorking** and **Family1Child**.

---

## 4. MILP Optimization Model (`solve_milp_s2`)

The optimization model is a **Mixed-Integer Linear Program (MILP)** defined in `solve_milp_s2()`.

### 4.1. Decision Variables

```python
def solve_milp_s2(p_pv, load_fixed_k, shiftable_tasks, ev_req_d, el_prices):
    T = len(p_pv)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(1, T)
    timestamps = el_prices.index
    pv = p_pv.values
    price = el_prices["price_eur_per_kwh"].values

    # Power flows
    m.G_imp       = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Grid import [kW]
    m.P_dis       = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Battery discharge [kW]
    m.P_ch_pv     = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Battery charge from PV [kW]
    m.P_ch_grid   = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Battery charge from grid [kW]
    m.P_curt      = pyo.Var(m.T, within=pyo.NonNegativeReals)  # PV curtailment [kW]
    m.P_pv_to_load = pyo.Var(m.T, within=pyo.NonNegativeReals) # PV directly to load [kW]

    # Battery SOC [kWh]
    m.SOC = pyo.Var(pyo.RangeSet(0, T), bounds=(SOC_MIN, SOC_MAX))
    m.u   = pyo.Var(m.T, within=pyo.Binary)  # Charge/discharge mode switch

    # EV charging [kW]
    w_couple, w_family = get_ev_windows(el_prices)
    m.p_EV_Couple = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.p_EV_Family = pyo.Var(m.T, within=pyo.NonNegativeReals)
```

For flexible appliances, we have **binary task-start variables**:

```python
    feasible_starts = []
    task_idx = 0
    task_info = {}
    shiftable_labels = get_shiftable_labels()

    c_start = CONFIG["COMFORT_START_HOUR"]
    c_end   = CONFIG["COMFORT_END_HOUR"]

    print("\nMapping valid integer schedules...")
    for arch in ARCHETYPES:
        for app, tasks in shiftable_tasks[arch].items():
            for task in tasks:
                dur = task["duration"]
                b_s = task["baseline_start"]
                day_d = task["baseline_day"]

                start_h = np.where(timestamps.date == day_d)[0][0]
                S_j = []
                h_min, h_max = 0, 24
                if CONFIG["SHIFT_MODE"] == "DEFERRAL_24H":
                    h_max = 48

                # Build feasible start times within comfort window
                for h in range(h_min, h_max):
                    global_idx = start_h + h
                    if global_idx >= T:
                        break

                    fits = True
                    for lh in range(dur):
                        hod = timestamps[global_idx + lh].hour
                        if hod < c_start or hod >= c_end:
                            fits = False
                            break
                    if fits:
                        S_j.append(global_idx + 1)  # Pyomo indices are 1-based

                task_info[task_idx] = {
                    "arch": arch,
                    "app": app,
                    "baseline_start": b_s + 1,
                    "dur": dur,
                    "S_j": S_j,
                    "profile": task["profile"],
                }
                for s in S_j:
                    feasible_starts.append((task_idx, s))
                task_idx += 1

    m.FeasStarts = pyo.Set(initialize=feasible_starts, dimen=2)
    m.x_var = pyo.Var(m.FeasStarts, within=pyo.Binary)
```

**Interpretation:**

- For each task `tid` and feasible start time `s` (hour index), `m.x_var[tid, s]` is 1 if the task starts at hour `s`, 0 otherwise.
- `S_j` encodes **comfort constraints** and (optional) 24h deferral.

### 4.2. Constraints

#### 4.2.1. Exactly-One Start per Task

```python
    def exactly_one_start(m, tid):
        return sum(m.x_var[tid, s] for s in task_info[tid]["S_j"]) == 1

    m.c_one_start = pyo.Constraint(task_info.keys(), rule=exactly_one_start)
```

Each task must be scheduled exactly once.

#### 4.2.2. Appliance Power Reconstruction (Convolution)

We define helper functions to reconstruct scheduled power from `x_var` and task profiles:

```python
    def get_sch(t):
        v = 0.0
        for tid, info in task_info.items():
            prof = info["profile"]
            for lh in range(info["dur"]):
                s = t - lh
                if s in info["S_j"]:
                    v += m.x_var[tid, s] * prof[lh] * NK
        return v

    def get_sch_app(arch, app, t):
        """Scheduled appliance ENERGY [kW] at time t for one archetype+appliance (building-level, ×NK)."""
        v = 0.0
        for tid, info in task_info.items():
            if info["arch"] == arch and info["app"] == app:
                prof = info["profile"]
                for lh in range(info["dur"]):
                    s = t - lh
                    if s in info["S_j"]:
                        v += m.x_var[tid, s] * prof[lh] * NK
        return v
```

**Important fix:** originally `get_sch_app` added just `m.x_var[tid, s]` (counts), not `profile × NK`. This was corrected to properly reconstruct **kW** values from the task profiles.

#### 4.2.3. Non-Overlap of Appliances

For each archetype and appliance type, tasks cannot overlap:

```python
    m.c_non_overlap = pyo.ConstraintList()
    for arch in ARCHETYPES:
        for app in shiftable_labels.keys():
            tids = [tid for tid, info in task_info.items()
                    if info["arch"] == arch and info["app"] == app]
            if not tids:
                continue
            for t in range(1, T+1):
                count = sum(
                    m.x_var[tid, t-lh]
                    for tid in tids
                    for lh in range(task_info[tid]["dur"])
                    if (t-lh) in task_info[tid]["S_j"]
                )
                if type(count) is not int:
                    m.c_non_overlap.add(count <= 1)
```

#### 4.2.4. EV Power and Daily Energy Requirements

```python
    m.c_ev_bounds = pyo.ConstraintList()
    m.c_ev_req = pyo.ConstraintList()
    for arch, v, w_mask in [("CoupleWorking", m.p_EV_Couple, w_couple),
                             ("Family1Child", m.p_EV_Family, w_family)]:
        p_max = 3.7 * NK  # EV charger power cap
        for t in range(1, T+1):
            if w_mask[t-1]:
                m.c_ev_bounds.add(v[t] <= p_max)
            else:
                m.c_ev_bounds.add(v[t] == 0)

        for start_dt, req_val in ev_req_d[arch].items():
            t_s = np.where(timestamps == start_dt)[0]
            if not len(t_s):
                continue
            idx1 = t_s[0] + 1
            idx2 = min(idx1 + 24, T+1)
            tot = sum(v[t] for t in range(idx1, idx2))
            # Enforce daily EV energy within a tiny tolerance
            m.c_ev_req.add(pyo.inequality(-1e-4, tot - req_val, 1e-4))
```

#### 4.2.5. Grid Power Balance and Cap

Grid balance at each hour:

```python
    m.c_grid_bal = pyo.ConstraintList()
    m.c_anti_cluster = pyo.ConstraintList()
    for t in range(1, T+1):
        f_sum = sum(load_fixed_k[k][t-1] for k in ARCHETYPES)
        load_tilde_t = f_sum + get_sch(t) + m.p_EV_Couple[t] + m.p_EV_Family[t]

        # Supply = PV_to_load + Discharge + Grid_import
        # Demand = Load_tilde + Grid_charge
        m.c_grid_bal.add(
            m.P_pv_to_load[t] + m.P_dis[t] + m.G_imp[t]
            == load_tilde_t + m.P_ch_grid[t]
        )
        # Grid cap
        m.c_anti_cluster.add(m.G_imp[t] <= CONFIG["P_CAP_GRID"])

    # PV balance
    m.pv_bal = pyo.Constraint(m.T, rule=lambda m, t:
        m.P_pv_to_load[t] + m.P_ch_pv[t] + m.P_curt[t] == pv[t-1]
    )
```

Battery SOC dynamics and limits:

```python
    m.grid_charge = pyo.Constraint(m.T, rule=lambda m, t:
        m.P_ch_grid[t] <= m.G_imp[t]
    )

    m.soc_dyn = pyo.Constraint(m.T, rule=lambda m, t:
        m.SOC[t] == m.SOC[t-1]
        + EFF_CH * (m.P_ch_pv[t] + m.P_ch_grid[t])
        - (1 / EFF_DIS) * m.P_dis[t]
    )

    m.soc_cyclic = pyo.Constraint(rule=lambda m: m.SOC[0] == m.SOC[T])

    m.p_ch_max = pyo.Constraint(m.T, rule=lambda m, t:
        m.P_ch_pv[t] + m.P_ch_grid[t] <= P_CH_MAX * m.u[t]
    )
    m.p_dis_max = pyo.Constraint(m.T, rule=lambda m, t:
        m.P_dis[t] <= P_DIS_MAX * (1 - m.u[t])
    )
```

### 4.3. Objective Function

Minimize total electricity cost over the year:

```python
    m.obj = pyo.Objective(
        rule=lambda m: sum(price[t-1] * m.G_imp[t] for t in m.T),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = 180
    solver.options["mip_rel_gap"] = 0.02
    print("Solving MILP via HiGHS...")
    res = solver.solve(m, tee=True)
```

---

## 5. Reconstructing Optimized Loads and Bug Fix

### 5.1. Extract Time Series Outputs

After solving, we extract all main time-series into a DataFrame `df` indexed by timestamps:

```python
    def get_var(v):
        return np.array([pyo.value(v[t]) for t in m.T])

    df = pd.DataFrame({
        "G_imp": get_var(m.G_imp),
        "P_dis": get_var(m.P_dis),
        "P_ch_pv": get_var(m.P_ch_pv),
        "P_ch_grid": get_var(m.P_ch_grid),
        "P_curt": get_var(m.P_curt),
        "P_pv_to_load": get_var(m.P_pv_to_load),
        "SOC": np.array([pyo.value(m.SOC[t]) for t in range(1, T+1)]),
        "p_EV_Couple": get_var(m.p_EV_Couple),
        "p_EV_Family": get_var(m.p_EV_Family),
        "u": get_var(m.u),
    }, index=timestamps)
```

We also write the per-task schedule to `scenario2_task_schedule.csv`.

### 5.2. Correct Per-Archetype Load Reconstruction

The **critical corrected part** is the reconstruction of per-archetype optimized load `L_tilde_cache[arch]` using the full power profiles, not just counts of active tasks:

```python
    # ── Reconstruct per-archetype optimized load using actual task PROFILES ──
    L_tilde_cache = {}
    sched_energy_k = {}   # total scheduled appliance energy per archetype
    baseline_energy_k = {}  # total baseline appliance energy per archetype

    for arch in ARCHETYPES:
        lt = load_fixed_k[arch].copy()  # fixed part [kW]
        sched_e_total = 0.0
        baseline_e_total = 0.0

        for app in shiftable_labels.keys():
            # get_sch_app returns building-level appliance power [kW] at each hour
            app_arr = np.array([
                pyo.value(get_sch_app(arch, app, t)) for t in range(1, T + 1)
            ])
            lt += app_arr
            sched_e_total += app_arr.sum()

            # Baseline appliance energy = sum of all task profiles × NK
            baseline_e_total += sum(
                sum(info["profile"]) * NK
                for tid, info in task_info.items()
                if info["arch"] == arch and info["app"] == app
            )

        # Add EV at building level for relevant archetypes
        if arch == "CoupleWorking":
            lt += df["p_EV_Couple"].values
        elif arch == "Family1Child":
            lt += df["p_EV_Family"].values

        L_tilde_cache[arch] = pd.Series(lt, index=timestamps)
        sched_energy_k[arch] = sched_e_total
        baseline_energy_k[arch] = baseline_e_total

    # Total building optimized load
    L_bld = pd.Series(sum(L_tilde_cache[arch] for arch in ARCHETYPES), index=timestamps)
```

**Bug fixed:** Previously, `get_sch_app` returned only a **count** of active tasks, which was then incorrectly scaled, leading to wrong optimized load curves used in plots. Now it returns the full energy-convolved appliance power, and the load is constructed exactly from profiles.

---

## 6. KPIs and Scenario Comparison

### 6.1. KPI Computation

We compute summary KPIs for S2 (and reuse S0/S1 KPIs) using `compute_kpi()`:

```python
def compute_kpi(df, p_load, p_pv, p_el, c_gas_0):
    e_imp = float(df["G_imp"].sum())
    c_el = float((df["G_imp"] * p_el).sum())
    e_pv = float(p_pv.sum())
    e_curt = float(df["P_curt"].sum())
    e_pv_used = e_pv - e_curt

    e_ch_pv = float(df["P_ch_pv"].sum())
    e_ch_grid = float(df["P_ch_grid"].sum())
    e_ch = e_ch_pv + e_ch_grid
    e_dis = float(df["P_dis"].sum())

    return {
        "E_imp_kWh": e_imp,
        "C_el_eur": c_el,
        "E_pv_used_kWh": e_pv_used,
        "E_curt_kWh": e_curt,
        "E_ch_kWh": e_ch,
        "E_dis_kWh": e_dis,
        "Cycles": e_dis / EB if EB > 0 else 0,
        "C_total_eur": c_el + c_gas_0,
    }
```

In `main()`, after solving S2, we compute:

- KPIs for S2,
- `Savings_DeltaFlex = C_el(S1_PV_BESS) − C_el(S2_Flexible)`.

### 6.2. Final Building-Level Numbers

From `scenario2_kpis_building.json` and `summary.md`:

- **Baseline (S0, no PV/BESS, no flex):**
  - Electricity cost: **€10,589**
  - Grid import: **87,886 kWh**

- **S1-PV-Only (PV, no BESS, no flex):**
  - Electricity cost: **€7,229**
  - Grid import: **61,018 kWh**

- **S1-PV+BESS (PV + battery, no flex):**
  - Electricity cost: **€6,574**
  - Grid import: **56,334 kWh**

- **S2-PV+BESS+Flex (PV + BESS + flexible appliances + EV):**
  - Electricity cost: **€6,040**
  - Grid import: **53,475 kWh**

- **Incremental savings from Flex (ΔFlex = S1-PV+BESS → S2):**
  - Cost savings: **€534 / year**
  - Grid import reduction: **2,859 kWh / year**

These values appear in the cost table written in `summary.md`:

```markdown
## Cost Summary
| Scenario | Annual Electricity Cost | Grid Import [kWh] |
|----------|------------------------|-------------------|
| Baseline (S0) | €10,589 | 87,886 |
| PV Only (S1a) | €7,229 | 61,018 |
| PV + BESS (S1b) | €6,574 | 56,334 |
| PV + BESS + Flex (S2) | €6,040 | 53,475 |
| **ΔFlex (S1b → S2 savings)** | **€534** | **2,859** |
```

You can directly include this table in your thesis.

---

## 7. Validation and Verification Checks

### 7.1. Comfort and EV Window Compliance

The function `run_tests_s2()` checks:

- Time-series integrity (8760 hours),
- SOC within [SOC_MIN, SOC_MAX],
- No appliance scheduled outside comfort window, and
- No EV charging outside allowed windows.

Key excerpt (comfort and EV):

```python
comfort_violations = 0
c_start, c_end = CONFIG["COMFORT_START_HOUR"], CONFIG["COMFORT_END_HOUR"]
for idx, row in sched_df.iterrows():
    s = row["scheduled_start"]
    if pd.isna(s):
        continue
    for h in range(int(s), int(s + row["duration"])):
        hod = (h - 1) % 24
        if hod < c_start or hod >= c_end:
            comfort_violations += 1

checks.append((
    f"S2-1: Comfort Bounds Compliance (Violations: {comfort_violations})",
    comfort_violations == 0,
))

# EV window compliance
ev_c_out = df_s2["p_EV_Couple"].values[~w_couple].sum()
ev_f_out = df_s2["p_EV_Family"].values[~w_family].sum()
checks.append((
    f"S2-2: EV Window Compliance (Out-of-bound kWh: {ev_c_out+ev_f_out:.2f})",
    ev_c_out + ev_f_out <= 1e-3,
))
```

Result: **0 comfort violations, 0 EV out-of-window kWh**.

### 7.2. Appliance Energy Conservation

We verify that optimized scheduling does not change the total appliance energy per archetype:

```python
energy_checks = []
tol = 1.0  # kWh tolerance
for arch in ARCHETYPES:
    sched_e = sched_energy_k[arch]
    base_e = baseline_energy_k[arch]
    diff = abs(sched_e - base_e)
    ok = diff < tol
    energy_checks.append((arch, base_e, sched_e, diff, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {arch}: baseline_appliance={base_e:.2f} kWh, "
          f"scheduled_appliance={sched_e:.2f} kWh, diff={diff:.4f} kWh")

# Total
total_base_app = sum(e[1] for e in energy_checks)
total_sched_app = sum(e[2] for e in energy_checks)
total_diff = abs(total_base_app - total_sched_app)
all_energy_ok = total_diff < tol
print(f"  [{'PASS' if all_energy_ok else 'FAIL'}] TOTAL: baseline_appliance={total_base_app:.2f} kWh, "
      f"scheduled_appliance={total_sched_app:.2f} kWh, diff={total_diff:.4f} kWh")
checks.append((f"S2-4: Appliance Energy Conservation (diff={total_diff:.4f} kWh)", all_energy_ok))
```

Result: **exact match** (`Δ = 0.0000 kWh` both per archetype and in total).

### 7.3. Power Balance Check (Sample Week)

We also verify hourly power balance over a summer week:

```python
print("\nPOST-FIX VERIFICATION: Power Balance (Summer Week 2025-06-15 to 2025-06-22)")
bal_check_start = "2025-06-15"
bal_check_end = "2025-06-21 23:00"
idx_mask = (L_bld.index >= bal_check_start) & (L_bld.index <= bal_check_end)

L_sub = L_bld.values[idx_mask]
pv_to_load_sub = df_s2["P_pv_to_load"].values[idx_mask]
g_imp_sub = df_s2["G_imp"].values[idx_mask]
p_dis_sub = df_s2["P_dis"].values[idx_mask]
p_ch_grid_sub = df_s2["P_ch_grid"].values[idx_mask]

supply_sub = pv_to_load_sub + p_dis_sub + g_imp_sub

demand_sub = L_sub + p_ch_grid_sub
bal_residual = np.abs(supply_sub - demand_sub)
max_bal_err = bal_residual.max()

bal_ok = max_bal_err < 0.01
print(f"  [{'PASS' if bal_ok else 'FAIL'}] Max hourly balance residual: {max_bal_err:.6f} kW")
checks.append((f"S2-5: Power Balance Week Check (max residual={max_bal_err:.6f} kW)", bal_ok))
```

Result: max residual **0.000000 kW** – numerically perfect power balance.

---

## 8. Plots and Narrative

All plots are generated in `plot_scenario2.py` using the exported CSVs and KPI JSON.

### 8.1. Comparison Plots (4 Scenarios)

**Plot 1 – Annual Cost Comparison (4 scenarios)**

```python
s0_cost   = kpis["S0"]["BUILDING"]["C_el_eur"]
s1pv_cost = kpis["S1_PV_Only"]["C_el_eur"]
s1b_cost  = kpis["S1_PV_BESS"]["C_el_eur"]
s2_cost   = kpis["S2_Flexible"]["C_el_eur"]

labels = ["Baseline", "PV Only", "PV + BESS", "PV + BESS\n+ Flex"]
colors = ["#e63946", "#e9c46a", "#f4a261", "#2a9d8f"]
c_el = [s0_cost, s1pv_cost, s1b_cost, s2_cost]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, c_el, color=colors, edgecolor="k", linewidth=0.5, width=0.6)
ax.set_ylabel("Annual Electricity Cost [€]")
ax.set_title("Building Electricity Cost Comparison")
```

You annotate incremental savings (S0→PV Only, PV Only→PV+BESS, PV+BESS→Flex) and highlight **ΔFlex**.

**Plot 2 – Annual Grid Import Comparison (4 scenarios)**

```python
s0_imp   = kpis["S0"]["BUILDING"]["E_el_kWh"]
s1pv_imp = kpis["S1_PV_Only"]["E_imp_kWh"]
s1b_imp  = kpis["S1_PV_BESS"]["E_imp_kWh"]
s2_imp   = kpis["S2_Flexible"]["E_imp_kWh"]

imports = [s0_imp, s1pv_imp, s1b_imp, s2_imp]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, imports, color=colors, edgecolor="k", linewidth=0.5, width=0.6)
ax.set_ylabel("Annual Grid Import [kWh]")
ax.set_title("Building Grid Import Comparison")
```

You also annotate the reduction in grid import due to Flex.

### 8.2. Task Start Histogram – Shifting Toward PV-Rich Hours

```python
sched_df["baseline_start_hour"] = sched_df["baseline_start"] % 24
sched_df["scheduled_start_hour"] = sched_df["scheduled_start"] % 24

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(sched_df["baseline_start_hour"].dropna(), bins=24, range=(0, 24),
             color="gray", alpha=0.7, edgecolor="k")
axes[0].set_title("Baseline Task Starts (Original)")

axes[1].hist(sched_df["scheduled_start_hour"].dropna(), bins=24, range=(0, 24),
             color="#f4a261", alpha=0.9, edgecolor="k")
axes[1].set_title("Optimized Task Starts (S2 Flex)\nShifted toward PV-rich & off-peak hours")
```

This plot is ideal to show how the algorithm shifts tasks from evening peaks toward midday PV peaks (while still respecting the 06:00–23:00 comfort window).

### 8.3. Summer and Winter Week Dispatch

Plots 4 and 5 display a summer week and a winter week, respectively, using:

- `load_bld_tilde` (optimized building load),
- `pv`, `pv_to_load`, `grid_import`, `batt_discharge`, `curtailment`,
- `soc` (state-of-charge) in a second subplot.

They visualize how:

- Appliances and EV charging are shifted into PV-rich hours.
- The battery charges from PV and discharges later to reduce grid imports.
- Curtailment is reduced compared to PV-only or PV+BESS without flex.

### 8.4. PV Utilization Pie Chart

Plot 6 shows PV allocation among:

- Direct PV → Load,
- PV → BESS (charging),
- Curtailed PV.

This helps you argue that adding Flex + BESS increases PV self-consumption.

---

## 9. Narrative Conclusion for Presentation

You can summarize Scenario 2 to your professor as follows:

1. **Model extension:** Scenario 2 extends the PV+BESS dispatch model with MILP-based scheduling of dishwasher, washing machine, dryer, and EV charging, under realistic comfort and presence constraints.
2. **Data handling:** We detect each appliance cycle from sub-metered data as a separate task with a power profile, and we derive daily EV energy requirements with custom overnight windows.
3. **Optimization:** Each task has binary decision variables for its start time. The objective minimizes annual electricity cost by choosing when tasks run, when the battery charges/discharges, and when EVs charge.
4. **Constraints:** Hourly power balance, battery SOC limits, grid power cap, EV windows, and comfort windows (06:00–23:00) are all enforced.
5. **Corrected load reconstruction:** A previous bug counted active tasks instead of using their full power profiles, which distorted plots. We fixed this by convolving binary decisions with task profiles so that the optimized load `L_tilde` is physically consistent.
6. **Validation:** We verified:
   - SOC remains within [1, 19] kWh,
   - No comfort or EV window violations,
   - Appliance energy is conserved per archetype and in total (Δ = 0 kWh),
   - Hourly power balance is numerically exact in a sample week.
7. **Results:**
   - S0 → S1 (PV): large savings and import reduction.
   - S1 → S1+BESS: further savings by time-shifting energy.
   - **S1+BESS → S2 (Flex)**: additional savings of ~**€534/year** and ~**2.9 MWh/year** less grid import, with no comfort or service loss.
8. **Interpretation:** Flexibility does **not** shift noisy appliances into F3 (night-time) because of the 06:00–23:00 comfort window. Instead, tasks are moved toward **PV-rich midday hours** and EV charging is shifted toward allowed off-peak windows.

This note (`report_scenario2.md`) now lives in your `Scenario 2` folder and documents all the important design choices, code structures, and results for Scenario 2.
