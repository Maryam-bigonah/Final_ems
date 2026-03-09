"""
src/verification.py – Strict verification checks for Scenario 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_all_checks(
    arch_data: dict,
    building_ts: pd.DataFrame,
    el_prices: pd.DataFrame,
    gas_prices: pd.Series,
    Nk: dict[str, int],
    ev_column_mapping: dict[str, str],
) -> list[tuple[str, bool, str]]:
    """
    Run all verification checks.

    *arch_data* maps archetype_name → {
        "e_el": pd.Series,
        "ev": pd.Series,
        "gas_df": pd.DataFrame,
        "c_el": pd.Series,
        "c_gas": pd.Series,
        "tou": pd.DataFrame,
        "kpi": dict,
    }

    Returns list of (check_name, passed, detail).
    """
    results: list[tuple[str, bool, str]] = []
    expected = 8_760

    # ── 1. Series length ────────────────────────────────────────────
    for name, d in arch_data.items():
        n = len(d["e_el"])
        ok = n == expected
        results.append(
            (f"Length {name} electricity", ok, f"{n} rows (expected {expected})")
        )
        n_gas = len(d["gas_df"])
        ok = n_gas == expected
        results.append(
            (f"Length {name} gas", ok, f"{n_gas} rows (expected {expected})")
        )

    n_el_price = len(el_prices)
    results.append(
        ("Length electricity price", n_el_price == expected, f"{n_el_price}")
    )
    n_gas_price = len(gas_prices)
    results.append(
        ("Length gas price", n_gas_price == expected, f"{n_gas_price}")
    )

    # ── 2. Timestamp alignment ──────────────────────────────────────
    ref_idx = el_prices.index
    for name, d in arch_data.items():
        match = d["e_el"].index.equals(ref_idx)
        results.append(
            (f"Timestamp alignment {name} electricity", match, "")
        )
        match_gas = d["gas_df"].index.equals(ref_idx)
        results.append(
            (f"Timestamp alignment {name} gas", match_gas, "")
        )
    match_gp = gas_prices.index.equals(ref_idx)
    results.append(("Timestamp alignment gas price", match_gp, ""))

    # ── 3. No NaNs ──────────────────────────────────────────────────
    for name, d in arch_data.items():
        ok = not d["e_el"].isna().any()
        results.append((f"No NaN {name} electricity", ok, ""))
        ok = not d["gas_df"].isna().any().any()
        results.append((f"No NaN {name} gas", ok, ""))
        ok = not d["c_el"].isna().any()
        results.append((f"No NaN {name} electricity cost", ok, ""))
        ok = not d["c_gas"].isna().any()
        results.append((f"No NaN {name} gas cost", ok, ""))

    # ── 4. No negatives ─────────────────────────────────────────────
    for name, d in arch_data.items():
        ok = (d["e_el"] >= 0).all()
        results.append((f"No negatives {name} electricity", ok, ""))
        ok = (d["gas_df"] >= 0).all().all()
        results.append((f"No negatives {name} gas", ok, ""))
    ok = (el_prices["price_eur_per_kwh"] >= 0).all()
    results.append(("No negative electricity prices", ok, ""))
    ok = (gas_prices >= 0).all()
    results.append(("No negative gas prices", ok, ""))

    # ── 5. TOU sum check ────────────────────────────────────────────
    tol = 1e-4
    for name, d in arch_data.items():
        tou = d["tou"]
        total_e = d["e_el"].sum()
        total_c = d["c_el"].sum()
        tou_e = tou["energy_kWh"].sum()
        tou_c = tou["cost_eur"].sum()
        ok_e = abs(tou_e - total_e) < tol
        ok_c = abs(tou_c - total_c) < tol
        results.append(
            (f"TOU energy sum {name}", ok_e,
             f"TOU={tou_e:.4f}, total={total_e:.4f}")
        )
        results.append(
            (f"TOU cost sum {name}", ok_c,
             f"TOU={tou_c:.4f}, total={total_c:.4f}")
        )

    # ── 6. EV enforcement ───────────────────────────────────────────
    for name, d in arch_data.items():
        if name not in ev_column_mapping:
            ev_sum = d["ev"].sum()
            ok = ev_sum == 0.0
            results.append(
                (f"EV zero enforcement {name}", ok,
                 f"EV sum={ev_sum:.6f}")
            )

    # ── 7. Sanity ranges ────────────────────────────────────────────
    for name, d in arch_data.items():
        annual_el = d["e_el"].sum()
        annual_gas = d["gas_df"]["gas_kWh"].sum()
        ok_el = 500 < annual_el < 20_000
        ok_gas = annual_gas >= 0  # gas can be very low in summer-only profiles
        results.append(
            (f"Sanity range {name} el", ok_el,
             f"{annual_el:.1f} kWh/year (expected 500–20,000)")
        )
        results.append(
            (f"Sanity range {name} gas", ok_gas,
             f"{annual_gas:.1f} kWh/year")
        )

    # ── 8. Building time series length ──────────────────────────────
    ok = len(building_ts) == expected
    results.append(("Building time series length", ok, f"{len(building_ts)}"))

    # ── 9. Building aggregate invariant ────────────────────────────
    bld_c0_ts = building_ts["C_el_bld"].sum() + building_ts["C_gas_bld"].sum()
    bld_c0_kpi = sum(
        d["kpi"]["C0_eur"] * Nk.get(name, 1)
        for name, d in arch_data.items()
    )
    ok = abs(bld_c0_ts - bld_c0_kpi) < 0.01
    results.append(
        ("Building C0 invariant", ok,
         f"TS={bld_c0_ts:.2f}, KPI={bld_c0_kpi:.2f}")
    )

    return results
