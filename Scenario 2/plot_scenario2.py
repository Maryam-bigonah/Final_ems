"""
plot_scenario2.py – Generate all Scenario 2 plots from corrected dispatch data.
Uses the energy-profile-based L_tilde (not binary-count version).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def plot_s2_results():
    base_dir = Path(__file__).resolve().parent / "results" / "scenario2"

    # ── 1. Load Data ──
    with open(base_dir / "scenario2_kpis_building.json", "r") as f:
        kpis = json.load(f)

    ts_df = pd.read_csv(base_dir / "scenario2_dispatch_timeseries.csv",
                         parse_dates=["timestamp"])
    ts_df.set_index("timestamp", inplace=True)

    sched_df = pd.read_csv(base_dir / "scenario2_task_schedule.csv")

    # All 4 scenarios
    s0_cost  = kpis["S0"]["BUILDING"]["C_el_eur"]
    s1pv_cost = kpis["S1_PV_Only"]["C_el_eur"]
    s1b_cost = kpis["S1_PV_BESS"]["C_el_eur"]
    s2_cost  = kpis["S2_Flexible"]["C_el_eur"]
    delta_flex = kpis["Savings_DeltaFlex"]

    s0_imp   = kpis["S0"]["BUILDING"]["E_el_kWh"]
    s1pv_imp = kpis["S1_PV_Only"]["E_imp_kWh"]
    s1b_imp  = kpis["S1_PV_BESS"]["E_imp_kWh"]
    s2_imp   = kpis["S2_Flexible"]["E_imp_kWh"]

    # ── Plot 1: Cost Comparison – 4 scenarios ──
    labels = ["Baseline", "PV Only", "PV + BESS", "PV + BESS\n+ Flex"]
    colors = ["#e63946", "#e9c46a", "#f4a261", "#2a9d8f"]
    c_el = [s0_cost, s1pv_cost, s1b_cost, s2_cost]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, c_el, color=colors, edgecolor="k", linewidth=0.5, width=0.6)
    ax.set_ylabel("Annual Electricity Cost [€]")
    ax.set_title("Building Electricity Cost Comparison")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 80,
                f"€{yval:,.0f}", ha="center", fontweight="bold", fontsize=9)

    # Savings annotations
    # S0 → PV Only
    sav_pv = s0_cost - s1pv_cost
    ax.annotate(f"−€{sav_pv:,.0f}",
                xy=(bars[1].get_x() + bars[1].get_width() / 2, s1pv_cost),
                xytext=(bars[1].get_x() + bars[1].get_width() / 2, s1pv_cost - 350),
                ha="center", fontsize=8, color="#555", fontstyle="italic")
    # PV Only → PV+BESS
    sav_bess = s1pv_cost - s1b_cost
    ax.annotate(f"−€{sav_bess:,.0f}",
                xy=(bars[2].get_x() + bars[2].get_width() / 2, s1b_cost),
                xytext=(bars[2].get_x() + bars[2].get_width() / 2, s1b_cost - 350),
                ha="center", fontsize=8, color="#555", fontstyle="italic")
    # PV+BESS → Flex (ΔFlex)
    if delta_flex > 0:
        mid_x = (bars[2].get_x() + bars[2].get_width() / 2 +
                 bars[3].get_x() + bars[3].get_width() / 2) / 2
        ax.annotate(f"ΔFlex: −€{delta_flex:,.0f}",
                    xy=(mid_x, min(s1b_cost, s2_cost)),
                    xytext=(mid_x, max(s1b_cost, s2_cost) + 400),
                    ha="center", fontweight="bold", color="green",
                    arrowprops=dict(arrowstyle="->", color="green"),
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="green", boxstyle="round,pad=0.3"))
    ax.set_ylim(0, max(c_el) * 1.18)
    plt.tight_layout()
    plt.savefig(base_dir / "plot1_cost_comparison.png")
    plt.savefig(base_dir / "plot1_cost_comparison.pdf")
    plt.close()

    # ── Plot 2: Grid Import Comparison – 4 scenarios ──
    imports = [s0_imp, s1pv_imp, s1b_imp, s2_imp]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, imports, color=colors, edgecolor="k", linewidth=0.5, width=0.6)
    ax.set_ylabel("Annual Grid Import [kWh]")
    ax.set_title("Building Grid Import Comparison")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 500,
                f"{yval:,.0f}", ha="center", fontweight="bold", fontsize=9)

    delta_imp_flex = s1b_imp - s2_imp
    if delta_imp_flex > 0:
        mid_x2 = (bars[2].get_x() + bars[2].get_width() / 2 +
                  bars[3].get_x() + bars[3].get_width() / 2) / 2
        ax.annotate(f"ΔFlex: −{delta_imp_flex:,.0f} kWh",
                    xy=(mid_x2, min(s1b_imp, s2_imp)),
                    xytext=(mid_x2, max(s1b_imp, s2_imp) + 3000),
                    ha="center", fontweight="bold", color="green",
                    arrowprops=dict(arrowstyle="->", color="green"),
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="green", boxstyle="round,pad=0.3"))
    ax.set_ylim(0, max(imports) * 1.18)
    plt.tight_layout()
    plt.savefig(base_dir / "plot2_grid_import_comparison.png")
    plt.savefig(base_dir / "plot2_grid_import_comparison.pdf")
    plt.close()

    # ── Plot 3: Task Start Histogram (baseline vs optimized) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sched_df["baseline_start_hour"] = sched_df["baseline_start"] % 24
    sched_df["scheduled_start_hour"] = sched_df["scheduled_start"] % 24

    axes[0].hist(sched_df["baseline_start_hour"].dropna(), bins=24, range=(0, 24),
                 color="gray", alpha=0.7, edgecolor="k")
    axes[0].set_title("Baseline Task Starts (Original)")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(range(0, 25, 2))

    axes[1].hist(sched_df["scheduled_start_hour"].dropna(), bins=24, range=(0, 24),
                 color="#f4a261", alpha=0.9, edgecolor="k")
    axes[1].set_title("Optimized Task Starts (S2 Flex)\nShifted toward PV-rich & off-peak hours")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_xticks(range(0, 25, 2))

    plt.suptitle("Appliance Task Start Distribution – Baseline vs Flex Optimized", y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(base_dir / "plot3_task_starts_hist.png", bbox_inches="tight")
    plt.savefig(base_dir / "plot3_task_starts_hist.pdf", bbox_inches="tight")
    plt.close()

    # ── Plot 4: Summer Week Dispatch (corrected L_bld_tilde + PV_to_load + curtailment) ──
    start_date = "2025-06-15"
    end_date = "2025-06-22"
    ts_sub = ts_df.loc[start_date:end_date].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(ts_sub.index, ts_sub["load_bld_tilde"], label="Optimized Building Load [kW]",
            color="blue", linewidth=1.2)
    ax.plot(ts_sub.index, ts_sub["pv"], label="PV Generation [kW]",
            color="orange", linewidth=1.2, alpha=0.85)
    ax.fill_between(ts_sub.index, ts_sub["pv_to_load"],
                     color="gold", alpha=0.3, label="PV → Load")
    ax.fill_between(ts_sub.index, ts_sub["grid_import"],
                     color="red", alpha=0.15, label="Grid Import")
    ax.fill_between(ts_sub.index, ts_sub["batt_discharge"],
                     color="green", alpha=0.2, label="Battery Discharge")
    if "curtailment" in ts_sub.columns:
        ax.fill_between(ts_sub.index, ts_sub["curtailment"],
                         color="purple", alpha=0.15, label="Curtailment")

    ax.set_ylabel("Power [kW]")
    ax.set_title("Scenario 2: Optimized Building Load vs PV – Summer Week\n"
                 "(Flex shifts appliances toward PV-rich midday hours)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # SOC subplot
    ax2 = axes[1]
    ax2.fill_between(ts_sub.index, ts_sub["soc"], color="steelblue", alpha=0.5)
    ax2.plot(ts_sub.index, ts_sub["soc"], color="steelblue", linewidth=0.8)
    ax2.set_ylabel("SOC [kWh]")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, 20)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    plt.tight_layout()
    plt.savefig(base_dir / "plot4_summer_dispatch.png", bbox_inches="tight")
    plt.savefig(base_dir / "plot4_summer_dispatch.pdf", bbox_inches="tight")
    plt.close()

    # ── Plot 5: Winter Week Dispatch ──
    start_w = "2025-01-13"
    end_w = "2025-01-20"
    ts_w = ts_df.loc[start_w:end_w].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(ts_w.index, ts_w["load_bld_tilde"], label="Optimized Building Load [kW]",
            color="blue", linewidth=1.2)
    ax.plot(ts_w.index, ts_w["pv"], label="PV Generation [kW]",
            color="orange", linewidth=1.2, alpha=0.85)
    ax.fill_between(ts_w.index, ts_w["pv_to_load"], color="gold", alpha=0.3, label="PV → Load")
    ax.fill_between(ts_w.index, ts_w["grid_import"], color="red", alpha=0.15, label="Grid Import")
    ax.fill_between(ts_w.index, ts_w["batt_discharge"], color="green", alpha=0.2, label="Battery Discharge")
    if "curtailment" in ts_w.columns:
        ax.fill_between(ts_w.index, ts_w["curtailment"], color="purple", alpha=0.15, label="Curtailment")

    ax.set_ylabel("Power [kW]")
    ax.set_title("Scenario 2: Optimized Building Load vs PV – Winter Week\n"
                 "(Flex shifts appliances toward PV-rich midday hours)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(ts_w.index, ts_w["soc"], color="steelblue", alpha=0.5)
    ax2.plot(ts_w.index, ts_w["soc"], color="steelblue", linewidth=0.8)
    ax2.set_ylabel("SOC [kWh]")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, 20)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())

    plt.tight_layout()
    plt.savefig(base_dir / "plot5_winter_dispatch.png", bbox_inches="tight")
    plt.savefig(base_dir / "plot5_winter_dispatch.pdf", bbox_inches="tight")
    plt.close()

    # ── Plot 6: PV Utilization Breakdown (Pie) ──
    e_pv_to_load = ts_df["pv_to_load"].sum()
    e_ch_pv = ts_df["batt_charge_pv"].sum()
    e_curt = ts_df["curtailment"].sum()
    pv_total = ts_df["pv"].sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [e_pv_to_load, e_ch_pv, e_curt]
    labels_pv = [f"PV → Load\n{e_pv_to_load:,.0f} kWh",
                 f"PV → BESS\n{e_ch_pv:,.0f} kWh",
                 f"Curtailed\n{e_curt:,.0f} kWh"]
    colors_pv = ["#2a9d8f", "#264653", "#e76f51"]
    ax.pie(sizes, labels=labels_pv, colors=colors_pv, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"PV Energy Allocation – S2 (Total: {pv_total:,.0f} kWh)")
    plt.tight_layout()
    plt.savefig(base_dir / "plot6_pv_utilization.png")
    plt.savefig(base_dir / "plot6_pv_utilization.pdf")
    plt.close()

    print(f"All plots saved to {base_dir}")


if __name__ == "__main__":
    plot_s2_results()
