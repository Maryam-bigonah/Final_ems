"""
src/plots.py – Thesis-ready plots for Scenario 0 (seaborn + matplotlib).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np


# ── global style ─────────────────────────────────────────────────────────────

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
    rc={
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    },
)

BAND_COLORS = {"F1": "#e63946", "F2": "#457b9d", "F3": "#2a9d8f"}
SEASON_COLORS = {"Winter": "#457b9d", "Spring": "#a8dadc", "Summer": "#e9c46a", "Autumn": "#e76f51"}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save a figure as PNG and PDF."""
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── 1 & 2: representative week time series ──────────────────────────────────

def plot_representative_week(
    e_el: pd.Series,
    ev: pd.Series | None,
    week_start: str,
    season_label: str,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Plot hourly electricity import for a 7-day window."""
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    sel = e_el.loc[start:end]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(sel.index, sel.values, alpha=0.3, color="#457b9d")
    ax.plot(sel.index, sel.values, linewidth=0.8, color="#1d3557", label="Electricity import")

    if ev is not None and ev.loc[start:end].sum() > 0:
        ev_sel = ev.loc[start:end]
        ax.plot(ev_sel.index, ev_sel.values, linewidth=0.8, color="#e63946",
                linestyle="--", label="EV charging")

    ax.set_ylabel("kWh / h")
    ax.set_xlabel("")
    ax.set_title(f"{entity_name} – {season_label} Week ({start.strftime('%d %b')}–{end.strftime('%d %b %Y')})")
    ax.legend()
    fig.autofmt_xdate()

    tag = entity_name.replace(" ", "_")
    _save(fig, out_dir, f"{tag}_{season_label.lower()}_week")


# ── 3 & 4: monthly electricity energy & cost ────────────────────────────────

def plot_monthly_electricity(
    monthly: pd.DataFrame,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Bar plots for monthly electricity energy (kWh) and cost (€)."""
    months = monthly.index.tolist()
    x = range(len(months))

    # Energy
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=list(x), y=monthly["E_el_kWh"].values, color="#457b9d", ax=ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("kWh")
    ax.set_title(f"{entity_name} – Monthly Electricity Consumption")
    tag = entity_name.replace(" ", "_")
    _save(fig, out_dir, f"{tag}_monthly_el_energy")

    # Cost
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=list(x), y=monthly["C_el_eur"].values, color="#e63946", ax=ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("€")
    ax.set_title(f"{entity_name} – Monthly Electricity Cost")
    _save(fig, out_dir, f"{tag}_monthly_el_cost")


# ── 5: monthly gas energy & cost ─────────────────────────────────────────────

def plot_monthly_gas(
    monthly: pd.DataFrame,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Bar plots for monthly gas energy (kWh) and cost (€)."""
    months = monthly.index.tolist()
    x = range(len(months))
    tag = entity_name.replace(" ", "_")

    # Energy
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=list(x), y=monthly["G_kWh"].values, color="#2a9d8f", ax=ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("kWh (fuel)")
    ax.set_title(f"{entity_name} – Monthly Gas Consumption")
    _save(fig, out_dir, f"{tag}_monthly_gas_energy")

    # Cost
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=list(x), y=monthly["C_gas_eur"].values, color="#e76f51", ax=ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("€")
    ax.set_title(f"{entity_name} – Monthly Gas Cost")
    _save(fig, out_dir, f"{tag}_monthly_gas_cost")


# ── 6: seasonal bar plots ───────────────────────────────────────────────────

def plot_seasonal(
    seasonal: pd.DataFrame,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Grouped bar: seasonal electricity+gas energy and cost."""
    tag = entity_name.replace(" ", "_")

    # Energy
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    seasons = seasonal.index.tolist()
    x = np.arange(len(seasons))
    ax.bar(x - width / 2, seasonal["E_el_kWh"], width, label="Electricity (kWh)",
           color="#457b9d")
    ax.bar(x + width / 2, seasonal["G_kWh"], width, label="Gas (kWh fuel)",
           color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("kWh")
    ax.set_title(f"{entity_name} – Seasonal Energy")
    ax.legend()
    _save(fig, out_dir, f"{tag}_seasonal_energy")

    # Cost
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, seasonal["C_el_eur"], width, label="Electricity (€)",
           color="#457b9d")
    ax.bar(x + width / 2, seasonal["C_gas_eur"], width, label="Gas (€)",
           color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("€")
    ax.set_title(f"{entity_name} – Seasonal Cost")
    ax.legend()
    _save(fig, out_dir, f"{tag}_seasonal_cost")


# ── 7: load duration curve ───────────────────────────────────────────────────

def plot_load_duration_curve(
    e_el: pd.Series,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Electricity load duration curve (sorted hourly kWh/h)."""
    sorted_vals = np.sort(e_el.values)[::-1]
    hours = np.arange(1, len(sorted_vals) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hours, sorted_vals, alpha=0.3, color="#1d3557")
    ax.plot(hours, sorted_vals, linewidth=0.6, color="#1d3557")
    ax.set_xlabel("Hours")
    ax.set_ylabel("kWh / h")
    ax.set_title(f"{entity_name} – Electricity Load Duration Curve")
    ax.set_xlim(1, len(sorted_vals))
    tag = entity_name.replace(" ", "_")
    _save(fig, out_dir, f"{tag}_load_duration_curve")


# ── 8: TOU stacked bar ──────────────────────────────────────────────────────

def plot_tou_split(
    tou: pd.DataFrame,
    entity_name: str,
    out_dir: Path,
) -> None:
    """Stacked bar for TOU energy and cost."""
    tag = entity_name.replace(" ", "_")
    bands = ["F1", "F2", "F3"]
    colors = [BAND_COLORS[b] for b in bands]

    # Reindex to ensure band order
    tou = tou.reindex(bands)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Energy
    ax = axes[0]
    bottom = 0.0
    for i, band in enumerate(bands):
        val = tou.loc[band, "energy_kWh"]
        ax.bar(0, val, bottom=bottom, color=colors[i], label=band, width=0.5)
        if val > 0:
            ax.text(0, bottom + val / 2, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        bottom += val
    ax.set_xticks([0])
    ax.set_xticklabels(["Energy"])
    ax.set_ylabel("kWh / year")
    ax.set_title("Electricity by TOU Band")
    ax.legend()

    # Cost
    ax = axes[1]
    bottom = 0.0
    for i, band in enumerate(bands):
        val = tou.loc[band, "cost_eur"]
        ax.bar(0, val, bottom=bottom, color=colors[i], label=band, width=0.5)
        if val > 0:
            ax.text(0, bottom + val / 2, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        bottom += val
    ax.set_xticks([0])
    ax.set_xticklabels(["Cost"])
    ax.set_ylabel("€ / year")
    ax.set_title("Electricity Cost by TOU Band")

    fig.suptitle(entity_name, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"{tag}_tou_split")


# ── master plot function ─────────────────────────────────────────────────────

def generate_all_plots(
    entity_name: str,
    e_el: pd.Series,
    ev: pd.Series | None,
    gas_df: pd.DataFrame,
    c_el: pd.Series,
    c_gas: pd.Series,
    tou: pd.DataFrame,
    season_map: dict[str, list[int]],
    representative_weeks: dict[str, str],
    out_dir: Path,
) -> None:
    """Generate the full set of 8+ plots for one entity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = entity_name.replace(" ", "_")
    print(f"  Generating plots for {entity_name} …")

    # Monthly breakdown (for bar plots)
    monthly_df = pd.DataFrame(
        {
            "E_el_kWh": e_el,
            "C_el_eur": c_el,
            "G_kWh": gas_df["gas_kWh"].values if gas_df is not None else 0,
            "G_Smc": gas_df["gas_Smc"].values if gas_df is not None else 0,
            "C_gas_eur": c_gas.values,
        },
        index=e_el.index,
    )
    monthly = monthly_df.resample("ME").sum()
    monthly.index = monthly.index.strftime("%b")

    # Seasonal breakdown
    month_to_season = {}
    for season, months in season_map.items():
        for m in months:
            month_to_season[m] = season
    monthly_df["season"] = monthly_df.index.month.map(month_to_season)
    seasonal = monthly_df.groupby("season")[
        ["E_el_kWh", "C_el_eur", "G_kWh", "G_Smc", "C_gas_eur"]
    ].sum()
    seasonal = seasonal.reindex(list(season_map.keys()))

    # 1 & 2: Representative weeks
    for label, start in representative_weeks.items():
        plot_representative_week(e_el, ev, start, label.capitalize(), entity_name, out_dir)

    # 3 & 4: Monthly electricity
    plot_monthly_electricity(monthly, entity_name, out_dir)

    # 5: Monthly gas
    plot_monthly_gas(monthly, entity_name, out_dir)

    # 6: Seasonal
    plot_seasonal(seasonal, entity_name, out_dir)

    # 7: Load duration curve
    plot_load_duration_curve(e_el, entity_name, out_dir)

    # 8: TOU split
    plot_tou_split(tou, entity_name, out_dir)

    print(f"  ✓ {entity_name}: all plots saved to {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ARCHETYPE COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════

ARCH_PALETTE = {
    "CoupleWorking":   "#1d3557",
    "Family1Child":    "#457b9d",
    "Family3Children": "#e9c46a",
    "RetiredCouple":   "#e76f51",
}


def _arch_color(name: str) -> str:
    return ARCH_PALETTE.get(name, "#888888")


# ── C1: Annual KPI comparison bar chart ──────────────────────────────────────

def plot_comparison_annual_kpis(
    kpi_list: list[dict],
    out_dir: Path,
) -> None:
    """Grouped bar chart comparing annual electricity, gas, and total cost
    across all archetypes."""
    names = [k["archetype"] for k in kpi_list]
    x = np.arange(len(names))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Energy
    ax = axes[0]
    el_vals = [k["E_el_kWh"] for k in kpi_list]
    gas_vals = [k["G_kWh"] for k in kpi_list]
    ax.bar(x - width / 2, el_vals, width, label="Electricity (kWh)", color="#457b9d")
    ax.bar(x + width / 2, gas_vals, width, label="Gas (kWh fuel)", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("kWh / year")
    ax.set_title("Annual Energy Consumption")
    ax.legend()
    for i, (ev, gv) in enumerate(zip(el_vals, gas_vals)):
        ax.text(i - width / 2, ev + 50, f"{ev:,.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width / 2, gv + 50, f"{gv:,.0f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: Cost
    ax = axes[1]
    c_el = [k["C_el_eur"] for k in kpi_list]
    c_gas = [k["C_gas_eur"] for k in kpi_list]
    c0 = [k["C0_eur"] for k in kpi_list]
    ax.bar(x - width, c_el, width, label="Electricity (€)", color="#457b9d")
    ax.bar(x, c_gas, width, label="Gas (€)", color="#e76f51")
    ax.bar(x + width, c0, width, label="Total C₀ (€)", color="#264653")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("€ / year")
    ax.set_title("Annual Procurement Cost")
    ax.legend()
    for i, val in enumerate(c0):
        ax.text(i + width, val + 10, f"€{val:,.0f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Archetype Comparison – Annual KPIs", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "comparison_annual_kpis")


# ── C2: Monthly electricity comparison ──────────────────────────────────────

def plot_comparison_monthly_electricity(
    arch_data: dict,
    out_dir: Path,
) -> None:
    """Grouped bars: monthly electricity for all archetypes side by side."""
    months = None
    arch_monthly = {}
    for name, d in arch_data.items():
        m = d["e_el"].resample("ME").sum()
        m.index = m.index.strftime("%b")
        arch_monthly[name] = m
        if months is None:
            months = m.index.tolist()

    n_arch = len(arch_monthly)
    x = np.arange(len(months))
    width = 0.8 / n_arch

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, m) in enumerate(arch_monthly.items()):
        offset = (i - n_arch / 2 + 0.5) * width
        ax.bar(x + offset, m.values, width, label=name, color=_arch_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("kWh")
    ax.set_title("Monthly Electricity Consumption – All Archetypes")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "comparison_monthly_electricity")


# ── C3: Monthly gas comparison ───────────────────────────────────────────────

def plot_comparison_monthly_gas(
    arch_data: dict,
    out_dir: Path,
) -> None:
    """Grouped bars: monthly gas for all archetypes side by side."""
    months = None
    arch_monthly = {}
    for name, d in arch_data.items():
        m = d["gas_df"]["gas_kWh"].resample("ME").sum()
        m.index = m.index.strftime("%b")
        arch_monthly[name] = m
        if months is None:
            months = m.index.tolist()

    n_arch = len(arch_monthly)
    x = np.arange(len(months))
    width = 0.8 / n_arch

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, m) in enumerate(arch_monthly.items()):
        offset = (i - n_arch / 2 + 0.5) * width
        ax.bar(x + offset, m.values, width, label=name, color=_arch_color(name))
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("kWh (fuel)")
    ax.set_title("Monthly Gas Consumption – All Archetypes")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "comparison_monthly_gas")


# ── C4: Overlaid load duration curves ────────────────────────────────────────

def plot_comparison_load_duration(
    arch_data: dict,
    out_dir: Path,
) -> None:
    """Overlaid load duration curves for all archetypes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(1, 8761)

    for name, d in arch_data.items():
        sorted_vals = np.sort(d["e_el"].values)[::-1]
        ax.plot(hours, sorted_vals, linewidth=1.0, label=name,
                color=_arch_color(name))
    ax.set_xlabel("Hours")
    ax.set_ylabel("kWh / h")
    ax.set_title("Electricity Load Duration Curves – All Archetypes")
    ax.set_xlim(1, 8760)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "comparison_load_duration")


# ── C5: TOU split comparison ────────────────────────────────────────────────

def plot_comparison_tou(
    arch_data: dict,
    out_dir: Path,
) -> None:
    """Stacked bar comparing TOU energy and cost across archetypes."""
    names = list(arch_data.keys())
    bands = ["F1", "F2", "F3"]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy
    ax = axes[0]
    bottom = np.zeros(len(names))
    for band in bands:
        vals = [arch_data[n]["tou"].reindex(bands).loc[band, "energy_kWh"]
                for n in names]
        ax.bar(x, vals, 0.5, bottom=bottom, label=band, color=BAND_COLORS[band])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("kWh / year")
    ax.set_title("Electricity by TOU Band")
    ax.legend()

    # Cost
    ax = axes[1]
    bottom = np.zeros(len(names))
    for band in bands:
        vals = [arch_data[n]["tou"].reindex(bands).loc[band, "cost_eur"]
                for n in names]
        ax.bar(x, vals, 0.5, bottom=bottom, label=band, color=BAND_COLORS[band])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("€ / year")
    ax.set_title("Electricity Cost by TOU Band")
    ax.legend()

    fig.suptitle("TOU Comparison – All Archetypes", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "comparison_tou_split")


# ── C6: Weekly overlay comparison ────────────────────────────────────────────

def plot_comparison_weekly_overlay(
    arch_data: dict,
    week_start: str,
    season_label: str,
    out_dir: Path,
) -> None:
    """Overlaid weekly time series for all archetypes."""
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, d in arch_data.items():
        sel = d["e_el"].loc[start:end]
        ax.plot(sel.index, sel.values, linewidth=1.0, label=name,
                color=_arch_color(name))
    ax.set_ylabel("kWh / h")
    ax.set_xlabel("")
    ax.set_title(
        f"Electricity Import – {season_label} Week "
        f"({start.strftime('%d %b')}–{end.strftime('%d %b %Y')})"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, out_dir, f"comparison_{season_label.lower()}_week")


# ── C7: Total cost breakdown (stacked bar) ──────────────────────────────────

def plot_comparison_cost_breakdown(
    kpi_list: list[dict],
    out_dir: Path,
) -> None:
    """Stacked bar showing electricity vs gas cost per archetype."""
    names = [k["archetype"] for k in kpi_list]
    c_el = [k["C_el_eur"] for k in kpi_list]
    c_gas = [k["C_gas_eur"] for k in kpi_list]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, c_el, 0.5, label="Electricity (€)", color="#457b9d")
    ax.bar(x, c_gas, 0.5, bottom=c_el, label="Gas (€)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("€ / year")
    ax.set_title("Total Baseline Cost Breakdown (C₀) – Per Archetype")
    ax.legend()

    # Annotate totals
    for i, (el, gas) in enumerate(zip(c_el, c_gas)):
        ax.text(i, el + gas + 10, f"€{el + gas:,.0f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir, "comparison_cost_breakdown")


# ── C8: Pie charts – share of total / electricity / gas cost ─────────────────

def _pie_chart(values, labels, colors, title, out_dir, filename):
    """Helper: single pie chart with percentages and absolute values."""
    fig, ax = plt.subplots(figsize=(7, 7))

    def _autopct(pct, allvals):
        absolute = pct / 100.0 * sum(allvals)
        return f"{pct:.1f}%\n(€{absolute:,.0f})"

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: _autopct(pct, values),
        startangle=140,
        pctdistance=0.72,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out_dir, filename)


def plot_comparison_pie_charts(
    kpi_list: list[dict],
    out_dir: Path,
) -> None:
    """Three pie charts comparing archetypes: total cost, electricity cost,
    gas cost."""
    names = [k["archetype"] for k in kpi_list]
    colors = [_arch_color(n) for n in names]

    # Total cost (C0)
    c0_vals = [k["C0_eur"] for k in kpi_list]
    _pie_chart(c0_vals, names, colors,
               "Total Baseline Cost (C₀) – Share by Archetype",
               out_dir, "comparison_pie_total_cost")

    # Electricity cost
    cel_vals = [k["C_el_eur"] for k in kpi_list]
    _pie_chart(cel_vals, names, colors,
               "Electricity Cost – Share by Archetype",
               out_dir, "comparison_pie_electricity_cost")

    # Gas cost
    cgas_vals = [k["C_gas_eur"] for k in kpi_list]
    _pie_chart(cgas_vals, names, colors,
               "Gas Cost – Share by Archetype",
               out_dir, "comparison_pie_gas_cost")

    # Electricity energy
    eel_vals = [k["E_el_kWh"] for k in kpi_list]
    _pie_chart_energy(eel_vals, names, colors,
                      "Electricity Consumption – Share by Archetype",
                      out_dir, "comparison_pie_electricity_energy")

    # Gas energy
    gkwh_vals = [k["G_kWh"] for k in kpi_list]
    _pie_chart_energy(gkwh_vals, names, colors,
                      "Gas Consumption – Share by Archetype",
                      out_dir, "comparison_pie_gas_energy")


def _pie_chart_energy(values, labels, colors, title, out_dir, filename):
    """Helper: pie chart for energy (kWh) with percentages + absolute."""
    fig, ax = plt.subplots(figsize=(7, 7))

    def _autopct(pct, allvals):
        absolute = pct / 100.0 * sum(allvals)
        return f"{pct:.1f}%\n({absolute:,.0f} kWh)"

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: _autopct(pct, values),
        startangle=140,
        pctdistance=0.72,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out_dir, filename)


# ── master comparison function ───────────────────────────────────────────────

def generate_comparison_plots(
    arch_data: dict,
    kpi_list: list[dict],
    representative_weeks: dict[str, str],
    out_dir: Path,
) -> None:
    """Generate all cross-archetype comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("  Generating cross-archetype comparison plots …")

    plot_comparison_annual_kpis(kpi_list, out_dir)
    plot_comparison_monthly_electricity(arch_data, out_dir)
    plot_comparison_monthly_gas(arch_data, out_dir)
    plot_comparison_load_duration(arch_data, out_dir)
    plot_comparison_tou(arch_data, out_dir)
    plot_comparison_cost_breakdown(kpi_list, out_dir)
    plot_comparison_pie_charts(kpi_list, out_dir)

    for label, start in representative_weeks.items():
        plot_comparison_weekly_overlay(arch_data, start, label.capitalize(), out_dir)

    print("  ✓ All comparison plots saved.")

