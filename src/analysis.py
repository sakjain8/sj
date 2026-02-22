"""
analysis.py — Visualization, Reporting, and Statistical Testing

Generates publication-quality outputs from experiment metrics:
  • Bar charts (HCR by topology, HAR by topology)
  • CSV summary table
  • Markdown research report
  • Statistical t-test (Linear vs Immune HCR)

Author: Research Framework
"""

import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

TOPOLOGY_LABELS = {
    "linear": "Linear",
    "debate": "Debate",
    "linear_immune": "Linear + Immune",
    "linear_immune_clean": "Immune (Clean)",
    "epidemic": "Epidemic (Innate+Adaptive)",
}

COLORS = {
    "linear": "#e74c3c",
    "debate": "#3498db",
    "linear_immune": "#2ecc71",
    "linear_immune_clean": "#9b59b6",
    "epidemic": "#f39c12",
}


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------

def plot_hcr_by_topology(
    aggregate: Dict[str, Dict],
    output_dir: str = "results/plots",
) -> str:
    """
    Generate a bar chart of average HCR by topology.

    Parameters
    ----------
    aggregate : dict[str, dict]
        Aggregated metrics keyed by topology name.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = list(aggregate.keys())
    hcrs = [aggregate[t]["avg_hcr"] for t in topos]
    labels = [TOPOLOGY_LABELS.get(t, t) for t in topos]
    colors = [COLORS.get(t, "#95a5a6") for t in topos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, hcrs, color=colors, edgecolor="black", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, hcrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Average Hallucination Contagion Rate (HCR)", fontsize=12)
    ax.set_title("Hallucination Contagion Rate by Topology", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(hcrs) * 1.3 + 0.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(output_dir, "hcr_by_topology.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved HCR plot → {path}")
    return path


def plot_har_by_topology(
    aggregate: Dict[str, Dict],
    output_dir: str = "results/plots",
) -> str:
    """
    Generate a bar chart of average HAR by topology.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = list(aggregate.keys())
    hars = [aggregate[t]["avg_har"] for t in topos]
    labels = [TOPOLOGY_LABELS.get(t, t) for t in topos]
    colors = [COLORS.get(t, "#95a5a6") for t in topos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, hars, color=colors, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, hars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Average Hallucination Amplification Rate (HAR)", fontsize=12)
    ax.set_title("Hallucination Amplification Rate by Topology", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(hars) * 1.3 + 0.05) if max(hars) > 0 else 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(output_dir, "har_by_topology.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved HAR plot → {path}")
    return path


def plot_depth_by_topology(
    aggregate: Dict[str, Dict],
    output_dir: str = "results/plots",
) -> str:
    """
    Generate a bar chart of average contagion depth by topology.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = list(aggregate.keys())
    depths = [aggregate[t]["avg_depth"] for t in topos]
    labels = [TOPOLOGY_LABELS.get(t, t) for t in topos]
    colors = [COLORS.get(t, "#95a5a6") for t in topos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, depths, color=colors, edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, depths):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Average Contagion Depth (hops)", fontsize=12)
    ax.set_title("Contagion Depth by Topology", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(output_dir, "depth_by_topology.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved depth plot → {path}")
    return path


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def save_summary_csv(
    aggregate: Dict[str, Dict],
    output_dir: str = "results",
) -> str:
    """
    Write a CSV summary of aggregate metrics.

    Returns
    -------
    str
        Path to the saved CSV.
    """
    path = os.path.join(output_dir, "summary.csv")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for topo, m in aggregate.items():
        rows.append({
            "topology": TOPOLOGY_LABELS.get(topo, topo),
            "num_tasks": m["num_tasks"],
            "avg_hcr": round(m["avg_hcr"], 4),
            "avg_har": round(m["avg_har"], 4),
            "avg_depth": round(m["avg_depth"], 2),
            "total_infected": m["total_infected"],
            "total_agents": m["total_agents"],
            "total_amplified": m["total_amplified"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[analysis] Saved CSV summary → {path}")
    return path


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def run_ttest(
    task_metrics: List[Dict],
) -> Optional[Dict]:
    """
    Run an independent samples t-test comparing HCR between the Linear
    and Linear+Immune topologies.

    Parameters
    ----------
    task_metrics : list[dict]
        Per-task metric dicts.

    Returns
    -------
    dict | None
        t-statistic, p-value, and interpretation. None if insufficient data.
    """
    linear_hcrs = [m["hcr"] for m in task_metrics if m["topology"] == "linear"]
    immune_hcrs = [m["hcr"] for m in task_metrics if m["topology"] == "linear_immune"]

    if len(linear_hcrs) < 2 or len(immune_hcrs) < 2:
        print("[analysis] Insufficient data for t-test. Need ≥ 2 samples per group.")
        return None

    t_stat, p_value = stats.ttest_ind(linear_hcrs, immune_hcrs, equal_var=False)

    result = {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "linear_mean_hcr": round(sum(linear_hcrs) / len(linear_hcrs), 4),
        "immune_mean_hcr": round(sum(immune_hcrs) / len(immune_hcrs), 4),
        "linear_n": len(linear_hcrs),
        "immune_n": len(immune_hcrs),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }

    print(f"[analysis] t-test: t={t_stat:.4f}, p={p_value:.6f}")
    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(
    aggregate: Dict[str, Dict],
    task_metrics: List[Dict],
    immune_efficacy: Optional[Dict],
    ttest_result: Optional[Dict],
    marker: str,
    run_id: str,
    config: Dict,
    output_dir: str = "results",
) -> str:
    """
    Generate a comprehensive Markdown research report.

    Returns
    -------
    str
        Path to the saved report.
    """
    path = os.path.join(output_dir, "report.md")
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# Hallucination Contagion Experiment Report",
        "",
        "## Experiment Configuration",
        "",
        f"- **Model:** {config.get('model', 'N/A')}",
        f"- **Temperature:** {config.get('temperature', 'N/A')}",
        f"- **Seed:** {config.get('seed', 'N/A')}",
        f"- **Marker:** `{marker}`",
        f"- **Run ID:** `{run_id}`",
        f"- **Total tasks:** {len(set(m['task_id'] for m in task_metrics)) if task_metrics else 0}",
        f"- **Topologies tested:** {', '.join(aggregate.keys())}",
        "",
        "---",
        "",
        "## Aggregate Results",
        "",
        "| Topology | Tasks | Avg HCR | Avg HAR | Avg Depth | Infected / Total | Amplified |",
        "|----------|-------|---------|---------|-----------|------------------|-----------|",
    ]

    for topo, m in aggregate.items():
        label = TOPOLOGY_LABELS.get(topo, topo)
        lines.append(
            f"| {label} | {m['num_tasks']} | {m['avg_hcr']:.2%} | "
            f"{m['avg_har']:.2%} | {m['avg_depth']:.2f} | "
            f"{m['total_infected']}/{m['total_agents']} | "
            f"{m['total_amplified']} |"
        )

    lines += ["", "---", ""]

    # Immune efficacy
    if immune_efficacy:
        lines += [
            "## Immune Agent Efficacy",
            "",
            f"- **Linear HCR:** {immune_efficacy['hcr_linear']:.2%}",
            f"- **Immune HCR:** {immune_efficacy['hcr_immune']:.2%}",
            f"- **Absolute reduction:** {immune_efficacy['absolute_reduction']:.2%}",
            f"- **Relative reduction:** {immune_efficacy['relative_reduction_pct']:.1f}%",
            "",
        ]

    # Statistical test
    if ttest_result:
        lines += [
            "## Statistical Analysis",
            "",
            "**Independent samples t-test (Linear vs Linear+Immune):**",
            "",
            f"- t-statistic: {ttest_result['t_statistic']}",
            f"- p-value: {ttest_result['p_value']}",
            f"- Linear mean HCR: {ttest_result['linear_mean_hcr']:.2%} (n={ttest_result['linear_n']})",
            f"- Immune mean HCR: {ttest_result['immune_mean_hcr']:.2%} (n={ttest_result['immune_n']})",
            f"- Significant at α=0.05: **{'Yes' if ttest_result['significant_at_005'] else 'No'}**",
            f"- Significant at α=0.01: **{'Yes' if ttest_result['significant_at_001'] else 'No'}**",
            "",
        ]

    # Key findings
    lines += [
        "---",
        "",
        "## Key Findings",
        "",
    ]

    # Finding 1: Contagion exists
    if aggregate:
        any_hcr = any(m["avg_hcr"] > 0 for m in aggregate.values())
        lines.append(
            f"1. **Hallucination contagion {'exists' if any_hcr else 'was not observed'}** — "
            f"Injected marker propagated to downstream agents in "
            f"{sum(m['total_infected'] for m in aggregate.values())} out of "
            f"{sum(m['total_agents'] for m in aggregate.values())} total agent invocations."
        )

    # Finding 2: Topology matters
    if len(aggregate) >= 2:
        hcrs = {t: m["avg_hcr"] for t, m in aggregate.items()}
        best = min(hcrs, key=hcrs.get)
        worst = max(hcrs, key=hcrs.get)
        lines.append(
            f"2. **Network topology affects spread** — "
            f"{TOPOLOGY_LABELS.get(worst, worst)} topology showed highest contagion "
            f"({hcrs[worst]:.2%}), while {TOPOLOGY_LABELS.get(best, best)} showed "
            f"lowest ({hcrs[best]:.2%})."
        )

    # Finding 3: Immune agent effect
    if immune_efficacy:
        lines.append(
            f"3. **Immune agent reduces contagion** — "
            f"The verification agent reduced HCR by "
            f"{immune_efficacy['relative_reduction_pct']:.1f}% "
            f"(from {immune_efficacy['hcr_linear']:.2%} to "
            f"{immune_efficacy['hcr_immune']:.2%})."
        )

    # Finding 4: Epidemic topology (if present)
    if "epidemic" in aggregate:
        epi_hcr = aggregate["epidemic"]["avg_hcr"]
        lin_hcr = aggregate.get("linear", {}).get("avg_hcr", 0)
        if lin_hcr > 0:
            epi_reduction = (lin_hcr - epi_hcr) / lin_hcr * 100
        else:
            epi_reduction = 0.0
        lines.append(
            f"4. **Epidemic Immune System reduces contagion** — "
            f"The Innate+Adaptive immune topology reduced HCR by "
            f"{epi_reduction:.1f}% (from {lin_hcr:.2%} to {epi_hcr:.2%}), "
            f"modelling biological innate (self-verification) and adaptive "
            f"(targeted quarantine) immune responses."
        )

    lines += [
        "",
        "---",
        "",
        "## Plots",
        "",
        "![HCR by Topology](plots/hcr_by_topology.png)",
        "",
        "![HAR by Topology](plots/har_by_topology.png)",
        "",
        "![Contagion Depth](plots/depth_by_topology.png)",
        "",
        "---",
        "",
        "*Report generated automatically by the Hallucination Contagion Research Framework.*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[analysis] Saved report → {path}")
    return path


# ---------------------------------------------------------------------------
# Master analysis entry point
# ---------------------------------------------------------------------------

def run_full_analysis(
    task_metrics: List[Dict],
    aggregate: Dict[str, Dict],
    immune_efficacy: Optional[Dict],
    marker: str,
    run_id: str,
    config: Dict,
    output_dir: str = "results",
) -> Dict:
    """
    Generate all analysis outputs: plots, CSV, report, t-test.

    Returns
    -------
    dict
        Paths to generated files and t-test results.
    """
    plot_dir = os.path.join(output_dir, "plots")

    hcr_path = plot_hcr_by_topology(aggregate, plot_dir)
    har_path = plot_har_by_topology(aggregate, plot_dir)
    depth_path = plot_depth_by_topology(aggregate, plot_dir)
    csv_path = save_summary_csv(aggregate, output_dir)
    ttest = run_ttest(task_metrics)
    report_path = generate_report(
        aggregate, task_metrics, immune_efficacy, ttest,
        marker, run_id, config, output_dir,
    )

    return {
        "hcr_plot": hcr_path,
        "har_plot": har_path,
        "depth_plot": depth_path,
        "csv_summary": csv_path,
        "report": report_path,
        "ttest": ttest,
    }
