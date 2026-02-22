"""
main.py — CLI Entry Point

Orchestrates the full hallucination contagion research pipeline:
  1. Fetch real GitHub issues
  2. Inject synthetic hallucination markers
  3. Run experiments across topologies
  4. Compute metrics
  5. Generate analysis (plots, CSV, report, t-test)

Usage:
    python -m src.main --help
    python -m src.main --num_tasks 5 --topology linear debate
    python -m src.main --analysis_only

Author: Research Framework
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

from src.data_loader import fetch_all_issues
from src.injection_engine import inject_all
from src.experiment_runner import run_experiments
from src.metrics_engine import compute_aggregate_metrics, compute_immune_efficacy
from src.analysis import run_full_analysis


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """Load the experiment configuration from disk."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[main] Config file not found at {config_path}, using defaults.")
    return {}


def save_run_config(config: dict, output_dir: str = "results") -> str:
    """Save the effective configuration for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Hallucination Contagion Research Framework — "
            "Empirically evaluate hallucination spread in multi-agent LLM systems."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.main --num_tasks 10\n"
            "  python -m src.main --topology linear debate --temperature 0.5\n"
            "  python -m src.main --analysis_only\n"
            "  python -m src.main --skip_fetch --num_tasks 5\n"
        ),
    )

    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--topology", nargs="+",
        choices=["linear", "debate", "linear_immune", "linear_immune_clean"],
        help="Topologies to run (default: all three)",
    )
    parser.add_argument(
        "--num_tasks", type=int,
        help="Number of tasks (issues) to process (default: from config)",
    )
    parser.add_argument(
        "--temperature", type=float,
        help="Sampling temperature (default: from config)",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility (default: from config)",
    )
    parser.add_argument(
        "--model", type=str,
        help="Ollama model name (default: from config)",
    )
    parser.add_argument(
        "--skip_fetch", action="store_true",
        help="Skip GitHub issue fetching (use cached data)",
    )
    parser.add_argument(
        "--force_fetch", action="store_true",
        help="Force re-fetch of GitHub issues even if cached",
    )
    parser.add_argument(
        "--analysis_only", action="store_true",
        help="Skip experiments and only regenerate analysis from existing data",
    )
    parser.add_argument(
        "--run_id", type=str,
        help="Reuse a specific run ID (for reproducing a previous run)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: results)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full research pipeline."""
    args = parse_args()

    # ── Load and merge configuration ────────────────────────────────────
    config = load_config(args.config)

    # CLI overrides
    if args.topology:
        config["topologies"] = args.topology
    if args.num_tasks is not None:
        config["num_tasks"] = args.num_tasks
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model:
        config["model"] = args.model
    if args.output_dir:
        config["output_dir"] = args.output_dir

    output_dir = config.get("output_dir", "results")
    data_dir = config.get("data_dir", "data")
    seed = config.get("seed", 42)

    # Set random seed
    random.seed(seed)

    print("=" * 70)
    print("  HALLUCINATION CONTAGION RESEARCH FRAMEWORK")
    print("=" * 70)
    print(f"  Model       : {config.get('model', 'llama3:8b')}")
    print(f"  Temperature : {config.get('temperature', 0.0)}")
    print(f"  Seed        : {seed}")
    print(f"  Topologies  : {config.get('topologies', ['linear', 'debate', 'linear_immune', 'linear_immune_clean'])}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Timestamp   : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # ── Analysis-only mode ──────────────────────────────────────────────
    if args.analysis_only:
        print("\n[main] Analysis-only mode — loading existing results …")
        return _run_analysis_only(config, output_dir)

    # ── Step 1: Fetch GitHub issues ─────────────────────────────────────
    if args.skip_fetch:
        print("\n[main] Skipping GitHub fetch — using cached issues …")
    else:
        print("\n[main] Step 1/5: Fetching GitHub issues …")

    issues = fetch_all_issues(
        repos=config.get("repos"),
        issues_per_repo=config.get("issues_per_repo", 10),
        min_body_length=config.get("min_issue_body_length", 100),
        max_total=config.get("max_issues", 50),
        data_dir=data_dir,
        force_refresh=args.force_fetch,
    )

    if not issues:
        print("[main] ERROR: No issues loaded. Aborting.")
        sys.exit(1)

    print(f"[main] Loaded {len(issues)} issues.")

    # ── Step 2: Inject hallucination markers ────────────────────────────
    print("\n[main] Step 2/5: Injecting hallucination markers …")
    run_id = args.run_id
    injected_issues, marker, run_id = inject_all(
        issues,
        marker_base=config.get("marker_base", "ENABLE_QUERY_BATCHING"),
        run_id=run_id,
    )

    config["marker"] = marker
    config["run_id"] = run_id

    # Save injected issues
    injected_path = os.path.join(data_dir, "injected_issues.json")
    os.makedirs(data_dir, exist_ok=True)
    with open(injected_path, "w", encoding="utf-8") as f:
        json.dump(injected_issues, f, indent=2, ensure_ascii=False)
    print(f"[main] Saved injected issues → {injected_path}")

    # ── Step 3: Run experiments ─────────────────────────────────────────
    print("\n[main] Step 3/5: Running experiments …")
    topologies = config.get("topologies", ["linear", "debate", "linear_immune", "linear_immune_clean"])
    num_tasks = config.get("num_tasks", 30)

    experiment_results = run_experiments(
        issues=injected_issues,
        marker=marker,
        topologies=topologies,
        config=config,
        output_dir=output_dir,
        num_tasks=num_tasks,
    )

    # ── Step 4: Compute metrics ─────────────────────────────────────────
    print("\n[main] Step 4/5: Computing metrics …")
    task_metrics = experiment_results["task_metrics"]
    aggregate = experiment_results["aggregate_metrics"]
    immune_eff = compute_immune_efficacy(aggregate)

    if immune_eff:
        eff_path = os.path.join(output_dir, "immune_efficacy.json")
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(immune_eff, f, indent=2)
        print(f"[main] Immune efficacy → {eff_path}")

    # ── Step 5: Generate analysis ───────────────────────────────────────
    print("\n[main] Step 5/5: Generating analysis …")
    analysis_results = run_full_analysis(
        task_metrics=task_metrics,
        aggregate=aggregate,
        immune_efficacy=immune_eff,
        marker=marker,
        run_id=run_id,
        config=config,
        output_dir=output_dir,
    )

    # Save run config for reproducibility
    save_run_config(config, output_dir)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Raw outputs  : {output_dir}/raw_outputs.json")
    print(f"  Metrics      : {output_dir}/task_metrics.json")
    print(f"  Aggregates   : {output_dir}/aggregate_metrics.json")
    print(f"  Report       : {output_dir}/report.md")
    print(f"  CSV summary  : {output_dir}/summary.csv")
    print(f"  Plots        : {output_dir}/plots/")
    print(f"  Run config   : {output_dir}/run_config.json")

    if analysis_results.get("ttest"):
        tt = analysis_results["ttest"]
        print(f"\n  t-test result : t={tt['t_statistic']}, p={tt['p_value']}")
        print(f"  Significant   : {'Yes' if tt['significant_at_005'] else 'No'} (α=0.05)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Analysis-only mode helper
# ---------------------------------------------------------------------------

def _run_analysis_only(config: dict, output_dir: str) -> None:
    """Regenerate analysis from existing metric files."""
    # Load existing metrics
    metrics_path = os.path.join(output_dir, "task_metrics.json")
    agg_path = os.path.join(output_dir, "aggregate_metrics.json")

    if not os.path.exists(metrics_path):
        print(f"[main] ERROR: {metrics_path} not found. Run experiments first.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        task_metrics = json.load(f)

    if os.path.exists(agg_path):
        with open(agg_path, "r", encoding="utf-8") as f:
            aggregate = json.load(f)
    else:
        aggregate = compute_aggregate_metrics(task_metrics)

    immune_eff = compute_immune_efficacy(aggregate)
    marker = config.get("marker", "UNKNOWN_MARKER")
    run_id = config.get("run_id", "UNKNOWN")

    run_full_analysis(
        task_metrics=task_metrics,
        aggregate=aggregate,
        immune_efficacy=immune_eff,
        marker=marker,
        run_id=run_id,
        config=config,
        output_dir=output_dir,
    )

    print("\n[main] Analysis regeneration complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
