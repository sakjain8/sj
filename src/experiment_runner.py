"""
experiment_runner.py — Full Experiment Orchestrator

Coordinates the end-to-end execution of hallucination contagion experiments:
  1. Iterates over all tasks (injected issues)
  2. Runs each task through specified topologies
  3. Computes per-task metrics
  4. Saves raw outputs and metrics to disk
  5. Skips already-completed task/topology pairs (idempotent)

Author: Research Framework
"""

import json
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from src.topology_runner import run_topology
from src.metrics_engine import compute_task_metrics, compute_aggregate_metrics


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _load_existing_outputs(path: str) -> List[Dict]:
    """Load previously saved raw outputs from disk."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_outputs(path: str, outputs: List[Dict]) -> None:
    """Persist raw outputs to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)


def _completed_keys(outputs: List[Dict]) -> set:
    """
    Build a set of (task_id, topology) keys that have already been completed.

    Each output record is a single agent result; we consider a task/topology
    pair complete when its reviewer agent has produced output.
    """
    keys = set()
    for rec in outputs:
        if rec.get("agent_name") == "reviewer":
            keys.add((rec["task_id"], rec["topology"]))
    return keys


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    issues: List[Dict],
    marker: str,
    topologies: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    output_dir: str = "results",
    num_tasks: Optional[int] = None,
) -> Dict:
    """
    Execute all experiments and return aggregated metrics.

    Parameters
    ----------
    issues : list[dict]
        Injected issue dicts.
    marker : str
        The hallucination marker used in this run.
    topologies : list[str] | None
        Topology names to execute. Defaults to all three.
    config : dict | None
        Experiment configuration.
    output_dir : str
        Directory for saving results.
    num_tasks : int | None
        Limit the number of tasks (issues) to process.

    Returns
    -------
    dict
        Keys: raw_outputs, task_metrics, aggregate_metrics.
    """
    config = config or {}
    topologies = topologies or ["linear", "debate", "linear_immune"]

    # Optionally limit the number of tasks
    task_list = issues[:num_tasks] if num_tasks else issues

    # Load existing outputs for idempotent reruns
    raw_path = os.path.join(output_dir, "raw_outputs.json")
    all_outputs = _load_existing_outputs(raw_path)
    completed = _completed_keys(all_outputs)

    all_task_metrics: List[Dict] = []

    total_runs = len(task_list) * len(topologies)
    progress = tqdm(total=total_runs, desc="Running experiments", unit="run")

    for issue in task_list:
        task_id = f"{issue['repo']}#{issue['number']}"

        for topo in topologies:
            progress.set_postfix_str(f"{topo} | {task_id[:40]}")

            # Skip already-completed task/topology pairs
            if (task_id, topo) in completed:
                progress.update(1)
                # Still compute metrics from existing data
                existing = [
                    r for r in all_outputs
                    if r["task_id"] == task_id and r["topology"] == topo
                ]
                if existing:
                    tm = compute_task_metrics(existing, marker, topo, config)
                    all_task_metrics.append(tm)
                continue

            # Execute the topology
            try:
                results = run_topology(topo, issue, config)
            except Exception as exc:
                print(f"\n[experiment] ERROR on {task_id}/{topo}: {exc}")
                progress.update(1)
                continue

            # Record results
            all_outputs.extend(results)
            completed.add((task_id, topo))

            # Compute per-task metrics
            tm = compute_task_metrics(results, marker, topo, config)
            all_task_metrics.append(tm)

            # Save after each task/topology to guard against crashes
            _save_outputs(raw_path, all_outputs)

            progress.update(1)

    progress.close()

    # Final save
    _save_outputs(raw_path, all_outputs)

    # Save per-task metrics
    metrics_path = os.path.join(output_dir, "task_metrics.json")
    _save_outputs(metrics_path, all_task_metrics)

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(all_task_metrics)

    agg_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n[experiment] Completed {len(all_task_metrics)} task/topology runs.")
    print(f"[experiment] Raw outputs  → {raw_path}")
    print(f"[experiment] Task metrics → {metrics_path}")
    print(f"[experiment] Aggregates   → {agg_path}")

    return {
        "raw_outputs": all_outputs,
        "task_metrics": all_task_metrics,
        "aggregate_metrics": aggregate,
    }
