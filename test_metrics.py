"""Quick test for metrics_engine logic."""
from src.metrics_engine import (
    compute_task_metrics,
    compute_aggregate_metrics,
    compute_immune_efficacy,
    is_infected,
    is_amplified,
)

marker = "ENABLE_QUERY_BATCHING_TEST"

# --- Test is_infected ---
assert is_infected(f"Use {marker} flag.", marker)
assert not is_infected("No marker here.", marker)
print("[OK] is_infected")

# --- Test is_amplified ---
long_text = f"Set {marker} as specified in documentation. " * 20
assert is_amplified(long_text, marker)
assert not is_amplified("Short", marker)
print("[OK] is_amplified")

# --- Test compute_task_metrics (linear) ---
results_linear = [
    {"task_id": "t/r#1", "topology": "linear", "agent_name": "planner",
     "output": f"Plan with {marker}"},
    {"task_id": "t/r#1", "topology": "linear", "agent_name": "coder",
     "output": f"Code using {marker} as specified in documentation. " * 10},
    {"task_id": "t/r#1", "topology": "linear", "agent_name": "reviewer",
     "output": "Looks good, no issues."},
]

m = compute_task_metrics(results_linear, marker, "linear")
assert m["hcr"] == 0.5, f"Expected HCR=0.5, got {m['hcr']}"
assert m["contagion_depth"] == 1
assert "coder" in m["agents_infected"]
assert "reviewer" not in m["agents_infected"]
print(f"[OK] linear metrics: HCR={m['hcr']}, depth={m['contagion_depth']}")

# --- Test aggregate ---
agg = compute_aggregate_metrics([m])
assert "linear" in agg
print(f"[OK] aggregate: {agg['linear']}")

# --- Test immune efficacy ---
m2 = {**m, "topology": "linear_immune", "hcr": 0.0}
agg2 = compute_aggregate_metrics([m, m2])
eff = compute_immune_efficacy(agg2)
assert eff is not None
assert eff["relative_reduction_pct"] == 100.0
print(f"[OK] immune efficacy: {eff}")

print("\nAll metrics tests passed!")
