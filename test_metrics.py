"""Quick test for metrics_engine logic, including epidemic topology."""
from src.metrics_engine import (
    compute_task_metrics,
    compute_aggregate_metrics,
    compute_immune_efficacy,
    compute_epidemic_efficacy,
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

# --- Test epidemic topology metrics (LOW confidence path) ---
results_epidemic_low = [
    {"task_id": "t/r#1", "topology": "epidemic", "agent_name": "planner",
     "output": f"Plan with {marker}",
     "epidemic_phase": "infection", "innate_confidence": None,
     "adaptive_activated": False},
    {"task_id": "t/r#1", "topology": "epidemic", "agent_name": "innate_verifier",
     "output": f"CONFIDENCE: LOW\nThe plan references {marker} which seems suspicious.",
     "epidemic_phase": "innate", "innate_confidence": "LOW",
     "adaptive_activated": True},
    {"task_id": "t/r#1", "topology": "epidemic", "agent_name": "adaptive_immune",
     "output": "Sanitized plan without suspicious references.",
     "epidemic_phase": "adaptive", "innate_confidence": "LOW",
     "adaptive_activated": True},
    {"task_id": "t/r#1", "topology": "epidemic", "agent_name": "coder",
     "output": "Clean code implementing the sanitized plan.",
     "epidemic_phase": "recovery", "innate_confidence": "LOW",
     "adaptive_activated": True},
    {"task_id": "t/r#1", "topology": "epidemic", "agent_name": "reviewer",
     "output": "Code looks clean, no suspicious references.",
     "epidemic_phase": "recovered", "innate_confidence": "LOW",
     "adaptive_activated": True},
]

m_epi_low = compute_task_metrics(results_epidemic_low, marker, "epidemic")
# Only innate_verifier mentions the marker (but it's the first downstream agent)
# adaptive_immune, coder, reviewer do NOT mention the marker
assert m_epi_low["innate_confidence"] == "LOW"
assert m_epi_low["adaptive_activated"] is True
print(f"[OK] epidemic LOW metrics: HCR={m_epi_low['hcr']}, "
      f"confidence={m_epi_low['innate_confidence']}, "
      f"adaptive={m_epi_low['adaptive_activated']}")

# --- Test epidemic topology metrics (HIGH confidence path) ---
results_epidemic_high = [
    {"task_id": "t/r#2", "topology": "epidemic", "agent_name": "planner",
     "output": f"Plan with {marker}",
     "epidemic_phase": "infection", "innate_confidence": None,
     "adaptive_activated": False},
    {"task_id": "t/r#2", "topology": "epidemic", "agent_name": "innate_verifier",
     "output": f"CONFIDENCE: HIGH\nAll requirements look legitimate. Code using {marker}...",
     "epidemic_phase": "innate", "innate_confidence": "HIGH",
     "adaptive_activated": False},
    {"task_id": "t/r#2", "topology": "epidemic", "agent_name": "reviewer",
     "output": f"The code uses {marker} as mentioned.",
     "epidemic_phase": "healthy", "innate_confidence": "HIGH",
     "adaptive_activated": False},
]

m_epi_high = compute_task_metrics(results_epidemic_high, marker, "epidemic")
assert m_epi_high["innate_confidence"] == "HIGH"
assert m_epi_high["adaptive_activated"] is False
# Both innate_verifier and reviewer mention the marker, so HCR = 2/2 = 1.0
assert m_epi_high["hcr"] == 1.0, f"Expected HCR=1.0, got {m_epi_high['hcr']}"
print(f"[OK] epidemic HIGH metrics: HCR={m_epi_high['hcr']}, "
      f"confidence={m_epi_high['innate_confidence']}, "
      f"adaptive={m_epi_high['adaptive_activated']}")

# --- Test epidemic efficacy ---
agg3 = compute_aggregate_metrics([m, m_epi_low])
epi_eff = compute_epidemic_efficacy(agg3, [m, m_epi_low])
assert epi_eff is not None
assert "innate_detection_rate" in epi_eff
assert "adaptive_activation_rate" in epi_eff
assert "recovery_rate" in epi_eff
print(f"[OK] epidemic efficacy: {epi_eff}")

# --- Test epidemic efficacy with both LOW and HIGH ---
agg4 = compute_aggregate_metrics([m, m_epi_low, m_epi_high])
epi_eff2 = compute_epidemic_efficacy(agg4, [m, m_epi_low, m_epi_high])
assert epi_eff2 is not None
assert epi_eff2["innate_detection_rate"] == 0.5  # 1 out of 2 epidemic tasks detected
assert epi_eff2["adaptive_activation_rate"] == 0.5  # 1 out of 2 activated adaptive
print(f"[OK] epidemic efficacy (mixed): {epi_eff2}")

print("\nAll metrics tests passed!")
