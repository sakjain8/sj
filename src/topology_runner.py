"""
topology_runner.py — Multi-Agent Topology Orchestration

Implements three agent network topologies for hallucination contagion
experiments:
  1. Linear:        Planner → Coder → Reviewer
  2. Debate:        Planner → (Coder_A ‖ Coder_B) → Reviewer
  3. Linear+Immune: Planner → Immune → Coder → Reviewer

Each topology passes the full output of the previous agent(s) to the next,
enabling the study of how hallucinated artifacts propagate through the
network.

Author: Research Framework
"""

from typing import Dict, List
import re

from src.agent_runner import run_agent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _agent_kwargs(config: Dict) -> Dict:
    """Extract common agent parameters from the experiment config."""
    return {
        "model": config.get("model", "llama3:8b"),
        "ollama_url": config.get("ollama_url", "http://localhost:11434"),
        "temperature": config.get("temperature", 0.0),
        "retry_max": config.get("retry_max", 3),
        "retry_delay": config.get("retry_delay_seconds", 5),
        "cache_dir": config.get("cache_dir", "data/cache"),
        "use_cache": config.get("cache_responses", True),
    }


def _task_id(issue: Dict) -> str:
    """Construct a deterministic task identifier from an issue."""
    return f"{issue['repo']}#{issue['number']}"


# ---------------------------------------------------------------------------
# Topology: Linear Chain
# ---------------------------------------------------------------------------

def run_linear(issue: Dict, config: Dict) -> List[Dict]:
    """
    Execute the linear chain topology: Planner → Coder → Reviewer.

    Parameters
    ----------
    issue : dict
        Injected issue dict.
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        List of agent result dicts in execution order.
    """
    tid = _task_id(issue)
    marker = issue.get("marker", "")
    kw = _agent_kwargs(config)
    results: List[Dict] = []

    # Step 1 — Planner
    planner = run_agent(
        agent_name="planner",
        issue=issue,
        task_id=tid,
        topology="linear",
        marker=marker,
        **kw,
    )
    results.append(planner)

    # Step 2 — Coder
    coder = run_agent(
        agent_name="coder",
        issue=issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="linear",
        marker=marker,
        **kw,
    )
    results.append(coder)

    # Step 3 — Reviewer
    reviewer = run_agent(
        agent_name="reviewer",
        issue=issue,
        previous_output=coder["output"],
        task_id=tid,
        topology="linear",
        marker=marker,
        **kw,
    )
    results.append(reviewer)

    return results


# ---------------------------------------------------------------------------
# Topology: Debate
# ---------------------------------------------------------------------------

def run_debate(issue: Dict, config: Dict) -> List[Dict]:
    """
    Execute the debate topology:
    Planner → (Coder_A ‖ Coder_B) → Reviewer.

    Reviewer receives both implementations.

    Parameters
    ----------
    issue : dict
        Injected issue dict.
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        List of agent result dicts in execution order.
    """
    tid = _task_id(issue)
    marker = issue.get("marker", "")
    kw = _agent_kwargs(config)
    results: List[Dict] = []

    # Step 1 — Planner
    planner = run_agent(
        agent_name="planner",
        issue=issue,
        task_id=tid,
        topology="debate",
        marker=marker,
        **kw,
    )
    results.append(planner)

    # Step 2a — Coder A (independent)
    coder_a = run_agent(
        agent_name="coder_a",
        issue=issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="debate",
        marker=marker,
        **kw,
    )
    results.append(coder_a)

    # Step 2b — Coder B (independent)
    coder_b = run_agent(
        agent_name="coder_b",
        issue=issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="debate",
        marker=marker,
        **kw,
    )
    results.append(coder_b)

    # Step 3 — Reviewer sees both implementations
    reviewer = run_agent(
        agent_name="reviewer",
        issue=issue,
        previous_output=coder_a["output"],
        extra_context=coder_b["output"],
        task_id=tid,
        topology="debate",
        marker=marker,
        **kw,
    )
    results.append(reviewer)

    return results


# ---------------------------------------------------------------------------
# Topology: Linear + Immune Agent
# ---------------------------------------------------------------------------

def run_linear_immune(issue: Dict, config: Dict) -> List[Dict]:
    """
    Execute the linear chain with an immune verification agent:
    Planner → Immune → Coder → Reviewer.

    The immune agent filters unverifiable claims before the coder sees them.

    Parameters
    ----------
    issue : dict
        Injected issue dict.
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        List of agent result dicts in execution order.
    """
    tid = _task_id(issue)
    marker = issue.get("marker", "")
    kw = _agent_kwargs(config)
    results: List[Dict] = []

    # Step 1 — Planner
    planner = run_agent(
        agent_name="planner",
        issue=issue,
        task_id=tid,
        topology="linear_immune",
        marker=marker,
        **kw,
    )
    results.append(planner)

    # Step 2 — Immune verification agent
    immune = run_agent(
        agent_name="immune",
        issue=issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="linear_immune",
        marker=marker,
        **kw,
    )
    results.append(immune)

    # Step 3 — Coder receives immune-filtered output
    coder = run_agent(
        agent_name="coder",
        issue=issue,
        previous_output=immune["output"],
        task_id=tid,
        topology="linear_immune",
        marker=marker,
        **kw,
    )
    results.append(coder)

    # Step 4 — Reviewer
    reviewer = run_agent(
        agent_name="reviewer",
        issue=issue,
        previous_output=coder["output"],
        task_id=tid,
        topology="linear_immune",
        marker=marker,
        **kw,
    )
    results.append(reviewer)

    return results


# ---------------------------------------------------------------------------
# Topology: Linear + Immune Agent (Clean Ground Truth)
# ---------------------------------------------------------------------------

def run_linear_immune_clean(issue: Dict, config: Dict) -> List[Dict]:
    """
    Execute the linear chain with an immune agent that has clean ground truth.

    Unlike ``run_linear_immune``, the immune agent receives the **original,
    non-injected** issue body so it can cross-reference the planner's output
    against uncontaminated source material.

    Flow: Planner(injected) → Immune(clean issue) → Coder → Reviewer

    Parameters
    ----------
    issue : dict
        Injected issue dict (must contain ``body_original``).
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        List of agent result dicts in execution order.
    """
    import copy

    tid = _task_id(issue)
    marker = issue.get("marker", "")
    kw = _agent_kwargs(config)
    results: List[Dict] = []

    # Step 1 — Planner (receives injected issue as usual)
    planner = run_agent(
        agent_name="planner",
        issue=issue,
        task_id=tid,
        topology="linear_immune_clean",
        marker=marker,
        **kw,
    )
    results.append(planner)

    # Build a clean copy of the issue using the original body
    clean_issue = copy.deepcopy(issue)
    if "body_original" in clean_issue:
        clean_issue["body"] = clean_issue["body_original"]

    # Step 2 — Immune agent receives CLEAN issue + infected planner output
    immune = run_agent(
        agent_name="immune",
        issue=clean_issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="linear_immune_clean",
        marker=marker,
        **kw,
    )
    results.append(immune)

    # Step 3 — Coder receives immune-filtered output (with injected issue)
    coder = run_agent(
        agent_name="coder",
        issue=issue,
        previous_output=immune["output"],
        task_id=tid,
        topology="linear_immune_clean",
        marker=marker,
        **kw,
    )
    results.append(coder)

    # Step 4 — Reviewer
    reviewer = run_agent(
        agent_name="reviewer",
        issue=issue,
        previous_output=coder["output"],
        task_id=tid,
        topology="linear_immune_clean",
        marker=marker,
        **kw,
    )
    results.append(reviewer)

    return results


# ---------------------------------------------------------------------------
# Topology: Epidemic (Innate + Adaptive Immune System)
# ---------------------------------------------------------------------------

def _parse_confidence(output: str) -> str:
    """
    Parse the confidence signal from the innate verifier's output.

    Searches for 'CONFIDENCE: HIGH' or 'CONFIDENCE: LOW' in the output.
    Defaults to 'LOW' if ambiguous or missing (fail-safe: assume infected).

    Returns
    -------
    str
        'HIGH' or 'LOW'.
    """
    text = output.upper()
    # Look for explicit confidence lines
    if re.search(r"CONFIDENCE\s*:\s*HIGH", text):
        return "HIGH"
    if re.search(r"CONFIDENCE\s*:\s*LOW", text):
        return "LOW"
    # Fail-safe: if no clear signal, default to LOW (trigger adaptive)
    return "LOW"


def run_epidemic(issue: Dict, config: Dict) -> List[Dict]:
    """
    Execute the Epidemic (Innate + Adaptive Immune System) topology.

    Models hallucination contagion mitigation as a biological immune
    response:

        Planner → Innate Verifier (Coder + self-check)
                    ├─ HIGH confidence → Reviewer   (healthy pathway)
                    └─ LOW confidence  → Adaptive Immune → Coder → Reviewer
                                          (quarantine + recovery pathway)

    The Innate Verifier acts as the Host Cell attempting to metabolise
    the Planner's output.  If it detects something suspicious it releases
    an "interferon" (CONFIDENCE: LOW), which activates the Adaptive
    Immune agent (the Macrophage) to sanitise the plan.

    Parameters
    ----------
    issue : dict
        Injected issue dict.
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        List of agent result dicts in execution order, each annotated
        with epidemiological metadata.
    """
    tid = _task_id(issue)
    marker = issue.get("marker", "")
    kw = _agent_kwargs(config)
    results: List[Dict] = []

    # ── Step 1 — Infection: Planner receives the injected issue ──────
    planner = run_agent(
        agent_name="planner",
        issue=issue,
        task_id=tid,
        topology="epidemic",
        marker=marker,
        **kw,
    )
    planner["epidemic_phase"] = "infection"
    planner["innate_confidence"] = None
    planner["adaptive_activated"] = False
    results.append(planner)

    # ── Step 2 — Innate Immune Response: Verifier evaluates the plan ─
    innate = run_agent(
        agent_name="innate_verifier",
        issue=issue,
        previous_output=planner["output"],
        task_id=tid,
        topology="epidemic",
        marker=marker,
        **kw,
    )

    confidence = _parse_confidence(innate["output"])
    innate["epidemic_phase"] = "innate"
    innate["innate_confidence"] = confidence
    innate["adaptive_activated"] = (confidence == "LOW")
    results.append(innate)

    print(
        f"  [epidemic] Innate verifier confidence: {confidence}"
        f" → {'ACTIVATING adaptive immune' if confidence == 'LOW' else 'proceeding to reviewer'}"
    )

    if confidence == "LOW":
        # ── Step 3a — Adaptive Immune: Sanitise the infected plan ────
        adaptive = run_agent(
            agent_name="adaptive_immune",
            issue=issue,
            previous_output=planner["output"],
            task_id=tid,
            topology="epidemic",
            marker=marker,
            **kw,
        )
        adaptive["epidemic_phase"] = "adaptive"
        adaptive["innate_confidence"] = confidence
        adaptive["adaptive_activated"] = True
        results.append(adaptive)

        # ── Step 3b — Recovery: Coder works from sanitised plan ──────
        coder = run_agent(
            agent_name="coder",
            issue=issue,
            previous_output=adaptive["output"],
            task_id=tid,
            topology="epidemic",
            marker=marker,
            **kw,
        )
        coder["epidemic_phase"] = "recovery"
        coder["innate_confidence"] = confidence
        coder["adaptive_activated"] = True
        results.append(coder)

        # ── Step 4a — Reviewer reviews recovered code ────────────────
        reviewer = run_agent(
            agent_name="reviewer",
            issue=issue,
            previous_output=coder["output"],
            task_id=tid,
            topology="epidemic",
            marker=marker,
            **kw,
        )
        reviewer["epidemic_phase"] = "recovered"
        reviewer["innate_confidence"] = confidence
        reviewer["adaptive_activated"] = True
        results.append(reviewer)
    else:
        # ── Step 3c — Healthy: Pass innate output directly to reviewer
        reviewer = run_agent(
            agent_name="reviewer",
            issue=issue,
            previous_output=innate["output"],
            task_id=tid,
            topology="epidemic",
            marker=marker,
            **kw,
        )
        reviewer["epidemic_phase"] = "healthy"
        reviewer["innate_confidence"] = confidence
        reviewer["adaptive_activated"] = False
        results.append(reviewer)

    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOPOLOGY_REGISTRY = {
    "linear": run_linear,
    "debate": run_debate,
    "linear_immune": run_linear_immune,
    "linear_immune_clean": run_linear_immune_clean,
    "epidemic": run_epidemic,
}


def run_topology(
    topology_name: str,
    issue: Dict,
    config: Dict,
) -> List[Dict]:
    """
    Dispatch to the appropriate topology runner.

    Parameters
    ----------
    topology_name : str
        One of "linear", "debate", "linear_immune".
    issue : dict
        Injected issue dict.
    config : dict
        Experiment configuration.

    Returns
    -------
    list[dict]
        Agent results from the topology execution.

    Raises
    ------
    ValueError
        If topology_name is not recognized.
    """
    runner = TOPOLOGY_REGISTRY.get(topology_name)
    if runner is None:
        raise ValueError(
            f"Unknown topology '{topology_name}'. "
            f"Available: {list(TOPOLOGY_REGISTRY.keys())}"
        )
    return runner(issue, config)
