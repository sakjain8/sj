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
# Dispatcher
# ---------------------------------------------------------------------------

TOPOLOGY_REGISTRY = {
    "linear": run_linear,
    "debate": run_debate,
    "linear_immune": run_linear_immune,
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
