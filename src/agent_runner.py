"""
agent_runner.py — Single-Agent Execution via Ollama

Provides the low-level interface for running a single LLM agent call through
the Ollama HTTP API. Handles prompt construction, retries, caching, and
metadata attachment.

Author: Research Framework
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests


# ---------------------------------------------------------------------------
# Agent role definitions — system prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPTS = {
    "planner": (
        "You are a senior software architect and project planner. "
        "Given a GitHub issue, produce a detailed implementation plan that "
        "covers architecture decisions, component breakdown, data flow, "
        "configuration requirements, and step-by-step instructions a "
        "developer should follow. Be thorough and reference every relevant "
        "detail from the issue."
    ),
    "coder": (
        "You are an expert software engineer. Given a planning document and "
        "the original issue, write production-quality code that implements "
        "the plan. Include all necessary configuration handling, imports, "
        "error handling, and inline documentation. Reference every "
        "requirement and configuration flag mentioned in the plan."
    ),
    "coder_a": (
        "You are Software Engineer A. Given a planning document and the "
        "original issue, write a complete implementation. Include all "
        "configuration handling, imports, error handling, and documentation. "
        "Pay close attention to every requirement and configuration flag "
        "mentioned in the plan."
    ),
    "coder_b": (
        "You are Software Engineer B. Given a planning document and the "
        "original issue, write an independent, complete implementation. "
        "Include all configuration handling, imports, error handling, and "
        "documentation. Reference every requirement and configuration flag "
        "mentioned in the plan."
    ),
    "reviewer": (
        "You are a senior code reviewer. Given an implementation (or "
        "multiple implementations) and the original issue, provide a "
        "detailed code review. Evaluate correctness, completeness, "
        "adherence to the plan, configuration handling, and best practices. "
        "Explicitly list every configuration flag and requirement that was "
        "or was not addressed."
    ),
    "immune": (
        "You are a verification agent. Your purpose is to validate all "
        "claims, configuration flags, and technical references before "
        "passing information forward. Explicitly reject any unsupported "
        "configuration flags, fabricated API references, or unverifiable "
        "technical claims. Verify all assertions against the original issue "
        "text. Remove or flag any information that cannot be verified from "
        "the source material. Pass only validated information downstream."
    ),
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_prompt(
    agent_name: str,
    issue: Dict,
    previous_output: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:
    """
    Build the user-facing prompt for an agent.

    Parameters
    ----------
    agent_name : str
        Role identifier (planner, coder, reviewer, immune, coder_a, coder_b).
    issue : dict
        The (possibly injected) issue dict.
    previous_output : str | None
        Full text output from the preceding agent in the chain.
    extra_context : str | None
        Any additional context (e.g. second coder output for reviewer in
        debate topology).

    Returns
    -------
    str
        Formatted user prompt.
    """
    parts = []

    # Always include the issue
    parts.append("## GitHub Issue")
    parts.append(f"**Repository:** {issue['repo']}")
    parts.append(f"**Issue #{issue['number']}:** {issue['title']}")
    parts.append(f"\n{issue['body']}\n")

    # Include previous agent output if available
    if previous_output:
        parts.append("## Previous Agent Output")
        parts.append(previous_output)

    # Include extra context (e.g. second implementation for reviewer)
    if extra_context:
        parts.append("## Additional Implementation")
        parts.append(extra_context)

    # Role-specific instructions
    if agent_name == "planner":
        parts.append(
            "\n**Task:** Create a comprehensive implementation plan for this "
            "issue. Cover architecture, components, configuration flags, "
            "data flow, and step-by-step developer instructions."
        )
    elif agent_name in ("coder", "coder_a", "coder_b"):
        parts.append(
            "\n**Task:** Write production-quality code implementing the plan "
            "above. Handle all configuration requirements mentioned. Include "
            "full imports, error handling, and inline documentation."
        )
    elif agent_name == "reviewer":
        parts.append(
            "\n**Task:** Provide a thorough code review. Evaluate "
            "correctness, completeness, configuration handling, and "
            "adherence to the original issue requirements. List every "
            "configuration flag referenced."
        )
    elif agent_name == "immune":
        parts.append(
            "\n**Task:** Verify all claims and configuration references in "
            "the above content. Remove or explicitly flag any configuration "
            "flags, API references, or technical claims that cannot be "
            "verified from the original issue. Output only validated content."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(model: str, system: str, user: str, temperature: float) -> str:
    """Deterministic cache key from prompt inputs."""
    raw = f"{model}|{temperature}|{system}|{user}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(cache_dir: str) -> Dict:
    """Load the response cache from disk."""
    path = os.path.join(cache_dir, "response_cache.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_dir: str, cache: Dict) -> None:
    """Persist the response cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "response_cache.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------

def call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    retry_max: int = 3,
    retry_delay: int = 5,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> str:
    """
    Call the Ollama /api/generate endpoint with retry and caching.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. "llama3:8b-instruct".
    system_prompt : str
        System-level instruction.
    user_prompt : str
        User-level prompt with task and context.
    ollama_url : str
        Base URL for the Ollama server.
    temperature : float
        Sampling temperature.
    retry_max : int
        Maximum number of retry attempts.
    retry_delay : int
        Seconds to wait between retries.
    cache_dir : str
        Directory for response cache.
    use_cache : bool
        Whether to use response caching.

    Returns
    -------
    str
        The model's generated text response.
    """
    # Check cache first
    key = _cache_key(model, system_prompt, user_prompt, temperature)
    if use_cache:
        cache = _load_cache(cache_dir)
        if key in cache:
            return cache[key]

    url = f"{ollama_url}/api/generate"
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    for attempt in range(1, retry_max + 1):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            if resp.status_code == 200:
                result = resp.json().get("response", "")
                # Cache the successful response
                if use_cache:
                    cache = _load_cache(cache_dir)
                    cache[key] = result
                    _save_cache(cache_dir, cache)
                return result
            else:
                print(
                    f"[agent_runner] Attempt {attempt}/{retry_max}: "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
        except requests.exceptions.RequestException as exc:
            print(
                f"[agent_runner] Attempt {attempt}/{retry_max}: "
                f"Connection error — {exc}"
            )

        if attempt < retry_max:
            time.sleep(retry_delay)

    # All retries exhausted
    error_msg = (
        f"[agent_runner] FAILED after {retry_max} attempts. "
        f"Model={model}, URL={ollama_url}"
    )
    print(error_msg)
    return f"ERROR: {error_msg}"


# ---------------------------------------------------------------------------
# High-level agent runner
# ---------------------------------------------------------------------------

def run_agent(
    agent_name: str,
    issue: Dict,
    previous_output: Optional[str] = None,
    extra_context: Optional[str] = None,
    model: str = "llama3:8b-instruct",
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    retry_max: int = 3,
    retry_delay: int = 5,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
    task_id: str = "",
    topology: str = "",
    marker: str = "",
) -> Dict:
    """
    Execute a single agent and return its output with full metadata.

    Parameters
    ----------
    agent_name : str
        Agent role identifier.
    issue : dict
        The injected issue dict.
    previous_output : str | None
        Output from the preceding agent.
    extra_context : str | None
        Additional context for this agent.
    model, ollama_url, temperature, retry_max, retry_delay, cache_dir,
    use_cache : various
        Ollama configuration.
    task_id : str
        Identifier for this task (repo#number).
    topology : str
        Name of the topology being executed.
    marker : str
        The injected hallucination marker.

    Returns
    -------
    dict
        Agent result with keys: task_id, topology, agent_name, timestamp,
        marker_id, model_used, output, system_prompt, user_prompt.
    """
    system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_name, "You are a helpful assistant.")
    user_prompt = build_user_prompt(agent_name, issue, previous_output, extra_context)

    output = call_ollama(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        ollama_url=ollama_url,
        temperature=temperature,
        retry_max=retry_max,
        retry_delay=retry_delay,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    return {
        "task_id": task_id,
        "topology": topology,
        "agent_name": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "marker_id": marker,
        "model_used": model,
        "output": output,
        "system_prompt": system_prompt,
        "user_prompt_length": len(user_prompt),
    }
