# Hallucination Contagion in Multi-Agent LLM Systems

A reproducible research framework for empirically evaluating how hallucinated artifacts propagate through multi-agent LLM networks using controlled synthetic injection on real-world GitHub issues.

## Research Questions

1. **Does hallucination contagion exist?** — Do fabricated technical claims injected into one agent's context spread to downstream agents?
2. **Does network topology affect spread?** — Do different agent communication structures amplify or reduce contagion?
3. **Can an immune agent reduce contagion?** — Does a dedicated verification agent statistically reduce hallucination propagation?

## Architecture

```
data_loader.py          Fetches real GitHub issues from public repos
injection_engine.py     Injects synthetic hallucination markers
agent_runner.py         Runs individual agents via Ollama
topology_runner.py      Orchestrates agent topologies (linear, debate, immune)
metrics_engine.py       Computes HCR, HAR, contagion depth, immune efficacy
experiment_runner.py    Manages the full experiment loop with caching
analysis.py             Generates plots, CSV, report, and t-test
main.py                 CLI entry point
```

### Agent Topologies

| Topology | Flow | Purpose |
|----------|------|---------|
| **Linear** | Planner → Coder → Reviewer | Baseline contagion chain |
| **Debate** | Planner → (Coder_A ∥ Coder_B) → Reviewer | Parallel redundancy effect |
| **Linear + Immune** | Planner → Immune → Coder → Reviewer | Mitigation via verification agent |

### Metrics

| Metric | Definition |
|--------|------------|
| **HCR** (Hallucination Contagion Rate) | % of downstream agents that mention the injected marker |
| **HAR** (Hallucination Amplification Rate) | % of infected agents that fabricate supporting documentation |
| **Contagion Depth** | Number of consecutive hops the marker survives |
| **Immune Efficacy** | Relative reduction in HCR with vs without immune agent |

## Prerequisites

### 1. Install Python 3.10+

```bash
python --version  # Verify ≥ 3.10
```

### 2. Install Ollama

Download from [ollama.com](https://ollama.com) and install for your platform.

```bash
# Verify installation
ollama --version
```

### 3. Pull the model

```bash
ollama pull llama3:8b-instruct
```

### 4. Start Ollama server

```bash
ollama serve
# Server runs on http://localhost:11434
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a small experiment (5 tasks, linear topology only)
python -m src.main --num_tasks 5 --topology linear

# Run full experiment (all topologies, 30 tasks)
python -m src.main --num_tasks 30

# Run all topologies with custom temperature
python -m src.main --num_tasks 10 --temperature 0.5

# Regenerate analysis from existing data
python -m src.main --analysis_only

# Skip GitHub fetch (use cached issues)
python -m src.main --skip_fetch --num_tasks 10
```

## CLI Reference

```
python -m src.main [OPTIONS]

Options:
  --config PATH          Path to config.json (default: config.json)
  --topology {linear,debate,linear_immune} [...]
                         Topologies to run (default: all three)
  --num_tasks N          Number of tasks to process
  --temperature FLOAT    Sampling temperature (default: 0.7)
  --seed INT             Random seed (default: 42)
  --model NAME           Ollama model name (default: llama3:8b-instruct)
  --skip_fetch           Use cached GitHub issues
  --force_fetch          Re-fetch issues even if cached
  --analysis_only        Only regenerate analysis, skip experiments
  --run_id ID            Reuse a specific run ID
  --output_dir DIR       Output directory (default: results)
```

## Output Structure

```
project_root/
├── config.json                  # Default configuration
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── injection_engine.py
│   ├── agent_runner.py
│   ├── topology_runner.py
│   ├── metrics_engine.py
│   ├── experiment_runner.py
│   ├── analysis.py
│   └── main.py
├── data/
│   ├── github_issues.json       # Cached real GitHub issues
│   ├── injected_issues.json     # Issues with injected markers
│   └── cache/
│       └── response_cache.json  # Cached Ollama responses
└── results/
    ├── raw_outputs.json         # All agent outputs with metadata
    ├── task_metrics.json        # Per-task metrics
    ├── aggregate_metrics.json   # Aggregated metrics by topology
    ├── immune_efficacy.json     # Immune agent efficacy stats
    ├── summary.csv              # CSV summary table
    ├── report.md                # Full Markdown research report
    ├── run_config.json          # Effective config for this run
    └── plots/
        ├── hcr_by_topology.png  # HCR bar chart
        ├── har_by_topology.png  # HAR bar chart
        └── depth_by_topology.png # Depth bar chart
```

## GitHub Rate Limits

The framework uses unauthenticated GitHub API calls (60 requests/hour). For higher limits:

```bash
export GITHUB_TOKEN=ghp_your_personal_access_token
```

The framework automatically detects rate limits and sleeps until reset.

## Reproducibility

Every run saves:
- `results/run_config.json` — exact parameters used
- Random seed (configurable via `--seed`)
- Response cache (all Ollama outputs cached to disk)
- Run ID embedded in the injected marker

To reproduce a previous run:
```bash
python -m src.main --run_id ABCD --seed 42 --skip_fetch
```

## Example Output

After running the full experiment, `results/report.md` will contain:
- Aggregate metrics table (HCR, HAR, depth per topology)
- Immune agent efficacy analysis
- Independent t-test results (Linear vs Immune)
- Key findings summary
- Embedded bar charts

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{hallucination_contagion_framework,
  title={Hallucination Contagion in Multi-Agent LLM Systems: A Reproducible Research Framework},
  year={2025},
  description={Empirical evaluation of hallucination propagation in multi-agent systems using controlled synthetic injection}
}
```

## License

MIT License
