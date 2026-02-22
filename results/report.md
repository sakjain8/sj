# Hallucination Contagion Experiment Report

## Experiment Configuration

- **Model:** llama3:8b
- **Temperature:** 0.0
- **Seed:** 42
- **Marker:** `ENABLE_QUERY_BATCHING_EA04`
- **Run ID:** `EA04`
- **Total tasks:** 30
- **Topologies tested:** linear, debate, linear_immune, linear_immune_clean, epidemic

---

## Aggregate Results

| Topology | Tasks | Avg HCR | Avg HAR | Avg Depth | Infected / Total | Amplified |
|----------|-------|---------|---------|-----------|------------------|-----------|
| Linear | 30 | 86.67% | 53.33% | 1.57 | 52/60 | 31 |
| Debate | 30 | 80.00% | 57.22% | 1.87 | 72/90 | 45 |
| Linear + Immune | 30 | 75.56% | 53.89% | 2.00 | 68/90 | 36 |
| Immune (Clean) | 30 | 82.22% | 49.44% | 2.23 | 74/90 | 35 |
| Epidemic (Innate+Adaptive) | 30 | 70.83% | 41.39% | 1.27 | 52/72 | 28 |

---

## Immune Agent Efficacy

- **Linear HCR:** 86.67%
- **Immune HCR:** 75.56%
- **Absolute reduction:** 11.11%
- **Relative reduction:** 12.8%

## Statistical Analysis

**Independent samples t-test (Linear vs Linear+Immune):**

- t-statistic: 1.563
- p-value: 0.123559
- Linear mean HCR: 86.67% (n=30)
- Immune mean HCR: 75.56% (n=30)
- Significant at α=0.05: **No**
- Significant at α=0.01: **No**

---

## Key Findings

1. **Hallucination contagion exists** — Injected marker propagated to downstream agents in 318 out of 402 total agent invocations.
2. **Network topology affects spread** — Linear topology showed highest contagion (86.67%), while Epidemic (Innate+Adaptive) showed lowest (70.83%).
3. **Immune agent reduces contagion** — The verification agent reduced HCR by 12.8% (from 86.67% to 75.56%).
4. **Epidemic Immune System reduces contagion** — The Innate+Adaptive immune topology reduced HCR by 18.3% (from 86.67% to 70.83%), modelling biological innate (self-verification) and adaptive (targeted quarantine) immune responses.

---

## Plots

![HCR by Topology](plots/hcr_by_topology.png)

![HAR by Topology](plots/har_by_topology.png)

![Contagion Depth](plots/depth_by_topology.png)

---

*Report generated automatically by the Hallucination Contagion Research Framework.*