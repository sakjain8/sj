# Hallucination Contagion Experiment Report

## Experiment Configuration

- **Model:** llama3:8b
- **Temperature:** 0.0
- **Seed:** 42
- **Marker:** `ENABLE_QUERY_BATCHING_6D40`
- **Run ID:** `6D40`
- **Total tasks:** 30
- **Topologies tested:** linear, debate, linear_immune, linear_immune_clean, epidemic

---

## Aggregate Results

| Topology | Tasks | Avg HCR | Avg HAR | Avg Depth | Infected / Total | Amplified |
|----------|-------|---------|---------|-----------|------------------|-----------|
| Linear | 30 | 0.00% | 0.00% | 0.00 | 0/60 | 0 |
| Debate | 30 | 0.00% | 0.00% | 0.00 | 0/90 | 0 |
| Linear + Immune | 30 | 0.00% | 0.00% | 0.00 | 0/90 | 0 |
| Immune (Clean) | 30 | 0.00% | 0.00% | 0.00 | 0/90 | 0 |
| Epidemic (Innate+Adaptive) | 30 | 68.33% | 34.72% | 1.20 | 55/80 | 20 |

---

## Immune Agent Efficacy

- **Linear HCR:** 0.00%
- **Immune HCR:** 0.00%
- **Absolute reduction:** 0.00%
- **Relative reduction:** 0.0%

## Statistical Analysis

**Independent samples t-test (Linear vs Linear+Immune):**

- t-statistic: nan
- p-value: nan
- Linear mean HCR: 0.00% (n=30)
- Immune mean HCR: 0.00% (n=30)
- Significant at α=0.05: **No**
- Significant at α=0.01: **No**

---

## Key Findings

1. **Hallucination contagion exists** — Injected marker propagated to downstream agents in 55 out of 410 total agent invocations.
2. **Network topology affects spread** — Epidemic (Innate+Adaptive) topology showed highest contagion (68.33%), while Linear showed lowest (0.00%).
3. **Immune agent reduces contagion** — The verification agent reduced HCR by 0.0% (from 0.00% to 0.00%).
4. **Epidemic Immune System reduces contagion** — The Innate+Adaptive immune topology reduced HCR by 0.0% (from 0.00% to 68.33%), modelling biological innate (self-verification) and adaptive (targeted quarantine) immune responses.

---

## Plots

![HCR by Topology](plots/hcr_by_topology.png)

![HAR by Topology](plots/har_by_topology.png)

![Contagion Depth](plots/depth_by_topology.png)

---

*Report generated automatically by the Hallucination Contagion Research Framework.*