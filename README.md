# Predictive Autoscaling & Shard Rebalancing for Vector Databases

A predictive, uncertainty-aware autoscaling and shard-rebalancing system built on top of a
self-hosted Qdrant vector database cluster. Instead of reacting to load after it happens
(like a standard threshold-based autoscaler), this system **forecasts** near-future load
per shard — with an uncertainty band, not just a point estimate — and pre-emptively scales
replicas and migrates shards before load actually arrives.

Combines ideas from predictive resource forecasting (MagicScaler-style), elastic vector-DB
infrastructure (Manu-style decoupled scaling), and production-trace-driven evaluation
(Alibaba / Azure public traces), plus a novel predictive shard-rebalancing policy that none
of those systems implement.

## Architecture

```
Workload generator (replays a real trace)
        |
Metrics collector (query rate, per-shard load, latency, vector count)
        |
Forecaster (predicts near-future load per shard, with uncertainty)
        |
Scaling / Rebalancing controller (decides: scale replicas? migrate shard?)
        |
Qdrant cluster (actuated via its API)
```

## Repo layout

- `infra/` — Qdrant cluster, Prometheus metrics, actuation API (**Person A / S**)
- `workload/` — trace parsing, replay engine, evaluation scenarios (**Person B / J**)
- `forecaster/` — EWMA baseline + quantile/Prophet forecaster, uncertainty output (**Person C / N**)
- `policy/` — static/reactive baselines, predictive scaling, predictive shard rebalancing,
  evaluation harness (**Person D**)
- `shared/` — shared data contracts (schemas) used by all modules
- `results/` — final charts, demo assets, architecture diagram

## Getting started

```bash
# 0. Install Python deps
pip install -r requirements.txt

# 1. Bring up the Qdrant cluster + Prometheus
docker-compose up -d

# 2. Parse a trace and generate a scenario
python workload/parsers/azure_trace_parser.py
python workload/scenarios/steady_state.py

# 3. Replay the workload against the live cluster
python workload/replay_engine.py --scenario steady_state

# 4. Run the baseline policies
python policy/baselines/static_provisioning.py
python policy/baselines/reactive_threshold.py

# 5. Run the predictive policy (forecaster + scaling + rebalancing)
python policy/predictive/predictive_scaling.py
python policy/predictive/predictive_rebalancing.py

# 6. Generate comparison charts
python policy/eval/generate_comparison_charts.py
```

## Related work / what we borrowed

- **Cost framing** (idle cost + scaling cost + SLA-violation cost) — MagicScaler
- **Uncertainty-aware forecasting rationale** — MagicScaler
- **Decoupled scaling components** so one bottleneck doesn't block others — Manu
- **Real-trace-driven evaluation methodology** — Autopilot, Manu (Taobao-style trace)
- **Predictive shard rebalancing** — novel to this project; none of the above implement it

We deliberately did **not** reimplement any of these papers wholesale. We borrowed the
framing/ideas and implemented much lighter-weight versions suited to a project timeline.

## Team

| Person | Module        | Responsibility                                             |
|--------|---------------|-------------------------------------------------------------|
| S      | `infra/`      | Qdrant cluster, metrics, actuation API                       |
| J      | `workload/`   | Trace parsing, replay engine, scenarios                      |
| N      | `forecaster/` | Uncertainty-aware load forecasting                           |
| D      | `policy/`     | Baselines, predictive scaling & rebalancing, evaluation       |

## Caveats (be upfront about these in any write-up)

- Uncertainty estimates and the cost model are approximations, not exact.
- The rebalancing policy's optimality is not formally proven — it's a heuristic.
- Results are from a small, self-hosted cluster; they won't perfectly predict behavior at
  much larger scale.
