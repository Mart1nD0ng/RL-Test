# RL Profiling Report

## 1. Script
- Path: `scripts/profile_timing.py`
- Run: `python scripts/profile_timing.py --cfg configs/config.yaml [--steps N] [--cprofile]`

## 2. Config
- Nodes: 100, Agents: 10, Steps: 256

## 3. Episode Timing
| Phase | Time (s) | % |
|-------|---------|---|
| Episode total | 347.438 | 100% |
| Rollout | 33.836 | 9.7% |
| trainer.update | 313.602 | 90.3% |

## 4. Rollout Component Breakdown

| Component | Cumul(s) | % | Mean(ms) | Min(ms) | Max(ms) |
|-----------|----------|---|----------|---------|---------|
| tick_mobility | 0.1067 | 0.3% | 0.42 | 0.24 | 0.64 |
| apply_join_leave | 0.1589 | 0.5% | 0.62 | 0.36 | 0.79 |
| build_observations | 1.6780 | 5.0% | 6.55 | 3.69 | 145.65 |
| embed_obs | 9.0852 | 26.9% | 35.49 | 29.12 | 52.50 |
| compute_global_context | 0.2311 | 0.7% | 0.90 | 0.56 | 1.64 |
| critic_forward | 0.5905 | 1.7% | 2.31 | 1.66 | 3.19 |
| actor_act | 16.1411 | 47.7% | 63.05 | 47.27 | 69.12 |
| build_neighbors | 0.0253 | 0.1% | 0.10 | 0.05 | 0.13 |
| quantize_noise | 0.0814 | 0.2% | 0.32 | 0.18 | 0.52 |
| bus_publish | 0.0080 | 0.0% | 0.03 | 0.01 | 0.05 |
| merger_merge | 0.1416 | 0.4% | 0.55 | 0.18 | 1.29 |
| env_step | 5.5412 | 16.4% | 21.65 | 13.75 | 140.58 |
| bus_tick | 0.0120 | 0.0% | 0.05 | 0.01 | 0.07 |
| autoscaler_step | 0.0022 | 0.0% | 0.01 | 0.01 | 0.02 |
| store_transition | 0.0054 | 0.0% | 0.02 | 0.01 | 0.03 |
| maybe_local_rebalance | 0.0007 | 0.0% | 0.00 | 0.00 | 0.01 |

## 5. Top Bottlenecks

1. **actor_act** (~48%): Per-agent sequential forward; consider batch inference.
2. **embed_obs** (~27%): Per-agent GNN; consider shared embedding cache.
3. **env_step** (~16%): _build_step_link_cache, _link_pl_db_3gpp; consider vectorization.

## 6. cProfile Function-Level Detail

When run with `--cprofile`, see `profile_episode_report.txt` or `profile_cumulative.txt`.
