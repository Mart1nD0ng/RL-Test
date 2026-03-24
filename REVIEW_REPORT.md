# RL Training Review Report

## Training Performance Summary (run_vis_v25)

- Episode 1-149 (Phase1): avg_return ≈ 4-5, success_proxy ≈ 0.95 (learning normally)
- Episode 150-399 (Phase2): avg_return ≈ 6.4 (peak performance)
- **Episode 400+ (Phase3): avg_return drops to -0.3~-0.5, success_proxy crashes from 0.95 to 0.03-0.07**
- Edge count drops from 400+ to ~99, topology effectively collapses

## Critical Issues

### P0: Phase3 density_penalty=2.0 causes reward signal inversion
**File**: `configs/config.yaml:192-194`

The density_penalty weight jumps from 0.5 to 2.0 at episode 400. For a 100-node network with 400 edges:
- penalty = 2.0 × (400/150 - 1)² = 5.58
- max consensus reward = 2.0 × 1.0³ = 2.0

The penalty far exceeds possible rewards, making edge deletion the only rational policy.

### P0: Truncation incorrectly marked as terminal — GAE bootstrap is zero
**File**: `train.py:421`, `mappo_trainer.py:109`

`done = (t == steps_T - 1)` treats truncation as terminal, causing GAE to bootstrap with 0 instead of V(s_T). This makes the policy myopic.

### P0: LSTM state inconsistency between act() and evaluate()
**File**: `actor.py:364-368`

During rollout, LSTM accumulates temporal context (`mutate_state=True`). During PPO re-evaluation, LSTM uses zero-initialized state (`mutate_state=False`). This distorts the importance sampling ratio.

### P1: Single Critic value shared across agents with per-agent rewards
**File**: `train.py:355-359`

One global V(s) is used for all agents, but rewards differ per agent (Shapley-based). This creates high variance in advantage estimation.

### P1: Candidate list mismatch between act() and evaluate()
**File**: `actor.py:728,776`

Candidates are regenerated during evaluate, potentially producing different lists. Index is silently clipped with `min(idx, len(eval_cands)-1)`.

### P1: Per-transition forward pass (no batching)
**File**: `mappo_trainer.py:191-205`

Each transition is processed individually, resulting in ~20K GNN+LSTM forward passes per update.

### P1: PopArt beta too small for phase transitions
**File**: `config.yaml:207`

β=0.0003 needs ~3333 batches to track distribution shift, far too slow for abrupt phase transitions.

## Validation Issues

### P2: test_once() never applies actions
**File**: `validation.py:84-86`

Creates empty DeltaE, so agent decisions are never executed during validation.

### P2: run_m_sweep() is a stub
**File**: `validation.py:33`

Always returns success=1.0 (placeholder).

### P2: Actor/Critic share optimizer and learning rate
**File**: `mappo_trainer.py:59-62`

KL gate LR reduction also slows Critic learning.

### P3: O(N²) GAE computation via buffer.index()
**File**: `mappo_trainer.py:118`

### P3: Redundant legacy network heads consuming gradients
**File**: `actor.py:168-177`
