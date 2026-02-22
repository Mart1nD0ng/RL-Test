"""Minimal profiling script for RL training loop.
Runs 2-3 episodes with 32 steps each, measuring time in each major component.
Uses time.perf_counter() for precise timing and cProfile for cumulative analysis.
"""
from __future__ import annotations

import os
import sys
import cProfile
import pstats
import io
import time

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.environment import CoreEnv
from core.merger_safety import MergerSafety, SafetyConfig
from core.messaging import MessageBus
from core.partitioner import Partitioner
from training.rollout import build_observations
from models.actor import Actor

# Minimal config matching train.py DEFAULT_CFG
DEFAULT_CFG = {
    'model': {'slots': 10, 'gnn_layers': 3, 'lstm_hidden': 128, 'msg_dim': 32, 'role_dim': 16},
    'safety': {'min_cross_edges_per_neighbor': 2, 'deltaE_budget_per_step': 20, 'cooldown_steps': 50},
    'partition': {'scheme': 'consistent_hash', 'rebalance': 'local_min_cost_flow',
                  'rho_high': 0.90, 'rho_low': 0.58, 'T_high': 120, 'T_low': 150},
    'messaging': {'period': 1, 'ttl': 3, 'dropout': 0.2, 'max_msgs_per_step': 1},
    'training': {'episodes': 3, 'T': 32, 'm_min': 4, 'm_max': 8},
    'sim': {'num_nodes': None},
}


def run_profiled_episodes(cfg: dict, num_episodes: int = 3, steps_per_episode: int = 32, verbose: bool = True):
    """Run episodes with perf_counter timing for each component."""
    tcfg = cfg.get('training', {}) or {}
    sim_cfg = cfg.get('sim') or {}
    N = int(sim_cfg.get('num_nodes') or int(cfg['model']['slots']) * 10)
    nodes = list(range(N))
    m = max(1, int(tcfg.get('m_min', 6)))

    env = CoreEnv(cfg)
    partitioner = Partitioner(cfg)
    safety_cfg = cfg.get('safety', {}) or {}
    merge_cfg = cfg.get('merge', {}) or {}
    merger = MergerSafety(SafetyConfig(
        min_cross_edges_per_neighbor=int(safety_cfg.get('min_cross_edges_per_neighbor', 2)),
        deltaE_budget_per_step=int(safety_cfg.get('deltaE_budget_per_step', 20)),
        cooldown_steps=int(safety_cfg.get('cooldown_steps', 50)),
        enforce_cross_edge_quota=bool(safety_cfg.get('enforce_cross_edge_quota', False)),
    ), cross_policy=str(merge_cfg.get('cross_policy', 'and')), score_thresh=float(merge_cfg.get('score_thresh', 0.2)))
    msg_cfg = cfg.get('messaging', {}) or {}
    bus = MessageBus(dim=int(cfg['model']['msg_dim']), ttl=int(msg_cfg.get('ttl', 3)), dropout=float(msg_cfg.get('dropout', 0.2)))
    actor = Actor(cfg)

    timers = {
        'env_reset': 0.0, 'tick_mobility': 0.0, 'apply_join_leave': 0.0,
        'build_observations': 0.0, 'actor_embed_obs': 0.0, 'actor_act': 0.0,
        'env_step': 0.0, 'merger_merge': 0.0, 'episode_total': [],
    }

    for ep in range(num_episodes):
        ep_start = time.perf_counter()
        partitions = partitioner.init(m, nodes, None)
        role_ids = partitioner.assign_roles(m)
        t0 = time.perf_counter()
        env.reset(partitions)
        timers['env_reset'] += time.perf_counter() - t0
        if env.node_pos:
            partitions = partitioner.init(m, nodes, env.node_pos)
            env.partitions = partitions
        bus.reset()
        merger.reset()
        if hasattr(actor, 'reset_rnn'):
            actor.reset_rnn()

        for t in range(steps_per_episode):
            if t > 0 and hasattr(actor, 'reset_rnn') and t % 128 == 0:
                actor.reset_rnn()
            t0 = time.perf_counter()
            env.tick_mobility()
            timers['tick_mobility'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            partitions, _ = env.apply_join_leave(partitioner, partitions)
            timers['apply_join_leave'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            obs_by_agent = build_observations(env, partitions, bus, role_ids, cfg)
            timers['build_observations'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for aid in sorted(obs_by_agent.keys()):
                actor.embed_obs(obs_by_agent[aid])
            timers['actor_embed_obs'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            actions = {}
            for aid, obs in obs_by_agent.items():
                act, _ = actor.act(obs)
                actions[aid] = act
            timers['actor_act'] += time.perf_counter() - t0
            part_map = env.partitions
            neighbors = {}
            for (u, v) in env.edges:
                a, b = part_map.get(u), part_map.get(v)
                if a is None or b is None or a == b:
                    continue
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
            neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
            actions_as_dict = {aid: {'msg_out': a.msg_out, 'internal_edges': [
                {'i': e.i, 'j': e.j, 'op': e.op, 'score': e.score} for e in a.internal_edges
            ]} for aid, a in actions.items()}
            bus.publish(actions_as_dict, neighbors)
            t0 = time.perf_counter()
            deltaE = merger.merge(actions, env.graph)
            timers['merger_merge'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            _, _, _ = env.step(deltaE, partitions)
            timers['env_step'] += time.perf_counter() - t0
            bus.tick()

        timers['episode_total'].append(time.perf_counter() - ep_start)

    total_time = sum(timers['episode_total'])
    if verbose:
        print("\n" + "=" * 60)
        print("PROFILING RESULTS (time.perf_counter)")
        print("=" * 60)
        print(f"Episodes: {num_episodes}, Steps per episode: {steps_per_episode}")
        print(f"Total wall time: {total_time:.4f}s")
        print("-" * 60)
        for k in ['env_reset','tick_mobility','apply_join_leave','build_observations','actor_embed_obs','actor_act','env_step','merger_merge']:
            pct = 100*timers[k]/total_time if total_time else 0
            print(f"{k:25} {timers[k]:.4f}s  ({pct:.1f}%)")
        print("=" * 60)
    return timers


def run_cprofile(cfg: dict, num_episodes: int = 3, steps_per_episode: int = 32, top_n: int = 35):
    profiler = cProfile.Profile()
    profiler.enable()
    tcfg = cfg.get('training', {}) or {}
    sim_cfg = cfg.get('sim') or {}
    N = int(sim_cfg.get('num_nodes') or int(cfg['model']['slots']) * 10)
    nodes = list(range(N))
    m = max(1, int(tcfg.get('m_min', 6)))
    env = CoreEnv(cfg)
    partitioner = Partitioner(cfg)
    safety_cfg = cfg.get('safety', {}) or {}
    merge_cfg = cfg.get('merge', {}) or {}
    merger = MergerSafety(SafetyConfig(
        min_cross_edges_per_neighbor=int(safety_cfg.get('min_cross_edges_per_neighbor', 2)),
        deltaE_budget_per_step=int(safety_cfg.get('deltaE_budget_per_step', 20)),
        cooldown_steps=int(safety_cfg.get('cooldown_steps', 50)),
        enforce_cross_edge_quota=bool(safety_cfg.get('enforce_cross_edge_quota', False)),
    ), cross_policy=str(merge_cfg.get('cross_policy', 'and')), score_thresh=float(merge_cfg.get('score_thresh', 0.2)))
    msg_cfg = cfg.get('messaging', {}) or {}
    bus = MessageBus(dim=int(cfg['model']['msg_dim']), ttl=int(msg_cfg.get('ttl', 3)), dropout=float(msg_cfg.get('dropout', 0.2)))
    actor = Actor(cfg)
    for ep in range(num_episodes):
        partitions = partitioner.init(m, nodes, None)
        role_ids = partitioner.assign_roles(m)
        env.reset(partitions)
        if env.node_pos:
            partitions = partitioner.init(m, nodes, env.node_pos)
            env.partitions = partitions
        bus.reset()
        merger.reset()
        if hasattr(actor, 'reset_rnn'):
            actor.reset_rnn()
        for t in range(steps_per_episode):
            if t > 0 and hasattr(actor, 'reset_rnn') and t % 128 == 0:
                actor.reset_rnn()
            env.tick_mobility()
            partitions, _ = env.apply_join_leave(partitioner, partitions)
            obs_by_agent = build_observations(env, partitions, bus, role_ids, cfg)
            for aid in sorted(obs_by_agent.keys()):
                actor.embed_obs(obs_by_agent[aid])
            actions = {}
            for aid, obs in obs_by_agent.items():
                act, _ = actor.act(obs)
                actions[aid] = act
            part_map = env.partitions
            neighbors = {}
            for (u, v) in env.edges:
                a, b = part_map.get(u), part_map.get(v)
                if a is None or b is None or a == b:
                    continue
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
            neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
            actions_as_dict = {aid: {'msg_out': a.msg_out, 'internal_edges': [
                {'i': e.i, 'j': e.j, 'op': e.op, 'score': e.score} for e in a.internal_edges
            ]} for aid, a in actions.items()}
            bus.publish(actions_as_dict, neighbors)
            deltaE = merger.merge(actions, env.graph)
            _, _, _ = env.step(deltaE, partitions)
            bus.tick()
    profiler.disable()
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(top_n)
    return s.getvalue()


def main():
    cfg = DEFAULT_CFG.copy()
    cfg['training'] = cfg.get('training', {}).copy()
    cfg['training']['episodes'] = 3
    cfg['training']['T'] = 32
    print("Running perf_counter profiling...")
    timers = run_profiled_episodes(cfg, num_episodes=3, steps_per_episode=32, verbose=True)
    print("\nRunning cProfile (cumulative time, top 35)...")
    cprofile_out = run_cprofile(cfg, num_episodes=3, steps_per_episode=32, top_n=35)
    print(cprofile_out)
    prof_path = os.path.join(ROOT, 'result_save', 'profile_cumulative.txt')
    os.makedirs(os.path.dirname(prof_path), exist_ok=True)
    with open(prof_path, 'w', encoding='utf-8') as f:
        f.write(cprofile_out)
    print(f"\ncProfile stats saved to: {prof_path}")


if __name__ == '__main__':
    main()
