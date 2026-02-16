from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random as _random
import torch
import torch.nn.functional as F


@dataclass
class Transition:
    agent_id: int
    obs: object
    global_state: torch.Tensor  # New: centralised state for critic
    idx: int
    op: int  # Now 0-3: internal_add, internal_del, cross_add, cross_del
    cands: List[Tuple[int, int]]
    internal_cands: List[Tuple[int, int]]  # For hierarchical action space
    cross_cands: List[Tuple[int, int]]  # For cross-partition actions
    logprob: float
    value: float
    reward: float
    done: bool


class MAPPOTrainer:
    """MAPPO with Centralized Critic (Set Transformer).

    - Actor: Decentralized execution (local obs)
    - Critic: Centralized training (global state)
    """

    def __init__(self, actor, critic, cfg: dict):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.buffer: List[Transition] = []
        tcfg = cfg.get('training', {}) or {}
        self.gamma = float(tcfg.get('gamma', 0.99))
        self.lmbda = float(tcfg.get('gae_lambda', 0.95))
        self.clip_eps = float(tcfg.get('clip', 0.15))
        self.lr = float(tcfg.get('lr', 3e-4))
        self.min_lr = float(tcfg.get('min_lr', 5e-5))
        self.epochs = int(tcfg.get('epochs', 8))
        self.minibatch_size = int(tcfg.get('minibatch_size', 128))
        self.max_grad_norm = 0.5
        
        # === 增强熵正则化 ===
        # 使用更高的熵系数，配合Actor中的条件熵计算
        lcfg = cfg.get('loss', {}) or {}
        self.entropy_coef = float(lcfg.get('entropy_coef', 0.02))  # 默认从0.01提升到0.02
        
        self.value_coef = float(lcfg.get('value_coef', 0.5))
        self.value_clip = float(lcfg.get('value_clip', 2.0))
        self.normalize_returns = bool(lcfg.get('normalize_returns', True))
        self.log_value_mean = bool(lcfg.get('log_value_mean', True))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )
        self.target_kl = float(tcfg.get('target_kl', 0.01))
        self.adapt_lr_on_kl = bool(tcfg.get('adapt_lr_on_kl', True))
        self.truncation_steps = int(tcfg.get('truncation_steps', 128))
        self._last_lr = self.lr
        
        # === 操作平衡统计 ===
        # 用于监控ADD/DEL操作的平衡性
        self.op_counter = {'add': 0, 'del': 0}

    def reset_rnn(self):
        if hasattr(self.actor, 'reset_rnn'):
            self.actor.reset_rnn()

    def store_transition(self, agent_id: int, obs, global_state, idx: int, op: int, cands: List[Tuple[int, int]],
                         logprob: float, value: float, reward: float, done: bool,
                         internal_cands: List[Tuple[int, int]] = None,
                         cross_cands: List[Tuple[int, int]] = None):
        if internal_cands is None:
            internal_cands = cands  # Fallback for backward compatibility
        if cross_cands is None:
            cross_cands = []
        
        # === 统计操作类型 ===
        # op: 0=internal_add, 1=internal_del, 2=cross_add, 3=cross_del
        if op in [0, 2]:  # ADD操作
            self.op_counter['add'] += 1
        else:  # DEL操作
            self.op_counter['del'] += 1
            
        self.buffer.append(Transition(agent_id, obs, global_state, idx, op, cands, internal_cands, cross_cands, logprob, value, reward, done))

    def _compute_gae(self):
        # Group transitions by agent, preserve temporal order
        by_agent: Dict[int, List[Transition]] = {}
        for tr in self.buffer:
            by_agent.setdefault(tr.agent_id, []).append(tr)

        advantages: List[float] = []
        returns: List[float] = []
        order: List[int] = []  # indices back into buffer

        for aid, traj in by_agent.items():
            # ensure time order is preserved (already appended in order)
            last_adv = 0.0
            # Bootstrap with last state value if not done, else 0
            next_value = float(traj[-1].value) if (len(traj) > 0 and not bool(traj[-1].done)) else 0.0
            for t in reversed(range(len(traj))):
                tr = traj[t]
                mask = 0.0 if tr.done else 1.0
                delta = tr.reward + self.gamma * next_value * mask - tr.value
                last_adv = delta + self.gamma * self.lmbda * mask * last_adv
                ret = last_adv + tr.value
                advantages.insert(0, last_adv)
                returns.insert(0, ret)
                order.insert(0, self.buffer.index(tr))
                next_value = tr.value

        # Normalize advantages
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        
        # === Winsorization: 截断极端值到 [-3σ, +3σ] ===
        # 数学原理: 将重尾分布转换为sub-Gaussian，确保梯度有界
        # 这消除了Advantage极端值对策略更新的过度影响
        adv_t = torch.clamp(adv_t, -3.0, 3.0)
        
        ret_t = torch.tensor(returns, dtype=torch.float32)
        return order, adv_t, ret_t

    def update(self):
        if not self.buffer:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._last_lr

        order, adv_t, ret_t = self._compute_gae()
        n_samples = len(order)
        # Stats for logging (pre-normalization)
        ret_mean = float(ret_t.mean().item())
        ret_std = float(ret_t.std(unbiased=False).item())

        # === PopArt: 更新统计量并归一化targets ===
        if hasattr(self.critic, 'update_stats'):
            self.critic.update_stats(ret_t)
            ret_hat = self.critic.normalize_targets(ret_t)
        elif self.normalize_returns:
            ret_hat = (ret_t - ret_t.mean()) / (ret_t.std(unbiased=False) + 1e-8)
        else:
            ret_hat = ret_t

        old_logps = torch.tensor([self.buffer[i].logprob for i in order], dtype=torch.float32)

        # PPO epochs with MINIBATCH support
        # Before: 1 gradient step per epoch (full-batch) → 4 updates total
        # After:  n_samples/minibatch_size steps per epoch → e.g. 20*8 = 160 updates
        observed_kl = 0.0
        lr_changed = False
        adaptive_clip = self.clip_eps
        kl_break = False
        mb = max(16, min(self.minibatch_size, n_samples))

        # Track last-epoch stats for logging (from last minibatch of last epoch)
        last_actor_loss = torch.tensor(0.0)
        last_critic_loss = torch.tensor(0.0)
        last_raw_delta = torch.zeros(1)

        for epoch in range(self.epochs):
            if kl_break:
                break

            # Shuffle indices each epoch for stochastic minibatch diversity
            perm = list(range(n_samples))
            _random.shuffle(perm)

            for mb_start in range(0, n_samples, mb):
                mb_end = min(mb_start + mb, n_samples)
                mb_indices = perm[mb_start:mb_end]

                new_logps = []
                values = []
                entropies = []

                for local_idx in mb_indices:
                    buf_idx = order[local_idx]
                    tr = self.buffer[buf_idx]
                    logp, _, entropy = self.actor.evaluate_logprob_and_value(
                        tr.obs, tr.idx, tr.op, tr.cands,
                        tr.internal_cands, tr.cross_cands)

                    if hasattr(self.critic, 'forward_normalized'):
                        val_pred = self.critic.forward_normalized(tr.global_state)
                    else:
                        val_pred = self.critic(tr.global_state)

                    new_logps.append(logp)
                    values.append(val_pred)
                    entropies.append(entropy)

                new_logps = torch.stack(new_logps).float()
                values = torch.stack(values).float()
                entropies = torch.stack(entropies).float()
                mb_old_logps = old_logps[mb_indices]
                mb_adv = adv_t[mb_indices]
                mb_ret_hat = ret_hat[mb_indices]

                # === 动态调整clip范围 ===
                if epoch > 0 and self.target_kl and observed_kl > 0:
                    adaptive_clip = self.clip_eps * min(1.0, self.target_kl / (observed_kl + 1e-8))
                    adaptive_clip = max(0.05, min(self.clip_eps, adaptive_clip))

                ratio = torch.exp(new_logps - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - adaptive_clip, 1.0 + adaptive_clip) * mb_adv

                entropy_loss = -self.entropy_coef * torch.mean(entropies)
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                actor_loss = policy_loss + entropy_loss

                # Critic loss (Huber-like)
                v_pred = values.squeeze(-1)
                raw_delta = v_pred - mb_ret_hat
                loss_unclipped = raw_delta * raw_delta

                if self.value_clip and self.value_clip > 0:
                    abs_delta = torch.abs(raw_delta)
                    loss_huber = torch.where(
                        abs_delta <= self.value_clip,
                        loss_unclipped,
                        self.value_clip * (2 * abs_delta - self.value_clip)
                    )
                    critic_loss = torch.mean(loss_huber)
                else:
                    critic_loss = torch.mean(loss_unclipped)

                loss = actor_loss + self.value_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()

                last_actor_loss = actor_loss.detach()
                last_critic_loss = critic_loss.detach()
                last_raw_delta = raw_delta.detach()

            # End-of-epoch KL check (evaluate on full buffer for stable estimate)
            with torch.no_grad():
                epoch_logps = []
                for local_idx in range(n_samples):
                    buf_idx = order[local_idx]
                    tr = self.buffer[buf_idx]
                    lp, _, _ = self.actor.evaluate_logprob_and_value(
                        tr.obs, tr.idx, tr.op, tr.cands,
                        tr.internal_cands, tr.cross_cands)
                    epoch_logps.append(lp)
                epoch_logps = torch.stack(epoch_logps).float()
                observed_kl = torch.mean(old_logps - epoch_logps).abs().item()

            if self.target_kl and observed_kl > 1.5 * self.target_kl:
                if self.adapt_lr_on_kl:
                    for g in self.optimizer.param_groups:
                        new_lr = max(float(self.min_lr), g['lr'] * 0.9)
                        if new_lr < g['lr']:
                            lr_changed = True
                        g['lr'] = new_lr
                        self._last_lr = g['lr']
                    if lr_changed:
                        try:
                            print(f"[KL EarlyStop] epoch={epoch} KL={observed_kl:.5f} > "
                                  f"{1.5*self.target_kl:.5f}; lr={self._last_lr:.2e}")
                        except Exception:
                            pass
                kl_break = True

        # Final KL estimate
        kl = observed_kl  # Already computed at end of last epoch

        # Adaptive LR: increase if KL is in healthy range
        if self.adapt_lr_on_kl and self.target_kl:
            if kl < 1.5 * self.target_kl and self._last_lr < self.lr:
                for g in self.optimizer.param_groups:
                    if kl < 0.3 * self.target_kl:
                        factor = 1.2
                    elif kl < self.target_kl:
                        factor = 1.1
                    else:
                        factor = 1.03
                    new_lr = min(self.lr, g['lr'] * factor)
                    if new_lr > g['lr']:
                        g['lr'] = new_lr
                        self._last_lr = g['lr']

        # Clear buffer and reset counters
        self.buffer.clear()

        total_ops = self.op_counter['add'] + self.op_counter['del']
        if total_ops > 0:
            pass  # add_ratio = self.op_counter['add'] / total_ops
        self.op_counter = {'add': 0, 'del': 0}

        # RMSE from last minibatch (representative)
        with torch.no_grad():
            mse_for_rmse = torch.mean(last_raw_delta * last_raw_delta)
        loss_v_rmse = float(torch.sqrt(mse_for_rmse).item())

        clip_fraction = 0.0
        if self.value_clip and self.value_clip > 0:
            with torch.no_grad():
                clip_fraction = float(
                    (torch.abs(last_raw_delta) >= self.value_clip - 1e-6).float().mean().item())
        return (float(last_actor_loss.item()), float(last_critic_loss.item()),
                float(kl), ret_mean, ret_std, loss_v_rmse, clip_fraction, self._last_lr)


class EarlyStopper:
    def __init__(self, metric_name: str, patience: int, min_delta: float):
        self.metric = metric_name
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0

    def step(self, metrics: Dict[str, float]) -> bool:
        val = float(metrics.get(self.metric, 0.0))
        if self.best is None or val > self.best + self.min_delta:
            self.best = val
            self.bad = 0
            return True
        self.bad += 1
        return False
