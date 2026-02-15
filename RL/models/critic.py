from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class Critic(nn.Module):
    """Set Transformer Critic for CTDE.

    Input: Set of agent embeddings [Batch, m, F] (m varies, handled by masking/padding or pure set op)
    Output: Global Value V(s)

    The Set Transformer (ICML 2019) is permutation invariant and handles variable set sizes naturally.
    We use SAB (Self-Attention Block) for encoding and PMA (Pooling by Multihead Attention) for aggregation.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, output_dim: int = 1,
                 num_heads: int = 4, num_inds: int = 16):
        super(Critic, self).__init__()
        self.enc = nn.Sequential(
            ISAB(input_dim, hidden_dim, num_heads, num_inds, ln=True),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=True)
        )
        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, 1, ln=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [batch, n_agents, feature_dim]
        # We assume X is already padded if batch > 1, but for typical rollout batch=1 it's just [1, m, F]
        # If m varies inside a batch, we'd need a mask, but here we process episode-steps which usually have uniform m per batch or batch=1.
        h = self.enc(X)
        out = self.dec(h).squeeze(-1).squeeze(-1) # [batch]
        return out


class PopArtCritic(nn.Module):
    """带有PopArt归一化的Critic包装器
    
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets)
    
    === 数学原理 ===
    将Critic的输出分解为: V(s) = σ * Ṽ(s) + μ
    其中:
    - μ, σ 是return分布的running mean和std
    - Ṽ(s) 是归一化后的预测
    
    当return分布变化时，只更新 μ, σ，同时调整最后一层权重以保持输出连续性。
    这解决了非平稳目标函数导致的Critic学习不稳定问题。
    
    === EMA更新策略 ===
    μ_new = (1-β) * μ_old + β * batch_mean
    σ²_new = (1-β) * σ²_old + β * batch_var  (直接对方差使用EMA，更稳定)
    
    参考: DeepMind "Multi-task RL with PopArt" (2018)
    """
    
    def __init__(self, base_critic: Critic, beta: float = 0.0003):
        super(PopArtCritic, self).__init__()
        self.critic = base_critic
        self.beta = beta  # EMA decay rate (小值=慢响应=稳定)
        
        # PopArt统计量 (使用buffer确保保存和加载时正确处理)
        self.register_buffer('mu', torch.tensor(0.0))           # Running mean E[G]
        self.register_buffer('sigma', torch.tensor(1.0))        # Running std sqrt(Var[G])
        self.register_buffer('var', torch.tensor(1.0))          # Running variance Var[G] (直接跟踪，避免数值问题)
        self.register_buffer('_count', torch.tensor(0))         # 样本计数 (用于预热期)
        self.register_buffer('_initialized', torch.tensor(False))
        
        # 预热期：前N个batch使用更大的beta快速收敛
        self.warmup_batches = 10
        self.warmup_beta = 0.1  # 预热期使用更大的更新率
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播，返回反归一化后的实际值预测（用于推理/rollout）"""
        v_normalized = self.critic(X)
        # V(s) = σ * Ṽ(s) + μ
        v_actual = self.sigma * v_normalized + self.mu
        return v_actual
    
    def forward_normalized(self, X: torch.Tensor) -> torch.Tensor:
        """前向传播，返回归一化的预测值（用于训练时与归一化目标对比）
        
        === 关键修复 ===
        训练时Critic应该输出归一化的值 Ṽ(s)，与归一化后的return目标 Ĝ = (G - μ) / σ 对比
        这样损失函数的量纲才一致，梯度才能正确传播
        """
        return self.critic(X)
    
    def update_stats(self, returns: torch.Tensor):
        """使用EMA更新统计量，并调整输出层权重
        
        === 数值稳定的EMA更新 ===
        直接对μ和σ²使用EMA，而非计算二阶矩后减法（避免 ν-μ²<0 的问题）
        
        μ_new = (1-β) * μ_old + β * batch_mean
        σ²_new = (1-β) * σ²_old + β * batch_var
        σ_new = sqrt(σ²_new)
        """
        with torch.no_grad():
            # 计算batch统计量
            batch_mean = returns.mean()
            batch_var = returns.var(unbiased=False)  # 使用有偏估计，与EMA一致
            batch_std = torch.sqrt(batch_var.clamp(min=1e-8))
            
            # 更新样本计数
            self._count.add_(1)
            count = self._count.item()
            
            # 首次初始化：直接使用batch统计量
            if not self._initialized.item():
                self.mu.copy_(batch_mean)
                self.var.copy_(batch_var.clamp(min=1e-8))
                self.sigma.copy_(batch_std.clamp(min=1e-4))
                self._initialized.fill_(True)
                return
            
            # 保存旧的统计量
            old_mu = self.mu.clone()
            old_sigma = self.sigma.clone()
            
            # === EMA更新 ===
            # 预热期使用更大的beta，加速初始收敛
            if count <= self.warmup_batches:
                effective_beta = self.warmup_beta
            else:
                effective_beta = self.beta
            
            # μ_new = (1-β) * μ_old + β * batch_mean
            new_mu = (1.0 - effective_beta) * self.mu + effective_beta * batch_mean
            
            # σ²_new = (1-β) * σ²_old + β * batch_var (直接对方差EMA，数值稳定)
            new_var = (1.0 - effective_beta) * self.var + effective_beta * batch_var
            new_var = new_var.clamp(min=1e-8)  # 确保非负
            
            # σ_new = sqrt(σ²_new)
            new_sigma = torch.sqrt(new_var).clamp(min=1e-4)
            
            # === 调整最后一层权重以保持输出连续性 ===
            # 数学推导见类docstring
            output_layer = self.critic.dec[-1]
            if isinstance(output_layer, nn.Linear):
                # 防止除零
                scale_factor = old_sigma / new_sigma.clamp(min=1e-6)
                
                # W_new = W_old * σ_old / σ_new
                output_layer.weight.data.mul_(scale_factor)
                
                # b_new = (σ_old * b_old + μ_old - μ_new) / σ_new
                output_layer.bias.data = (old_sigma * output_layer.bias.data + old_mu - new_mu) / new_sigma.clamp(min=1e-6)
            
            # 更新统计量buffer
            self.mu.copy_(new_mu)
            self.var.copy_(new_var)
            self.sigma.copy_(new_sigma)
    
    def normalize_targets(self, returns: torch.Tensor) -> torch.Tensor:
        """将return归一化为训练目标: G_normalized = (G - μ) / σ"""
        return (returns - self.mu) / self.sigma.clamp(min=1e-6)
    
    def get_stats(self) -> dict:
        """返回当前统计量，用于日志记录"""
        return {
            'popart_mu': float(self.mu.item()),
            'popart_sigma': float(self.sigma.item()),
            'popart_var': float(self.var.item()),
            'popart_count': int(self._count.item()),
        }

