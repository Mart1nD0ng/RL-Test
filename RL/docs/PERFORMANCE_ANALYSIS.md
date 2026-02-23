# 每轮训练计算用时分析与优化方案

## 一、运行现象解读

| 阶段 | CPU 占用 | 可能原因 |
|------|----------|----------|
| 每轮开始时 | ~100% | 1) ep==0 时：`plot_topology` + `plot_logical_topology`（spring 布局 O(N²)）<br>2) `env.reset` 拓扑初始化（KNN）<br>3) `partitioner.init` 空间划分 |
| 每轮进行中 | ~50% 持续 | 主循环单线程，受 Python GIL 限制；部分 NumPy/SciPy 释放 GIL，但整体仍以单核为主 |

---

## 二、Profiling 实测数据

**测试条件**：3 episodes × 32 steps，100 节点，~10 agents

| 组件 | 耗时 (s) | 占比 |
|------|----------|------|
| **actor.act** | 4.75 | **42.3%** |
| **env.step** | 3.04 | **27.1%** |
| actor.embed_obs | 1.64 | 14.7% |
| apply_join_leave | 0.61 | 5.4% |
| build_observations | 0.58 | 5.1% |
| env_reset | 0.42 | 3.7% |
| tick_mobility | 0.07 | 0.6% |
| merger.merge | 0.07 | 0.6% |
| **合计** | **11.2** | 100% |

### cProfile 函数级热点

| 函数 | 累计耗时 | 调用次数 |
|------|----------|----------|
| actor.act | 7.76s | 384 |
| actor.compute_logits | 7.28s | 384 |
| env.step | 6.18s | 96 |
| actor._candidates | 4.25s | 384 |
| actor._gnn_forward | 3.70s | 768 |
| env._build_observations | 2.98s | 195 |
| env._build_step_link_cache | 2.43s | 96 |
| actor._build_edge_features | 2.26s | 768 |
| env._link_pl_db_3gpp | 1.76s | **38,418** |
| halo.build_for_partition | 1.58s | 780 |
| env.is_blocked | 0.95s | **33,138** |

---

## 三、瓶颈分析

### 1. actor.act（42.3%）——Agent 前向推理

**子项**：`_candidates`（4.25s）、`_gnn_forward`（3.70s）、`_build_edge_features`（2.26s）、`_score_candidates`

**原因**：
- 每步对每个 agent 单独调用 `actor.act(obs)`，顺序执行
- `_candidates` 中大量 `itertools.combinations`、`tuple(sorted())`、`math.sqrt`
- 每次 act 内部重复做 `_gnn_forward` + `_candidates`，无共享
- `embed_obs` 与 act 内部重复做 GNN，且 embed 也要做多次

**优化方向**：
- 将多 agent 的 obs 打包，做一次批量 `actor.act_batch()`
- 在 rollout 时只做一次 GNN，然后对每个 agent 复用边嵌入做打分，减少 GNN 调用
- 对 `_candidates` 使用 NumPy 批量计算距离，减少 Python 循环与 `tuple` 构造

---

### 2. env.step（27.1%）——环境转移

**子项**：`_build_step_link_cache`（2.43s）、`_link_pl_db_3gpp`（38k 次）、`is_blocked`（33k 次）、`_compute_marginal_contributions`

**原因**：
- 每条边单独调用 `_link_pl_db_3gpp` 与 `is_blocked`，Python 循环开销大
- `_link_pl_db_3gpp` 依赖距离、LOS、阴影，本身可向量化
- `is_blocked` 来自 mobility，每边都做射线检测

**优化方向**：
- 在 `_build_step_link_cache` 中，利用已有 `_physical_cache['dist_matrix']` 做批量路径损耗
- 用 NumPy 实现 `_link_pl_db_3gpp` 的批量版本（距离、LOS 概率、阴影）
- 对 `is_blocked`：缓存射线检测结果，或只在边集变化时重算

---

### 3. actor_embed_obs + build_observations（~20%）

**子项**：`_build_observations`（2.98s）、`halo.build_for_partition`（1.58s）、大量 `sorted()`（约 1.67M 次）

**原因**：
- 每个 agent 单独 `embed_obs`，内部都重新跑 GNN
- `_build_observations` 为每个 agent 构建 obs，调用 `halo.build_for_partition`
- 多处 `sorted()` 用于邻居、候选等，数量巨大

**优化方向**：
- 批量 `embed_obs`：一次 GNN 得到全图 embedding，再按 agent 切片
- 在分区未变时缓存 halo 结构，避免重复 `build_for_partition`
- 对不关心顺序的邻居集合用 `frozenset` 或预排序缓存替代现场 `sorted()`

---

### 4. apply_join_leave（5.4%）

**原因**：`env.apply_join_leave` 中含节点增删、边清理、`is_blocked` 等

**优化方向**：复用 env.step 中的物理缓存；对新边仅对新增部分做 `is_blocked`，其余复用缓存

---

### 5. 每轮初始 100% CPU 尖峰

**可能来源**：
- `plot_topology` + `plot_logical_topology`：特别是 `spring_layout`（networkx）在大图上很重
- `env.reset`：KNN 拓扑初始化、多轮距离计算

**优化方向**：
- 初始可视化改为异步或延后，或只在指定 episode 输出
- 降低初始图的绘制频率（如 `save_every_episodes` 调大）

---

## 四、优化方案与优先级

| 优先级 | 方案 | 预估收益 | 实现难度 |
|--------|------|----------|----------|
| P0 | 向量化 `_build_step_link_cache` 中的路径损耗 | env.step 减 30–50% | 中 |
| P0 | 批量 Actor 推理 `act_batch` | actor 总时减少约 40% | 高 |
| P1 | Halo 与 obs 缓存（分区未变时复用） | build_observations 减 30% | 中 |
| P1 | 减少/替代 `sorted()` 调用 | 全局减 5–10% | 低 |
| P2 | 延后或异步 ep==0 可视化 | 消除首轮 100% CPU 尖峰 | 低 |
| P2 | `_candidates` 用 NumPy 批量距离计算 | `_candidates` 减约 50% | 低 |

---

## 五、运行 Profiling 脚本

```bash
python scripts/profile_timing.py
```

将生成 `result_save/profile_cumulative.txt` 和 `PROFILING_REPORT.md`。
