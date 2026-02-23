# Optuna 超参数优化 + 全栈固定随机种子 — 设计方案

## 一、run_vis_v25 结果摘要

| 指标 | 早期 (ep 1–50) | 中期 (ep 150–300) | 后期 (ep 450–528) |
|------|----------------|------------------|-------------------|
| success_proxy | 0.6–0.9 | 0.7–0.9 | 0.5–0.8（波动大） |
| edges | 149–339 | 350–450 | 350–500 |
| avg_return | 0.6–3.8 | 1.5–3.5 | **-5 ~ -2** |
| energy_proxy | 0.07–0.14 | 0.14–0.17 | 0.18–0.25 |
| delay_proxy | 0.017–0.03 | 0.02 | 0.02 |
| KL | 0.003–0.1 | 0.05–0.1 | 0.00–0.1 |

课程阶段：Phase1 (ep 1–149) → Phase2 (ep 150–399) → Phase3 (ep 400+)。

主要现象：
- Phase3 后 avg_return 变为负，与 ret_mean/ret_std 增长有关，但不代表策略必然变差。
- success_proxy 在 0.5–0.9 之间波动，说明策略尚未稳定。
- edges 后期偏高，说明 density_penalty 可能仍偏弱，或需更长时间收敛。

---

## 二、基于 Optuna 的奖励权重优化

### 2.1 目标与搜索空间

**优化目标（建议）：**
- 主目标：最大化 `success_proxy` 在最后 N 个 episode 的均值或中位数。
- 可选多目标：`success_proxy - α * energy_proxy - β * delay_proxy`（α、β 可设为 0.1、0.05）。

**搜索空间（7 维 reward_weights）：**

| 参数 | 当前 Phase1 | 建议搜索范围 | 类型 |
|------|-------------|--------------|------|
| consensus | 5.0 | [2.0, 8.0] | float |
| robustness | 0.5 | [0.1, 1.0] | float |
| energy | 0.1 | [0.05, 0.5] | float |
| delay | 0.3 | [0.1, 0.8] | float |
| density_penalty | 0.1 | [0.0, 1.0] | float |
| long_edge | 0.2 | [0.1, 0.5] | float |
| edit | 0.05 | [0.01, 0.2] | float |

可采用 `suggest_float` 或 `suggest_categorical`（离散档位）以减少搜索空间。

### 2.2 压缩课程与单次 Trial 设置

为降低单 trial 时间，建议：

| 项目 | 原配置 | 压缩后（Tune 模式） |
|------|--------|---------------------|
| episodes / trial | 800 | **60–120** |
| Phase1 结束 | ep 150 | ep **20** |
| Phase2 结束 | ep 400 | ep **50** |
| Phase3 | ep 400+ | ep **51–80** |
| max_steps_per_episode | 256 | **128** 或 256 |
| sim.num_nodes | 100 | **60** 或 80 |

实现方式：
- 在 `config.yaml` 或 Optuna 专用配置中定义 `course_schedule_tune`（压缩版）。
- Optuna 脚本载入该配置，覆盖 `training.episodes`、`training.course_schedule`。
- 保持 physics/INTERF_K_LOCAL 三阶段递增逻辑不变，仅压缩 episode 边界。

### 2.3 脚本与流程设计

**新建脚本**：`scripts/optuna_reward_tune.py`

```
流程:
1. 解析 CLI：--n-trials, --episodes-per-trial, --study-name 等
2. 创建 Optuna Study（sampler=TPESampler, n_startup_trials=5）
3. objective(trial):
   a. suggest 7 个 reward_weights
   b. 固定随机种子（见第三节）
   c. 构建 cfg override：reward_weights + course_schedule_tune
   d. 调用 run() 或等效训练循环（需支持传入 cfg 和 seed）
   e. 读取 metrics.csv 或 run 返回值
   f. 返回 target = mean(success_proxy[-20:]) 或自定义组合
4. study.optimize(objective, n_trials=N)
5. 输出 best params、保存到 result_save/optuna_best_reward_weights.yaml
```

**run() 接口扩展**：
- 增加 `seed: int | None`、`cfg_override: dict | None`（包含 reward_weights、course_schedule）。
- 若传入 seed，在 run() 开头调用统一的 seeding 函数（见第三节）。

### 2.4 与现有课程逻辑的耦合

- `_apply_course_stage` 会更新 `cfg` 中的 `reward_weights`，并调用 `env.set_reward_weights()`。
- Optuna 只需在训练开始前把 trial 的 reward_weights 写入 `cfg['reward_weights']` 或 `course_schedule` 的第一阶段 `reward_weights`，后续阶段可按比例缩放（例如 Phase2 = 0.8 * Phase1，Phase3 = 0.6 * Phase1），或完全由 trial 搜索各阶段的权重组合（空间更大，但计算更贵）。

---

## 三、全栈严格固定随机种子

### 3.1 当前随机源分布

| 模块 | 随机源 | 当前行为 |
|------|--------|----------|
| train.py | 无 | 无 seed |
| mappo_trainer | random.shuffle | 无 seed |
| core/environment | random, random.Random | 部分用 hash/time_step 做 seed |
| core/environment | mobility, LOS/NLOS | random.Random(seed), random.gauss |
| training/rollout | random | maybe_quantize_noise, maybe_inject_birth_death |
| core/messaging | random | dropout |
| core/partitioner | random | 局部 random.seed(42) |
| core/visualize | random.Random | 固定 0, 789 等 |
| training/validation | random, np, torch | test_once(seed=1337) |
| scripts/evaluate | 无 | 无 seed |

### 3.2 统一 Seeding 函数

**新建**：`core/seeding.py`（或放在 utils 下）

```python
def set_global_seed(seed: int) -> None:
    """在进程启动时调用一次，固定所有随机源。"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 3.3 各模块需改动的点

| 文件 | 改动 |
|------|------|
| **train.py** | 启动时若 `cfg['training'].get('seed')` 存在，调用 `set_global_seed(seed)`；`run()` 接受 `seed` 参数 |
| **mappo_trainer** | 使用 `random` 前由调用方保证已 seed；或接受 `rng` 参数（可选） |
| **core/environment** | `CoreEnv.__init__` 接受 `master_seed: int | None`；内部用 `random.Random(master_seed + offset)` 替代裸 `random` |
| **core/environment** | LOS/NLOS、阴影等保持 `hash(key, time_step)` 确定性，但可改为 `hash((key, time_step, master_seed))` 若传入 seed |
| **CityGridMobility** | 初始化/步进用 `self.rng = random.Random(master_seed + ...)` |
| **training/rollout** | `maybe_quantize_noise` 等使用 `random`，由全局 seed 统一控制 |
| **core/messaging** | 同上 |
| **core/partitioner** | 移除局部 `random.seed(42)`，改由全局 seed 或传入 `rng` |
| **profile_timing.py** | 若传入 seed，启动时调用 `set_global_seed` |
| **scripts/evaluate.py** | 支持 `--seed`，调用 `set_global_seed` |
| **optuna_reward_tune.py** | 每个 trial 使用 `trial.number` 或 `seed_base + trial.number` 作为 seed |

### 3.4 配置项设计

在 `config.yaml` 的 `training` 下增加：

```yaml
training:
  seed: null   # 设为整数则全栈固定；null 则保持当前非确定性行为
```

CLI 扩展：`--seed 42` 覆盖 `training.seed`。

---

## 四、实现优先级与依赖

| 阶段 | 内容 | 依赖 |
|------|------|------|
| **P0** | `core/seeding.py` + `set_global_seed()` | 无 |
| **P1** | train.py 支持 `--seed` 并在启动时调用 | P0 |
| **P2** | environment / partitioner / messaging 等改为可复现 | P0 |
| **P3** | 压缩课程配置 `course_schedule_tune` | 无 |
| **P4** | `scripts/optuna_reward_tune.py` | P1, P2, P3 |
| **P5** | requirements 或 文档说明添加 `optuna` | 无 |

---

## 五、预期效果与注意事项

- **Optuna**：在 20–50 trials 内得到优于当前手调权重的组合；单 trial 约 10–30 分钟（视压缩程度）。
- **Seeding**：相同 seed 下，训练曲线应可复现；注意 `torch.backends.cudnn.deterministic=True` 可能略微影响性能。
- **课程压缩**：可能导致收敛质量略降，但便于快速筛选权重；最终可再用完整课程对 best 权重做长跑验证。
