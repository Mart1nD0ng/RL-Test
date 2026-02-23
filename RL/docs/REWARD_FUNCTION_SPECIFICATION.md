# 奖励函数规格说明：计算方法、数学公式与物理意义

## 一、总体结构

全局奖励（global reward）由以下加权项组成：

$$
R = \left( w_{\text{consensus}} \cdot r_{\text{consensus}}^{\text{convex}} 
       + w_{\text{robustness}} \cdot \lambda_2 
       - w_{\text{energy}} \cdot E_{\text{norm}} 
       - w_{\text{delay}} \cdot D_{\text{norm}} 
       - w_{\text{long\_edge}} \cdot L 
       - w_{\text{edit}} \cdot C_{\text{edit}} \right) \cdot \mathbb{1}_{\text{conn}}
       - w_{\text{density}} \cdot P_{\text{edge}}
       - 0.5 \cdot B_{\text{bridge}}
       + \text{bonus}_{\text{near\_perfect}}
$$

其中 $\mathbb{1}_{\text{conn}}$ 为连通性乘子：若图连通则为 1，否则为 0.1。

各智能体的 Difference Reward 在此基础上叠加边际贡献：

$$
R_a = R + \text{ShapleyWeight} \cdot \Delta_a
$$

---

## 二、各项计算方法与物理意义

### 1. Consensus Success（共识成功率）$r_{\text{consensus}}$

**物理意义**：衡量 PBFT 类共识协议的可行性——网络能否支撑至少 $2f+1$ 节点参与，且路径可靠性足够高。

**计算流程**：
1. 构建**可靠性加权邻接矩阵** $A_{\text{rel}}$，边权 $w_{uv} = -\ln p_{uv}$（$p_{uv} > 0.01$ 的边才计入）
2. 求**最大连通分量 (LCC)** 的节点集 $\mathcal{V}_{\text{LCC}}$
3. 计算 **Quorum 比例**：
   $$f = \lfloor (N-1)/3 \rfloor,\quad q_{\min} = 2f + 1,\quad q_{\text{ratio}} = \frac{|\mathcal{V}_{\text{LCC}}|}{q_{\min}}$$
4. **Quorum 因子**（Sigmoid）：
   $$\phi_q = \frac{1}{1 + \exp(-k_q (q_{\text{ratio}} - 1))},\quad k_q = 8$$
5. **LCC 覆盖比例**：$\rho_{\text{LCC}} = |\mathcal{V}_{\text{LCC}}| / N$
6. **路径可靠性**：在 LCC 内随机采样 5 个源、30 对 (源, 目标)，用 Dijkstra 求最短“可靠性距离” $d_{ij}$（边权为 $-\ln p$），定义：
   $$r_{\text{path}}^{(ij)} = e^{-d_{ij}}$$
   取 30 次采样的平均 $\overline{r}_{\text{path}}$
7. **组合得分**：
   $$r_{\text{consensus}} = \phi_q \cdot \left( 0.7 \cdot \overline{r}_{\text{path}} + 0.3 \cdot \rho_{\text{LCC}} \right)$$

**凸化处理**（用于奖励）：
$$r_{\text{consensus}}^{\text{convex}} = r_{\text{consensus}}^3$$

立方放大使高共识区间（如 0.8→0.9）的边际奖励远大于低区间。

---

### 2. Robustness（代数连通度 / Fiedler 值）$\lambda_2$

**物理意义**：图的 Laplacian 第二小特征值，刻画网络抗割能力。$\lambda_2 = 0$ 表示不连通；越大表示冗余路径越多、越难被割裂。

**公式**：
$$L_{ij} = \begin{cases}
-\max(\varepsilon, p_{uv}) & (i,j) \in E \\
\sum_{k} \max(\varepsilon, p_{ik}) & i = j \\
0 & \text{otherwise}
\end{cases}$$

$p_{uv}$ 为链路成功概率（HARQ 后的端到端可靠性）。

$$\lambda_2 = \text{second\_smallest\_eigenvalue}(L)$$

**归一化**（与节点数无关的稳定尺度）：
$$\lambda_2^{\text{norm}} = \frac{\lambda_2}{\max(1, N)}$$

---

### 3. Energy Cost（归一化能耗）$E_{\text{norm}}$

**物理意义**：全网边总能耗的归一化值。低 SINR 链路需更多 HARQ 重传，单边能耗更高。

**单边能耗**：
$$E_{(u,v)} = P_{\text{TX}} \cdot T_{\text{slot}} \cdot \mathbb{E}[N_{\text{tx}}(u,v)]$$

其中 $\mathbb{E}[N_{\text{tx}}]$ 为 HARQ 下的期望传输次数：

$$\mathbb{E}[N_{\text{tx}}] = \frac{1 - p_{\text{err}}^{M+1}}{1 - p_{\text{err}}}$$

$p_{\text{err}} = \text{BLER}(\gamma)$，$M$ 为最大重传次数。

**BLER 模型**（Sigmoid 逼近 AMC 曲线）：
$$\text{BLER} = \frac{1}{1 + \exp(k (\gamma_{\text{dB}} - \gamma_{\text{thresh}}))}$$

$k = 1.5$，$\gamma_{\text{thresh}} = -5\,\text{dB}$（QPSK 1/8 鲁棒调制）。

**总能耗与归一化**：
$$E_{\text{total}} = \sum_{(u,v) \in E} E_{(u,v)},\quad E_{\text{norm}} = \min\left(1,\, \frac{E_{\text{total}}}{E_{\max}}\right)$$

$E_{\max}$ 为配置的归一化天花（如 1.0 J）。

---

### 4. Latency Cost（归一化时延）$D_{\text{norm}}$

**物理意义**：端到端路径期望延迟的统计量，反映信息传播速度。

**单边延迟**：
$$D_{(u,v)} = \mathbb{E}[N_{\text{tx}}] \cdot T_{\text{slot}}$$

与能耗共用 $\mathbb{E}[N_{\text{tx}}]$，低 SINR 链路延迟更大。

**时延加权图**：边权为 $D_{(u,v)}$，Dijkstra 求最小时延路径。

**采样估计**：随机 5 源、30 对 (源, 目标)，求平均路径时延 $\overline{D}$。

**归一化**：
$$D_{\text{norm}} = \min\left(0.8,\, \frac{\overline{D}}{6 \cdot D_{\max}}\right)$$

$D_{\max}$ 为物理最大时延（如 50 ms）。

---

### 5. Edge Density Penalty（边密度惩罚）$P_{\text{edge}}$

**物理意义**： penalize 边数过多，鼓励稀疏拓扑以降低干扰与能耗。

**公式**：
$$e_{\text{target}} = \max(1,\, 1.5 N),\quad \rho_e = \frac{|E|}{e_{\text{target}}}$$

$$e_{\text{excess}} = \max(0,\, \rho_e - 1)$$

$$P_{\text{edge}} = e_{\text{excess}}^2$$

二次惩罚使超出目标后的边际成本迅速上升。

---

### 6. Long-Edge Cost（长边惩罚）$L$

**物理意义**： penalize 平均边长远大于参考距离 $D_0$ 的拓扑，鼓励短链路。

**公式**：
$$\bar{d}_{\text{norm}} = \frac{1}{|E|_{\text{valid}}} \sum_{(u,v)} \frac{d_{uv}}{D_0}$$

$$L = \text{clip}_{[0,1]} \left( \frac{\bar{d}_{\text{norm}} - 0.5}{1.0} \right)$$

$D_0 = 200\,\text{m}$，$d_{uv}$ 为欧氏距离。

---

### 7. Edit Cost（编辑成本）$C_{\text{edit}}$

**物理意义**： penalize 单步内拓扑变化幅度过大，鼓励平滑策略。

**公式**：
$$C_{\text{edit}} = \min\left(1,\, \frac{|\Delta E_{\text{add}}| + |\Delta E_{\text{delete}}|}{20}\right)$$

---

### 8. Bridge-Cut Penalty（桥边删除惩罚）$B_{\text{bridge}}$

**物理意义**： penalize 删除**桥边**（无共同邻居的边），减少割裂连通分量。

**计算**：对每个计划删除的边 $(u,v)$，若 $\text{CN}(u,v) = 0$ 则 $B_{\text{bridge}} \mathrel{+}= 1$。

**最终惩罚**：$-0.5 \cdot B_{\text{bridge}}$（固定系数）。

---

### 9. Connectivity Multiplier（连通性乘子）$\mathbb{1}_{\text{conn}}$

**物理意义**：图不连通时，主奖励项乘以 0.1，强烈抑制断开行为。

$$\mathbb{1}_{\text{conn}} = \begin{cases} 1 & \text{connected} \\ 0.1 & \text{disconnected} \end{cases}$$

---

### 10. Near-Perfect Bonus（高共识加成）

**物理意义**：对接近完美共识的额外正向激励。

$$
\text{bonus} = \begin{cases}
2.0 & r_{\text{consensus}} > 0.95 \\
0.5 & r_{\text{consensus}} > 0.85 \\
0 & \text{otherwise}
\end{cases}
$$

该加成不受连通性乘子影响。

---

### 11. Difference Rewards（边际贡献）$\Delta_a$

**物理意义**：Shapley 值近似，衡量智能体 $a$ 对当前 LCC 规模的边际贡献。

**公式**：
$$\Delta_a = \frac{\max(0,\, |\text{LCC}| - |\text{LCC}_{-a}|)}{N}$$

$\text{LCC}_{-a}$ 为移除智能体 $a$ 所管理的节点后，剩余图的最大连通分量大小。

**智能体奖励**：
$$R_a = R + \text{ShapleyWeight} \cdot \Delta_a,\quad \text{ShapleyWeight} = 1.0$$

---

## 三、底层物理模型依赖

| 模块 | 作用 |
|------|------|
| **3GPP TR 38.901 UMi** | 路径损耗、LOS/NLOS 概率、阴影衰落 |
| **INTERF_K_LOCAL** | 接收端度数驱动的局部干扰：$N_{\text{eff}}(v) = N_{\text{therm}} \cdot (1 + \lambda \cdot \text{deg}(v))$ |
| **HARQ** | 重传模型，决定 $\mathbb{E}[N_{\text{tx}}]$ 和端到端可靠性 |
| **BLER Sigmoid** | AMC 曲线近似，参数 $\gamma_{\text{thresh}}=-5\,\text{dB}$，$k=1.5$ |

---

## 四、默认权重与课程阶段

| 参数 | Phase1 | Phase2 | Phase3 | 含义 |
|------|--------|--------|---------|------|
| consensus | 5.0 | 3.0 | 2.0 | 共识主驱动力 |
| robustness | 0.5 | 0.3 | 0.2 | 抗割能力 |
| energy | 0.1 | 0.5 | 0.5 | 能耗惩罚 |
| delay | 0.3 | 0.5 | 0.5 | 时延惩罚 |
| density_penalty | 0.1 | 0.5 | 2.0 | 边数惩罚 |
| long_edge | 0.2 | 0.3 | 0.5 | 长边惩罚 |
| edit | 0.05 | 0.1 | 0.1 | 编辑惩罚 |

课程阶段通过 `course_schedule` 在训练过程中逐步切换上述权重与物理参数（如 INTERF_K_LOCAL）。
