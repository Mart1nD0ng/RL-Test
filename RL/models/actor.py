from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data_structures import Observation, Action, EdgeEdit


# ---------------------------------------------------------------------------
# GatedGCN Layer — edge-centric message passing with residual connections
# ---------------------------------------------------------------------------

class GatedGCNLayer(nn.Module):
    """Edge-Centric Gated GCN with residual connections.

    Simultaneously updates **node** features *h* and **edge** features *e*:

    1. Edge update (residual):
       ê = ReLU(LN( W_src·h_u + W_dst·h_v + W_e·e_uv ))
       e' = e + ê

    2. Gate (anisotropic filter):
       σ_uv = Sigmoid( W_g · e'_uv )          ∈ ℝ^{D_h}

    3. Gated node aggregation (residual):
       h'_v = h_v + ReLU(LN( Σ_{u∈N(v)} σ_uv ⊙ W_m·h_u ))
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.W_src = nn.Linear(node_dim, edge_dim, bias=False)
        self.W_dst = nn.Linear(node_dim, edge_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, edge_dim, bias=False)
        self.ln_e = nn.LayerNorm(edge_dim)

        self.W_gate = nn.Linear(edge_dim, node_dim)

        self.W_msg = nn.Linear(node_dim, node_dim, bias=False)
        self.ln_h = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:    [N, D_h]       node features
            e:    [N, N, D_e]    edge features
            adj:  [N, N]         adjacency (float, with self-loops)
            mask: [N]            1.0 = valid, 0.0 = padded
        Returns:
            h_new [N, D_h],  e_new [N, N, D_e]
        """
        # 2-D validity mask for edges between valid nodes
        if mask is not None:
            m2d = mask.unsqueeze(0) * mask.unsqueeze(1)          # [N, N]
        else:
            m2d = None

        edge_mask = adj.unsqueeze(-1)                            # [N, N, 1]

        # --- edge update ---
        src = self.W_src(h)                                      # [N, D_e]
        dst = self.W_dst(h)                                      # [N, D_e]
        e_in = src.unsqueeze(1) + dst.unsqueeze(0) + self.W_e(e) # [N, N, D_e]
        e_hat = F.relu(self.ln_e(e_in))
        e_new = e + e_hat
        e_new = e_new * edge_mask
        if m2d is not None:
            e_new = e_new * m2d.unsqueeze(-1)

        # --- gate ---
        gate = torch.sigmoid(self.W_gate(e_new))                 # [N, N, D_h]
        gate = gate * edge_mask
        if m2d is not None:
            gate = gate * m2d.unsqueeze(-1)

        # --- gated aggregation ---
        # For dest v:  agg_v = Σ_u  gate[u,v,:] ⊙ W_msg(h_u)
        msg = self.W_msg(h)                                      # [N, D_h]
        # broadcast:  msg[u] → [N_src, 1, D_h] × gate [N_src, N_dst, D_h]
        gated_msg = gate * msg.unsqueeze(1)                      # [N, N, D_h]
        agg = gated_msg.sum(dim=0)                               # [N, D_h]

        h_new = h + F.relu(self.ln_h(agg))
        if mask is not None:
            h_new = h_new * mask.unsqueeze(-1)

        return h_new, e_new


# ---------------------------------------------------------------------------
# Actor — Edge-Centric Gated-GCN  +  LSTM Context Broadcasting
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Edge-Centric Gated-GCN Actor with LSTM Context Broadcasting.

    Pipeline
    --------
    1. **Dynamic edge features** from positions / velocities / link quality.
    2. **GatedGCN** (2 layers): simultaneous node + edge feature update with
       learned gates that suppress noisy / low-quality links.
    3. **Masked mean pooling** → concat halo/time/msg/role → **LSTM** for
       global temporal context.
    4. **Edge-centric scoring**: GNN edge embeddings ⊕ LSTM context →
       per-candidate MLP scores (Pointer-Network style).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get('model', {}) or {}
        self.slots = int(model_cfg.get('slots', 10))
        self.msg_dim = int(model_cfg.get('msg_dim', 32))
        self.lstm_hidden = int(model_cfg.get('lstm_hidden', 128))

        # ---- dimensions ----
        self.node_fdim = 10   # mask battery snr loss rtt role dpos(2) dvel(2)
        self.edge_fdim = 6    # Δx Δy Δvx Δvy dist snr_avg

        node_hidden = 64
        edge_hidden = 32
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden

        # ---- feature projections ----
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_fdim, node_hidden), nn.ReLU())
        self.edge_proj = nn.Sequential(
            nn.Linear(self.edge_fdim, edge_hidden), nn.ReLU())

        # ---- GatedGCN stack ----
        self.gnn_layers = nn.ModuleList([
            GatedGCNLayer(node_hidden, edge_hidden),
            GatedGCNLayer(node_hidden, edge_hidden),
        ])

        # ---- global aggregation → LSTM ----
        halo_dim = 8 + 3 + 1 + 1 + 4 * 4       # 29
        time_dim = 2
        role_dim = 1
        ctx_in = node_hidden + halo_dim + time_dim + self.msg_dim + role_dim

        self.context_proj = nn.Sequential(
            nn.Linear(ctx_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.lstm = nn.LSTM(128, self.lstm_hidden, batch_first=True)

        # ---- edge-centric scoring head ----
        score_in = edge_hidden + self.lstm_hidden
        self.edge_score = nn.Sequential(
            nn.Linear(score_in, 64), nn.ReLU(), nn.Linear(64, 1))
        self.raw_edge_proj = nn.Sequential(
            nn.Linear(self.edge_fdim, edge_hidden), nn.ReLU())

        # ---- policy heads ----
        self.max_internal_candidates = 4
        self.max_cross_candidates = 4
        self.max_candidates = self.max_internal_candidates + self.max_cross_candidates

        self.head_op_type = nn.Linear(self.lstm_hidden, 4)
        self.head_edge_idx = nn.Linear(
            self.lstm_hidden,
            max(self.max_internal_candidates, self.max_cross_candidates))

        # legacy heads kept for callers that unpack the old return tuple
        self.head_internal = nn.Linear(self.lstm_hidden, self.max_internal_candidates)
        self.head_cross = nn.Linear(self.lstm_hidden, self.max_cross_candidates)
        self.head_op = nn.Linear(self.lstm_hidden, 2)
        self.head_edge = self.head_internal

        self.head_msg = nn.Linear(self.lstm_hidden, self.msg_dim)
        self.head_value = nn.Linear(self.lstm_hidden, 1)

        # ---- per-agent LSTM state ----
        self._lstm_state: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # conditional-entropy coefficients
        self.entropy_coef_op = 1.0
        self.entropy_coef_edge = 1.5

    # ------------------------------------------------------------------
    # RNN helpers
    # ------------------------------------------------------------------

    def reset_rnn(self, **_kwargs):
        """Clear all per-agent recurrent states."""
        self._lstm_state.clear()

    def _init_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        h0 = torch.zeros(1, 1, self.lstm_hidden, device=device)
        c0 = torch.zeros(1, 1, self.lstm_hidden, device=device)
        return h0, c0

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _node_features(self, obs: Observation) -> torch.Tensor:
        """[N, node_fdim] from observation slots."""
        feats: List[List[float]] = []
        for s in obs.slots[:self.slots]:
            feats.append([
                float(s.mask), float(s.battery),
                float(s.link_snr), float(s.link_loss), float(s.link_rtt),
                float(s.role_bit),
                float(s.delta_pos[0]), float(s.delta_pos[1]),
                float(s.delta_vel[0]), float(s.delta_vel[1]),
            ])
        while len(feats) < self.slots:
            feats.append([0.0] * self.node_fdim)
        return torch.tensor(feats, dtype=torch.float32)

    def _slot_mask(self, obs: Observation) -> torch.Tensor:
        masks: List[float] = []
        for s in obs.slots[:self.slots]:
            masks.append(float(s.mask))
        while len(masks) < self.slots:
            masks.append(0.0)
        return torch.tensor(masks, dtype=torch.float32)

    def _adjacency_from_obs(self, obs: Observation) -> torch.Tensor:
        id_to_idx = {s.node_id: i
                     for i, s in enumerate(obs.slots[:self.slots]) if s.mask > 0}
        N = self.slots
        A = torch.zeros((N, N), dtype=torch.float32)
        for (i, j, w) in obs.adj_local:
            if i in id_to_idx and j in id_to_idx:
                u, v = id_to_idx[i], id_to_idx[j]
                A[u, v] = max(A[u, v].item(), float(w))
                A[v, u] = max(A[v, u].item(), float(w))
        A += torch.eye(N, dtype=torch.float32)
        return A

    def _build_edge_features(self, obs: Observation) -> torch.Tensor:
        """[N, N, edge_fdim] — dynamic pairwise features.

        Per pair (i, j): [Δx, Δy, Δvx, Δvy, ‖Δp‖, snr_avg]
        """
        N = self.slots
        slots = obs.slots[:N]

        pos = torch.zeros(N, 2)
        vel = torch.zeros(N, 2)
        snr = torch.zeros(N)
        msk = torch.zeros(N)

        for i, s in enumerate(slots):
            if s.mask > 0:
                pos[i, 0] = float(s.delta_pos[0])
                pos[i, 1] = float(s.delta_pos[1])
                vel[i, 0] = float(s.delta_vel[0])
                vel[i, 1] = float(s.delta_vel[1])
                snr[i] = float(s.link_snr)
                msk[i] = 1.0

        dx  = pos[:, 0].unsqueeze(1) - pos[:, 0].unsqueeze(0)
        dy  = pos[:, 1].unsqueeze(1) - pos[:, 1].unsqueeze(0)
        dvx = vel[:, 0].unsqueeze(1) - vel[:, 0].unsqueeze(0)
        dvy = vel[:, 1].unsqueeze(1) - vel[:, 1].unsqueeze(0)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        snr_avg = (snr.unsqueeze(1) + snr.unsqueeze(0)) * 0.5

        e = torch.stack([dx, dy, dvx, dvy, dist, snr_avg], dim=-1)
        m2d = msk.unsqueeze(0) * msk.unsqueeze(1)
        return e * m2d.unsqueeze(-1)

    def _halo_vector(self, obs: Observation) -> torch.Tensor:
        hs = obs.halo_summary
        vec: List[float] = []
        vec.extend([float(x) for x in hs.deg_hist])
        vec.extend([float(x) for x in hs.snr_stats])
        vec.append(float(hs.min_degree))
        vec.append(float(hs.cross_candidates))
        for a in hs.topk_anchor:
            vec.extend([float(a[0]), float(a[1]), float(a[2]), float(a[3])])
        return torch.tensor(vec, dtype=torch.float32)

    def _id_to_slot_idx(self, obs: Observation) -> Dict[int, int]:
        return {s.node_id: i
                for i, s in enumerate(obs.slots[:self.slots]) if s.mask > 0}

    # ------------------------------------------------------------------
    # GNN + LSTM forward
    # ------------------------------------------------------------------

    def _gnn_forward(self, obs: Observation):
        """Run GatedGCN stack.

        Returns (h, e, mask) — all on model device.
        """
        device = next(self.parameters()).device
        x     = self._node_features(obs).to(device)
        adj   = self._adjacency_from_obs(obs).to(device)
        mask  = self._slot_mask(obs).to(device)
        e_raw = self._build_edge_features(obs).to(device)

        h = self.node_proj(x)
        e = self.edge_proj(e_raw)

        for layer in self.gnn_layers:
            h, e = layer(h, e, adj, mask)

        return h, e, mask

    def embed_obs(self, obs: Observation) -> torch.Tensor:
        """Return a fixed-size embedding [128] for one observation.

        Used by the training loop to build the Critic's global state
        tensor.  Replaces the old ``_post_embed`` method.
        """
        h, _e, mask = self._gnn_forward(obs)
        device = h.device
        n_valid = mask.sum().clamp(min=1.0)
        graph_embed = (h * mask.unsqueeze(-1)).sum(dim=0) / n_valid

        halo_v = self._halo_vector(obs).to(device)
        time_v = torch.tensor(
            [float(obs.time_feat[0]), float(obs.time_feat[1])],
            dtype=torch.float32, device=device)
        msg_raw = [float(v) for v in obs.msg_in]
        while len(msg_raw) < self.msg_dim:
            msg_raw.append(0.0)
        msg_in = torch.tensor(msg_raw[:self.msg_dim],
                              dtype=torch.float32, device=device)
        role_t = torch.tensor([float(obs.role_id)],
                              dtype=torch.float32, device=device)

        ctx = torch.cat([graph_embed, halo_v, time_v, msg_in, role_t], dim=0)
        return self.context_proj(ctx)                            # [128]

    def _lstm_forward(self, h: torch.Tensor, mask: torch.Tensor,
                      obs: Observation, *, mutate_state: bool) -> torch.Tensor:
        """Masked mean pool → context concat → LSTM → h_lstm."""
        device = h.device
        n_valid = mask.sum().clamp(min=1.0)
        graph_embed = (h * mask.unsqueeze(-1)).sum(dim=0) / n_valid

        halo_v = self._halo_vector(obs).to(device)
        time_v = torch.tensor(
            [float(obs.time_feat[0]), float(obs.time_feat[1])],
            dtype=torch.float32, device=device)
        msg_raw = [float(v) for v in obs.msg_in]
        while len(msg_raw) < self.msg_dim:
            msg_raw.append(0.0)
        msg_in = torch.tensor(msg_raw[:self.msg_dim],
                              dtype=torch.float32, device=device)
        role_t = torch.tensor([float(obs.role_id)],
                              dtype=torch.float32, device=device)

        ctx = torch.cat([graph_embed, halo_v, time_v, msg_in, role_t], dim=0)
        ctx = self.context_proj(ctx).view(1, 1, -1)

        role_id = int(obs.role_id)
        if mutate_state:
            st = self._lstm_state.get(role_id, self._init_state())
            y, st2 = self.lstm(ctx, st)
            self._lstm_state[role_id] = st2
        else:
            y, _ = self.lstm(ctx, self._init_state())

        return y[:, -1, :].squeeze(0)                            # [lstm_hidden]

    # ------------------------------------------------------------------
    # Edge-centric scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: List[Tuple[int, int]],
        e_gnn: torch.Tensor,
        h_lstm: torch.Tensor,
        obs: Observation,
        id_to_idx: Dict[int, int],
    ) -> torch.Tensor:
        """Score each candidate edge via GNN embedding ⊕ LSTM context → MLP.

        If both endpoints live in the slot graph the GNN edge embedding is
        used directly.  Otherwise raw geometric features are projected to
        the same dimensionality so the same scoring MLP can be reused.
        """
        if not candidates:
            return torch.zeros(0, device=h_lstm.device)

        device = h_lstm.device
        node_pos: Dict[int, Tuple[float, float]] = (
            getattr(obs, 'node_positions', None) or {})

        embeds: List[torch.Tensor] = []
        for u, v in candidates:
            iu = id_to_idx.get(u)
            iv = id_to_idx.get(v)

            if iu is not None and iv is not None:
                embeds.append(e_gnn[iu, iv])
            else:
                pu = node_pos.get(u, (0.0, 0.0))
                pv = node_pos.get(v, (0.0, 0.0))
                ddx = pu[0] - pv[0]
                ddy = pu[1] - pv[1]
                d = math.sqrt(ddx * ddx + ddy * ddy + 1e-8)

                vu = vv = (0.0, 0.0)
                su = sv = 0.0
                for s in obs.slots[:self.slots]:
                    if s.mask > 0:
                        if s.node_id == u:
                            vu, su = s.delta_vel, s.link_snr
                        if s.node_id == v:
                            vv, sv = s.delta_vel, s.link_snr

                raw = torch.tensor(
                    [ddx, ddy,
                     vu[0] - vv[0], vu[1] - vv[1],
                     d,
                     (su + sv) * 0.5],
                    dtype=torch.float32, device=device)
                embeds.append(self.raw_edge_proj(raw))

        stacked = torch.stack(embeds)                            # [C, edge_hidden]
        ctx = h_lstm.unsqueeze(0).expand(stacked.size(0), -1)   # [C, lstm_hidden]
        logits = self.edge_score(
            torch.cat([stacked, ctx], dim=-1)).squeeze(-1)       # [C]
        return logits

    # ------------------------------------------------------------------
    # Candidate generation  (unchanged logic)
    # ------------------------------------------------------------------

    def _candidates(self, obs: Observation) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]],
        List[Tuple[int, int]], List[Tuple[int, int]],
    ]:
        """Hierarchical candidate edges for ADD / DEL × internal / cross.

        Returns (internal_add, cross_add, internal_del, cross_del).
        """
        node_pos: Dict[int, Tuple[float, float]] = (
            getattr(obs, 'node_positions', None) or {})
        if not node_pos:
            for s in obs.slots:
                if s.mask > 0:
                    node_pos[s.node_id] = s.delta_pos

        internal_nodes = getattr(obs, 'internal_nodes', None)
        boundary_nodes = getattr(obs, 'boundary_nodes', None)
        halo_nodes     = getattr(obs, 'halo_nodes', None)
        existing_internal = set(
            tuple(sorted(e))
            for e in getattr(obs, 'existing_internal_edges', []))
        existing_cross = set(
            tuple(sorted(e))
            for e in getattr(obs, 'existing_cross_edges', []))

        if internal_nodes is None:
            valid = [s.node_id for s in obs.slots if s.mask > 0]
            pairs = [(int(u), int(v))
                     for u, v in itertools.combinations(valid[:6], 2)]
            return (pairs[:self.max_internal_candidates], [],
                    pairs[:4], [])

        internal_set = set(internal_nodes)

        # --- ADD candidates (prefer short) ---
        ia: List[Tuple[float, int, int]] = []
        for u, v in itertools.combinations(internal_nodes, 2):
            if tuple(sorted((u, v))) in existing_internal:
                continue
            if u in node_pos and v in node_pos:
                dx = node_pos[u][0] - node_pos[v][0]
                dy = node_pos[u][1] - node_pos[v][1]
                ia.append((math.sqrt(dx * dx + dy * dy), int(u), int(v)))
        ia.sort(key=lambda t: t[0])
        internal_add = [(u, v) for _, u, v in ia[:self.max_internal_candidates]]

        ca: List[Tuple[float, int, int]] = []
        for u in (boundary_nodes or []):
            for v in (halo_nodes or []):
                if tuple(sorted((u, v))) in existing_cross:
                    continue
                if u in node_pos and v in node_pos:
                    dx = node_pos[u][0] - node_pos[v][0]
                    dy = node_pos[u][1] - node_pos[v][1]
                    d = math.sqrt(dx * dx + dy * dy)
                else:
                    d = 0.5
                ca.append((d, int(u), int(v)))
        ca.sort(key=lambda t: t[0])
        cross_add = [(u, v) for _, u, v in ca[:self.max_cross_candidates]]

        # --- DEL candidates (longest first) ---
        id_: List[Tuple[float, int, int]] = []
        for u, v in existing_internal:
            if u in node_pos and v in node_pos:
                dx = node_pos[u][0] - node_pos[v][0]
                dy = node_pos[u][1] - node_pos[v][1]
                d = math.sqrt(dx * dx + dy * dy)
            else:
                d = 0.3
            id_.append((d, int(u), int(v)))
        id_.sort(key=lambda t: -t[0])
        internal_del = [(u, v) for _, u, v in id_[:self.max_internal_candidates]]

        cd: List[Tuple[float, int, int]] = []
        for u, v in existing_cross:
            u_in = u in internal_set
            ln = u if u_in else v
            rn = v if u_in else u
            if ln in node_pos and rn in node_pos:
                dx = node_pos[ln][0] - node_pos[rn][0]
                dy = node_pos[ln][1] - node_pos[rn][1]
                d = math.sqrt(dx * dx + dy * dy)
            else:
                d = 0.5
            cd.append((d, int(u), int(v)))
        cd.sort(key=lambda t: -t[0])
        cross_del = [(u, v) for _, u, v in cd[:self.max_cross_candidates]]

        return internal_add, cross_add, internal_del, cross_del

    def _legacy_candidates(self, obs: Observation) -> List[Tuple[int, int]]:
        ia, ca, _, _ = self._candidates(obs)
        return (ia + ca)[:self.max_candidates]

    # ------------------------------------------------------------------
    # Main interface: compute_logits
    # ------------------------------------------------------------------

    def compute_logits(self, obs: Observation):
        """Forward pass → all logits needed by act / evaluate.

        Returns the **same 7-tuple** as the previous architecture so that
        every downstream caller keeps working:

            (logits_internal, logits_cross, logits_op,
             msg_out, value, cands, hier_info)
        """
        h, e, mask = self._gnn_forward(obs)
        id_to_idx = self._id_to_slot_idx(obs)
        h_lstm = self._lstm_forward(h, mask, obs, mutate_state=True)

        logits_op_type = self.head_op_type(h_lstm)

        internal_add, cross_add, internal_del, cross_del = self._candidates(obs)

        scores_ia = self._score_candidates(internal_add, e, h_lstm, obs, id_to_idx)
        scores_ca = self._score_candidates(cross_add, e, h_lstm, obs, id_to_idx)
        scores_id = self._score_candidates(internal_del, e, h_lstm, obs, id_to_idx)
        scores_cd = self._score_candidates(cross_del, e, h_lstm, obs, id_to_idx)

        msg_out = torch.tanh(self.head_msg(h_lstm))
        value = self.head_value(h_lstm).squeeze(-1)

        # legacy heads (kept for backward-compat callers)
        logits_edge_idx = self.head_edge_idx(h_lstm)
        logits_internal = self.head_internal(h_lstm)
        logits_cross = self.head_cross(h_lstm)
        logits_op = self.head_op(h_lstm)
        cands = internal_add + cross_add

        hier_info = {
            'internal_add': internal_add,
            'cross_add': cross_add,
            'internal_del': internal_del,
            'cross_del': cross_del,
            'logits_op_type': logits_op_type,
            'logits_edge_idx': logits_edge_idx,
            'scores_internal_add': scores_ia,
            'scores_cross_add': scores_ca,
            'scores_internal_del': scores_id,
            'scores_cross_del': scores_cd,
        }

        return (logits_internal, logits_cross, logits_op,
                msg_out, value, cands, hier_info)

    # ------------------------------------------------------------------
    # act  —  sample one action (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: Observation):
        """Sample an action using edge-centric GNN scoring.

        Action Space (hierarchical):
          op_type  : 4 choices  [int_add, int_del, cross_add, cross_del]
          edge_idx : variable   (scored by GNN edge embedding + LSTM ctx)
        """
        (logits_internal, logits_cross, logits_op,
         msg_out, value, cands, hier_info) = self.compute_logits(obs)

        logits_op_type = hier_info['logits_op_type']
        device = logits_op_type.device

        op_map = {
            0: ('add', 'internal', hier_info['internal_add'],
                hier_info['scores_internal_add']),
            1: ('del', 'internal', hier_info['internal_del'],
                hier_info['scores_internal_del']),
            2: ('add', 'cross', hier_info['cross_add'],
                hier_info['scores_cross_add']),
            3: ('del', 'cross', hier_info['cross_del'],
                hier_info['scores_cross_del']),
        }

        # ---- mask unavailable ops ----
        masked_op = logits_op_type.clone()
        for oi, (_, _, cl, _) in op_map.items():
            if len(cl) == 0:
                masked_op[oi] = float('-inf')

        if getattr(obs, 'edge_budget_exceeded', False):
            masked_op[0] = float('-inf')
            masked_op[2] = float('-inf')

        n_add = len(hier_info['internal_add']) + len(hier_info['cross_add'])
        n_del = len(hier_info['internal_del']) + len(hier_info['cross_del'])
        if n_del > 0 and n_add > 0:
            ratio = n_add / max(1, n_del)
            if ratio > 2.0:
                boost = min(1.0, math.log(ratio))
                masked_op[1] = masked_op[1] + boost
                masked_op[3] = masked_op[3] + boost

        probs_op = torch.softmax(masked_op, dim=-1)
        dist_op = torch.distributions.Categorical(probs_op)
        op_idx = dist_op.sample()
        logp_op = dist_op.log_prob(op_idx)
        ent_op = dist_op.entropy()

        op_str, etype, sel_cands, edge_scores = op_map[op_idx.item()]
        is_internal = (etype == 'internal')

        # ---- edge selection via GNN scores ----
        max_c = (self.max_internal_candidates
                 if is_internal else self.max_cross_candidates)

        if len(sel_cands) == 0:
            sel_cands = [(-1, -1)]
            edge_logits = torch.zeros(1, device=device)
        else:
            edge_logits = edge_scores[:len(sel_cands)]

        if edge_logits.numel() < max_c:
            pad = torch.full((max_c - edge_logits.numel(),),
                             float('-inf'), device=device)
            edge_logits = torch.cat([edge_logits, pad])
        edge_logits = edge_logits[:max_c]

        probs_e = torch.softmax(edge_logits, dim=-1)
        dist_e = torch.distributions.Categorical(probs_e)
        e_idx = dist_e.sample()
        logp_e = dist_e.log_prob(e_idx)
        ent_e = dist_e.entropy()

        logprob = (logp_op + logp_e).detach()
        entropy = (self.entropy_coef_op * ent_op
                   + self.entropy_coef_edge * ent_e).detach()

        # ---- build Action ----
        internal_edges: List[EdgeEdit] = []
        cross_edges: List[EdgeEdit] = []

        if e_idx.item() < len(sel_cands):
            u, v = sel_cands[e_idx.item()]
            if u >= 0 and v >= 0 and u != v:
                sc = float(probs_e[e_idx].item())
                ee = EdgeEdit(i=u, j=v, op=op_str, score=sc)
                if is_internal:
                    internal_edges.append(ee)
                else:
                    cross_edges.append(ee)

        action = Action(
            internal_edges=internal_edges,
            cross_edge_scores=cross_edges,
            msg_out=msg_out.tolist())

        aux = {
            'idx': int(e_idx.item()),
            'op': int(op_idx.item()),
            'cands': cands,
            'internal_cands': sel_cands if is_internal else [],
            'cross_cands': sel_cands if not is_internal else [],
            'logprob': float(logprob.item()),
            'value': float(value.item()),
            'entropy': float(entropy.item()),
            'entropy_op': float(ent_op.item()),
            'entropy_edge': float(ent_e.item()),
        }
        return action, aux

    # ------------------------------------------------------------------
    # evaluate_logprob_and_value  —  PPO re-evaluation (with grad)
    # ------------------------------------------------------------------

    def evaluate_logprob_and_value(
        self,
        obs: Observation,
        idx: int,
        op: int,
        cands: List[Tuple[int, int]],
        internal_cands: Optional[List[Tuple[int, int]]] = None,
        cross_cands: Optional[List[Tuple[int, int]]] = None,
    ):
        """Re-evaluate (logprob, value, entropy) for a stored transition.

        Uses non-mutating LSTM and differentiable edge scoring so that
        gradients flow back through the GatedGCN and scoring MLP.
        """
        h, e, mask = self._gnn_forward(obs)
        id_to_idx = self._id_to_slot_idx(obs)
        h_lstm = self._lstm_forward(h, mask, obs, mutate_state=False)

        logits_op_type = self.head_op_type(h_lstm)
        value = self.head_value(h_lstm).squeeze(-1)
        device = logits_op_type.device

        # regenerate candidates for masking
        ia, ca, idl, cdl = self._candidates(obs)
        cand_lists = {0: ia, 1: idl, 2: ca, 3: cdl}

        masked_op = logits_op_type.clone()
        for oi, cl in cand_lists.items():
            if len(cl) == 0:
                masked_op[oi] = float('-inf')

        n_add = len(ia) + len(ca)
        n_del = len(idl) + len(cdl)
        if n_del > 0 and n_add > 0:
            ratio = n_add / max(1, n_del)
            if ratio > 2.0:
                boost = min(1.0, math.log(ratio))
                masked_op[1] = masked_op[1] + boost
                masked_op[3] = masked_op[3] + boost

        probs_op = torch.softmax(masked_op, dim=-1)
        dist_op = torch.distributions.Categorical(probs_op)
        logp_op = dist_op.log_prob(torch.tensor(op, device=device))
        ent_op = dist_op.entropy()

        # resolve which candidate list was used
        is_internal = (op in [0, 1])
        if is_internal:
            eval_cands = (internal_cands
                          if internal_cands else cands[:self.max_internal_candidates])
        else:
            eval_cands = cross_cands if cross_cands else []

        max_c = (self.max_internal_candidates
                 if is_internal else self.max_cross_candidates)

        if not eval_cands:
            eval_cands = [(-1, -1)]

        edge_scores = self._score_candidates(
            eval_cands, e, h_lstm, obs, id_to_idx)

        if edge_scores.numel() < max_c:
            pad = torch.full((max_c - edge_scores.numel(),),
                             float('-inf'), device=device)
            edge_scores = torch.cat([edge_scores, pad])
        edge_scores = edge_scores[:max_c]

        probs_e = torch.softmax(edge_scores, dim=-1)
        dist_e = torch.distributions.Categorical(probs_e)

        idx_c = min(idx, len(eval_cands) - 1) if eval_cands else 0
        logp_e = dist_e.log_prob(torch.tensor(idx_c, device=device))
        ent_e = dist_e.entropy()

        logprob = logp_op + logp_e
        entropy = (self.entropy_coef_op * ent_op
                   + self.entropy_coef_edge * ent_e)
        return logprob, value, entropy
