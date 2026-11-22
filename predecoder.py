"""Bipartite Tanner-Graph Learned Pre-Decoder for the [[144, 12, 12]] Gross code."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import circuit as kit_circuit


def build_graph_data(stim_circuit, basis):
    """Build exact DEM-to-column mapping, basis matrix H, and cross-basis coupling H_cross."""
    other_basis = "Z" if basis == "X" else "X"
    basis_ids = kit_circuit.detector_ids(stim_circuit, basis)
    cross_ids = kit_circuit.detector_ids(stim_circuit, other_basis)
    basis_rows = {int(d): r for r, d in enumerate(basis_ids)}
    cross_rows = {int(d): r for r, d in enumerate(cross_ids)}

    H, L, priors = kit_circuit.dem_matrices(stim_circuit, basis)
    dem = stim_circuit.detector_error_model().flattened()

    # Reproduce circuit._merge_duplicates exact column ordering while mapping raw DEM errors.
    effect_to_col = {}
    raw_to_col = []
    col_total_p = {}
    cross_acc = {}  # (cross_row, col) -> weighted prob sum

    for inst in dem:
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        targets = inst.targets_copy()
        b_dets = tuple(sorted(basis_rows[t.val] for t in targets if t.is_relative_detector_id() and t.val in basis_rows))
        obs = tuple(sorted(t.val for t in targets if t.is_logical_observable_id()))
        effect = (b_dets, obs)
        if effect == ((), ()):
            raw_to_col.append(-1)
            continue
        if effect not in effect_to_col:
            effect_to_col[effect] = len(effect_to_col)
        col = effect_to_col[effect]
        raw_to_col.append(col)
        col_total_p[col] = col_total_p.get(col, 0.0) + p

        c_dets = [cross_rows[t.val] for t in targets if t.is_relative_detector_id() and t.val in cross_rows]
        for cr in c_dets:
            key = (cr, col)
            cross_acc[key] = cross_acc.get(key, 0.0) + p

    assert len(effect_to_col) == H.shape[1], "Column mismatch with circuit.dem_matrices!"

    # Build sparse cross-basis matrix H_cross (conditional probability >= 0.05)
    c_r, c_c, c_v = [], [], []
    for (cr, col), p_sum in cross_acc.items():
        w = p_sum / max(col_total_p[col], 1e-12)
        if w >= 0.05:
            c_r.append(cr)
            c_c.append(col)
            c_v.append(min(w, 1.0))
    H_cross = sp.csc_matrix(
        (np.array(c_v, dtype=np.float32), (c_r, c_c)),
        shape=(len(cross_ids), H.shape[1]),
    )

    # Precompute column -> detector row arrays for microsecond greedy peeling
    H_csc = H.tocsc()
    col_rows = [
        H_csc.indices[H_csc.indptr[k] : H_csc.indptr[k + 1]].astype(np.int32)
        for k in range(H_csc.shape[1])
    ]

    return {
        "basis": basis,
        "H": H_csc,
        "H_cross": H_cross,
        "L": L,
        "priors": np.ascontiguousarray(priors, dtype=np.float64),
        "basis_ids": basis_ids,
        "cross_ids": cross_ids,
        "raw_to_col": np.array(raw_to_col, dtype=np.int32),
        "col_rows": col_rows,
        "dem": dem,
    }


class BipartiteLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v2b = nn.Linear(dim, dim, bias=False)
        self.v2c = nn.Linear(dim, dim, bias=False)
        self.b2v = nn.Linear(dim, dim, bias=False)
        self.c2v = nn.Linear(dim, dim, bias=False)

        self.b_upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.c_upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.v_upd = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

        self.b_norm = nn.LayerNorm(dim)
        self.c_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)

    def forward(self, h_v, h_b, h_c, eb_det, eb_var, deg_b_det, deg_b_var, ec_det, ec_var, ec_w, deg_c_det, deg_c_var):
        # 1. Variable -> Basis checks
        m_v2b = self.v2b(h_v)[eb_var]
        agg_b = torch.zeros_like(h_b)
        agg_b.index_add_(0, eb_det, m_v2b)
        agg_b = agg_b / deg_b_det[:, None, None]
        h_b = self.b_norm(h_b + self.b_upd(torch.cat([h_b, agg_b], dim=-1)))

        # 2. Variable -> Cross-basis checks
        if ec_det.numel() > 0:
            m_v2c = self.v2c(h_v)[ec_var] * ec_w[:, None, None]
            agg_c = torch.zeros_like(h_c)
            agg_c.index_add_(0, ec_det, m_v2c)
            agg_c = agg_c / deg_c_det[:, None, None]
            h_c = self.c_norm(h_c + self.c_upd(torch.cat([h_c, agg_c], dim=-1)))

        # 3. Checks -> Variables
        m_b2v = self.b2v(h_b)[eb_det]
        agg_vb = torch.zeros_like(h_v)
        agg_vb.index_add_(0, eb_var, m_b2v)
        agg_vb = agg_vb / deg_b_var[:, None, None]

        agg_vc = torch.zeros_like(h_v)
        if ec_det.numel() > 0:
            m_c2v = self.c2v(h_c)[ec_det] * ec_w[:, None, None]
            agg_vc.index_add_(0, ec_var, m_c2v)
            agg_vc = agg_vc / deg_c_var[:, None, None]

        h_v = self.v_norm(h_v + self.v_upd(torch.cat([h_v, agg_vb, agg_vc], dim=-1)))
        return h_v, h_b, h_c


class TannerPreDecoder(nn.Module):
    """Lightweight Tanner-graph neural pre-decoder (works on MPS and CPU)."""

    def __init__(self, H_basis, H_cross, dim=32, num_layers=3):
        super().__init__()
        self.n_b, self.n_v = H_basis.shape
        self.n_c = H_cross.shape[0]
        self.dim = dim
        self.num_layers = num_layers

        Hb = H_basis.tocoo()
        Hc = H_cross.tocoo()

        self.register_buffer("eb_det", torch.tensor(Hb.row, dtype=torch.long))
        self.register_buffer("eb_var", torch.tensor(Hb.col, dtype=torch.long))
        self.register_buffer("ec_det", torch.tensor(Hc.row, dtype=torch.long))
        self.register_buffer("ec_var", torch.tensor(Hc.col, dtype=torch.long))
        self.register_buffer("ec_w", torch.tensor(Hc.data, dtype=torch.float32))

        deg_b_det = np.maximum(np.asarray(H_basis.sum(axis=1)).ravel(), 1.0).astype(np.float32)
        deg_b_var = np.maximum(np.asarray(H_basis.sum(axis=0)).ravel(), 1.0).astype(np.float32)
        deg_c_det = np.maximum(np.asarray(H_cross.sum(axis=1)).ravel(), 1.0).astype(np.float32)
        deg_c_var = np.maximum(np.asarray(H_cross.sum(axis=0)).ravel(), 1.0).astype(np.float32)

        self.register_buffer("deg_b_det", torch.tensor(deg_b_det))
        self.register_buffer("deg_b_var", torch.tensor(deg_b_var))
        self.register_buffer("deg_c_det", torch.tensor(deg_c_det))
        self.register_buffer("deg_c_var", torch.tensor(deg_c_var))

        self.v_enc = nn.Linear(4, dim)
        self.b_enc = nn.Linear(2, dim)
        self.c_enc = nn.Linear(2, dim)
        self.layers = nn.ModuleList([BipartiteLayer(dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        # Start near zero delta so initial output matches exact DEM prior log-odds
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, s_basis, s_cross, log_priors):
        """s_basis: (B, n_b), s_cross: (B, n_c), log_priors: (n_v,) -> logits (B, n_v)."""
        B = s_basis.shape[0]
        sb_t = s_basis.transpose(0, 1)  # (n_b, B)
        sc_t = s_cross.transpose(0, 1)  # (n_c, B)

        # Dynamic first-hop syndrome-match features on each mechanism node
        b_hits = torch.zeros((self.n_v, B), device=s_basis.device, dtype=s_basis.dtype)
        b_hits.index_add_(0, self.eb_var, sb_t[self.eb_det])
        b_frac = b_hits / self.deg_b_var[:, None]

        c_hits = torch.zeros((self.n_v, B), device=s_basis.device, dtype=s_basis.dtype)
        if self.ec_det.numel() > 0:
            c_hits.index_add_(0, self.ec_var, sc_t[self.ec_det] * self.ec_w[:, None])

        lp_norm = (log_priors / 10.0)[:, None].expand(self.n_v, B)
        v_in = torch.stack([lp_norm, b_hits * 0.25, b_frac, c_hits * 0.25], dim=-1)
        b_in = torch.stack([sb_t, (self.deg_b_det / 40.0)[:, None].expand(self.n_b, B)], dim=-1)
        c_in = torch.stack([sc_t, (self.deg_c_det / 40.0)[:, None].expand(self.n_c, B)], dim=-1)

        h_v = self.v_enc(v_in)
        h_b = self.b_enc(b_in)
        h_c = self.c_enc(c_in)

        for layer in self.layers:
            h_v, h_b, h_c = layer(
                h_v, h_b, h_c,
                self.eb_det, self.eb_var, self.deg_b_det, self.deg_b_var,
                self.ec_det, self.ec_var, self.ec_w, self.deg_c_det, self.deg_c_var,
            )

        delta = self.head(h_v).squeeze(-1).transpose(0, 1)  # (B, n_v)
        return log_priors.unsqueeze(0) + delta


class LearnedPreDecoderWrapper:
    """NVIDIA Ising-style residual pre-decoder wrapping BP-OSD or Relay-BP."""

    def __init__(self, model, graph_data, fallback_decoder, threshold=0.80):
        self.model = model.to("cpu").eval()
        self.basis_ids = graph_data["basis_ids"]
        self.cross_ids = graph_data["cross_ids"]
        self.col_rows = graph_data["col_rows"]
        self.n_v = graph_data["H"].shape[1]
        p = np.clip(graph_data["priors"], 1e-7, 1.0 - 1e-7)
        self.log_priors = torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        self.fallback = fallback_decoder
        self.threshold = threshold
        self.reset_stats()

    def reset_stats(self):
        self.total_shots = 0
        self.zero_syndrome_shots = 0
        self.fast_exits = 0
        self.fallbacks = 0
        self.orig_weight_sum = 0
        self.res_weight_sum = 0

    def __call__(self, syndrome_all):
        self.total_shots += 1
        s_b = syndrome_all[self.basis_ids]
        orig_w = int(s_b.sum())
        self.orig_weight_sum += orig_w

        if orig_w == 0:
            self.zero_syndrome_shots += 1
            self.fast_exits += 1
            return np.zeros(self.n_v, dtype=np.uint8)

        s_c = syndrome_all[self.cross_ids]
        with torch.inference_mode():
            sb_t = torch.from_numpy(s_b.astype(np.float32)).unsqueeze(0)
            sc_t = torch.from_numpy(s_c.astype(np.float32)).unsqueeze(0)
            probs = torch.sigmoid(self.model(sb_t, sc_t, self.log_priors)).squeeze(0).numpy()

        # Greedy confidence-ordered peeling: only accept faults that strictly reduce syndrome weight
        e_pre = np.zeros(self.n_v, dtype=np.uint8)
        s_res = s_b.copy()
        cands = np.flatnonzero(probs >= self.threshold)
        if len(cands) > 0:
            order = cands[np.argsort(-probs[cands])]
            for k in order:
                rows = self.col_rows[k]
                if len(rows) == 0:
                    continue
                hits = int(s_res[rows].sum())
                if 2 * hits > len(rows):
                    s_res[rows] ^= 1
                    e_pre[k] ^= 1

        res_w = int(s_res.sum())
        self.res_weight_sum += res_w

        if res_w == 0:
            self.fast_exits += 1
            return e_pre

        self.fallbacks += 1
        e_fb = np.asarray(self.fallback.decode(s_res), dtype=np.uint8)
        return e_pre ^ e_fb
