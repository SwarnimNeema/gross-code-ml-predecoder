"""Stage-2 ML Confidence Router and V2 Hierarchical Decoder (fast-relay -> Router -> ml+relay)."""
import numpy as np


def build_cross_lookup(H_cross):
    """Precompute per-mechanism opposite-basis check indices and weights for <0.05 ms scoring."""
    Hc = H_cross.tocsc()
    lookup = []
    for k in range(Hc.shape[1]):
        start, end = Hc.indptr[k], Hc.indptr[k + 1]
        lookup.append((Hc.indices[start:end].astype(np.int32), Hc.data[start:end].astype(np.float32)))
    return lookup


def extract_router_features(res, s_b, s_c, cross_lookup, norm_nlo):
    """Extract 14 sub-millisecond features from Stage-1 fast-relay + cross-basis checks."""
    err = np.asarray(res.decoding, dtype=np.uint8)
    err_idx = np.flatnonzero(err)

    syn_w = float(s_b.sum())
    cross_w = float(s_c.sum())
    err_w = float(len(err_idx))
    iters = float(res.iterations)
    iter_frac = iters / max(float(res.max_iter), 1.0)

    # Normalized DEM negative log-odds cost of Stage-1's proposed error
    err_cost = float(norm_nlo[err_idx].sum()) if len(err_idx) > 0 else 0.0
    cost_per_syn = err_cost / max(syn_w, 1.0)
    w_per_syn = err_w / max(syn_w, 1.0)

    # Cross-basis consistency against the 792 opposite-basis detectors
    cross_match = 0.0
    cross_miss = 0.0
    if len(err_idx) > 0:
        for k in err_idx:
            rows, vals = cross_lookup[k]
            if len(rows) > 0:
                hits = s_c[rows]
                cross_match += float(vals[hits > 0].sum())
                cross_miss += float(vals[hits == 0].sum())
    cross_ratio = cross_match / max(cross_match + cross_miss, 1e-6)

    # Soft confidence stats from Relay-BP posterior_ratios
    pr = np.nan_to_num(np.asarray(res.posterior_ratios, dtype=np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    abs_pr = np.clip(np.abs(pr), 0.0, 50.0)
    min_abs_pr = float(abs_pr.min())
    low_conf_cnt = float((abs_pr < 2.0).sum())
    if len(err_idx) > 0:
        err_pr = abs_pr[err_idx]
        err_min_pr = float(err_pr.min())
        err_mean_pr = float(err_pr.mean())
    else:
        err_min_pr = 50.0
        err_mean_pr = 50.0

    return np.array(
        [
            syn_w,
            cross_w,
            err_w,
            iters,
            iter_frac,
            err_cost,
            cost_per_syn,
            w_per_syn,
            cross_match,
            cross_miss,
            cross_ratio,
            min_abs_pr,
            low_conf_cnt,
            err_min_pr,
            err_mean_pr,
        ],
        dtype=np.float64,
    )


class FastLinearRouter:
    """Standardized logistic router evaluated in pure NumPy (~5 microseconds per shot)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)

    def predict_risk(self, x):
        z = float(np.dot((x - self.mean) / self.scale, self.coef) + self.intercept)
        z = max(min(z, 30.0), -30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["scale"], d["coef"], float(d["intercept"]))


class HierarchicalMLDecoder:
    """V2 Hierarchical Decoder: fast-relay -> ML Confidence Router -> ml+relay (tau=0.96)."""

    def __init__(self, fast_relay, router, stage3_decoder, graph_data, route_threshold=0.75):
        self.fast_relay = fast_relay
        self.router = router
        self.stage3 = stage3_decoder
        self.basis_ids = graph_data["basis_ids"]
        self.cross_ids = graph_data["cross_ids"]
        self.n_v = graph_data["H"].shape[1]
        self.cross_lookup = build_cross_lookup(graph_data["H_cross"])

        p = np.clip(graph_data["priors"], 1e-7, 1.0 - 1e-7)
        nlo = np.log((1.0 - p) / p)
        self.norm_nlo = nlo / max(float(nlo.mean()), 1e-6)
        self.route_threshold = route_threshold
        self.reset_stats()

    def reset_stats(self):
        self.total_shots = 0
        self.zero_exits = 0
        self.stage1_accepts = 0
        self.unconv_escalations = 0
        self.router_escalations = 0

    def __call__(self, syndrome_all):
        self.total_shots += 1
        s_b = syndrome_all[self.basis_ids]
        if s_b.sum() == 0:
            self.zero_exits += 1
            return np.zeros(self.n_v, dtype=np.uint8)

        res = self.fast_relay.decode_detailed(s_b)
        if res.success:
            s_c = syndrome_all[self.cross_ids]
            feats = extract_router_features(res, s_b, s_c, self.cross_lookup, self.norm_nlo)
            risk = self.router.predict_risk(feats)
            if risk < self.route_threshold:
                self.stage1_accepts += 1
                return np.asarray(res.decoding, dtype=np.uint8)
            self.router_escalations += 1
        else:
            self.unconv_escalations += 1

        return self.stage3(syndrome_all)
