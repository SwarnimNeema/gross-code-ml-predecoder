"""Train Stage-2 ML Confidence Routers for X and Z bases."""
import argparse
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import circuit as kit_circuit
import decoders
from predecoder import build_graph_data
from router import FastLinearRouter, build_cross_lookup, extract_router_features

FAST_RELAY_SETTINGS = {"pre_iter": 40, "num_sets": 10, "set_max_iter": 30, "stop_nconv": 1}


def collect_router_data(basis, p_shots_list):
    X_list, y_list = [], []
    total_shots = 0
    total_unconv = 0

    for idx, (p, shots) in enumerate(p_shots_list):
        c = kit_circuit.load_circuit(kit_circuit.GROSS_CIRCUITS[basis], p)
        g = build_graph_data(c, basis)
        fast_rl = decoders.relay_decoder(g["H"], g["priors"], **FAST_RELAY_SETTINGS)
        cross_lookup = build_cross_lookup(g["H_cross"])

        pr = np.clip(g["priors"], 1e-7, 1.0 - 1e-7)
        nlo = np.log((1.0 - pr) / pr)
        norm_nlo = nlo / max(float(nlo.mean()), 1e-6)

        det, obs = kit_circuit.sample(c, shots, seed=5000 + idx * 10 + (0 if basis == "X" else 1))
        det_b = det[:, g["basis_ids"]]
        det_c = det[:, g["cross_ids"]]
        L = g["L"]

        unconv = 0
        conv_wrong = 0
        for s_b, s_c, truth in zip(det_b, det_c, obs, strict=True):
            total_shots += 1
            if s_b.sum() == 0:
                continue
            res = fast_rl.decode_detailed(s_b)
            if not res.success:
                unconv += 1
                total_unconv += 1
                continue
            err = np.asarray(res.decoding, dtype=np.uint8)
            wrong = int(((L @ err) % 2 != truth).any())
            conv_wrong += wrong
            X_list.append(extract_router_features(res, s_b, s_c, cross_lookup, norm_nlo))
            y_list.append(wrong)

        print(f"  basis={basis} p={p}: shots={shots} | unconverged={unconv} | converged_wrong={conv_wrong}")

    return np.vstack(X_list), np.array(y_list, dtype=np.int32), total_shots, total_unconv


def train_basis(basis, p_shots_list):
    t0 = time.perf_counter()
    print(f"\nCollecting Stage-1 router data for basis={basis}...")
    X, y, total_shots, total_unconv = collect_router_data(basis, p_shots_list)
    print(f"  Converged samples: {len(y)} (positive/wrong: {y.sum()}) | elapsed {time.perf_counter() - t0:.1f}s")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if y.sum() >= 2:
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=500)
        clf.fit(Xs, y)
        probs = clf.predict_proba(Xs)[:, 1]
        auc = roc_auc_score(y, probs)
        print(f"  Router AUROC (converged shots): {auc:.4f}")
        for th in (0.50, 0.70, 0.80, 0.90):
            esc = 100.0 * (probs >= th).mean()
            rec = 100.0 * (probs[y == 1] >= th).mean()
            print(f"    threshold={th:.2f} -> escalates {esc:4.1f}% of converged shots | catches {rec:5.1f}% of converged_wrong")
        router = FastLinearRouter(scaler.mean_, scaler.scale_, clf.coef_[0], clf.intercept_[0])
    else:
        print("  Very few converged_wrong examples; saving conservative iteration/cost fallback router.")
        coef = np.zeros(X.shape[1], dtype=np.float64)
        coef[3] = 1.0  # iterations
        coef[5] = 1.0  # err_cost
        router = FastLinearRouter(scaler.mean_, scaler.scale_, coef, -3.0)

    os.makedirs("models", exist_ok=True)
    out_path = f"models/router_{basis}.npz"
    router.save(out_path)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots-p4", type=int, default=1200)
    parser.add_argument("--shots-p5", type=int, default=1800)
    parser.add_argument("--shots-p6", type=int, default=1000)
    args = parser.parse_args()

    p_shots = [(0.004, args.shots_p4), (0.005, args.shots_p5), (0.006, args.shots_p6)]
    for basis in ("X", "Z"):
        train_basis(basis, p_shots)


if __name__ == "__main__":
    main()
