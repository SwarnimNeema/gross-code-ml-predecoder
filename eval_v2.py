"""Benchmark V2 Hierarchical ML-Routed Pre-Decoder against BP-OSD and Relay-BP."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import circuit as kit_circuit
import decoders
from predecoder import LearnedPreDecoderWrapper, TannerPreDecoder, build_graph_data
from router import FastLinearRouter, HierarchicalMLDecoder

FAST_RELAY_SETTINGS = {"pre_iter": 40, "num_sets": 10, "set_max_iter": 30, "stop_nconv": 1}


def load_predecoder(ckpt_path, graph):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = TannerPreDecoder(graph["H"], graph["H_cross"], dim=ckpt["dim"], num_layers=ckpt["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_at_p(p, shots, pre_tau, route_tau, include_ml_relay=True):
    tag = f"p{int(round(p * 1000)):03d}"
    datasets, graphs, models, routers = {}, {}, {}, {}

    for basis in ("X", "Z"):
        d = np.load(f"data/test_{basis}_{tag}.npz")
        datasets[basis] = (d["detectors"][:shots], d["observables"][:shots], d["basis_detectors"])
        c = kit_circuit.load_circuit(kit_circuit.GROSS_CIRCUITS[basis], p)
        graphs[basis] = build_graph_data(c, basis)
        models[basis] = load_predecoder(f"models/predecoder_{basis}.pt", graphs[basis])
        routers[basis] = FastLinearRouter.load(f"models/router_{basis}.npz")

    n_params = sum(param.numel() for param in models["X"].parameters()) + 16
    print(f"\n=== p = {p} | {shots} shots/basis | pre_tau={pre_tau} route_tau={route_tau} | params/basis={n_params:,} ===")

    summary = {}
    configs = ["bposd", "relay-bp"]
    if include_ml_relay:
        configs.append("ml+relay")
    configs.append("hier-ml")

    for name in configs:
        failed_cols, sec_cols = [], []
        s1_rates, unconv_rates, r_esc_rates = [], [], []

        for basis in ("X", "Z"):
            det_all, obs, b_ids = datasets[basis]
            g = graphs[basis]

            if name == "bposd":
                dec = decoders.bposd_decoder(g["H"], g["priors"])
                f, s = decoders.score(dec.decode, g["L"], det_all[:, b_ids], obs)
            elif name == "relay-bp":
                dec = decoders.relay_decoder(g["H"], g["priors"])
                f, s = decoders.score(dec.decode, g["L"], det_all[:, b_ids], obs)
            elif name == "ml+relay":
                fb = decoders.relay_decoder(g["H"], g["priors"])
                wrap = LearnedPreDecoderWrapper(models[basis], g, fb, threshold=pre_tau)
                f, s = decoders.score(wrap, g["L"], det_all, obs)
            elif name == "hier-ml":
                fast_rl = decoders.relay_decoder(g["H"], g["priors"], **FAST_RELAY_SETTINGS)
                full_rl = decoders.relay_decoder(g["H"], g["priors"])
                stage3 = LearnedPreDecoderWrapper(models[basis], g, full_rl, threshold=pre_tau)
                hier = HierarchicalMLDecoder(fast_rl, routers[basis], stage3, g, route_threshold=route_tau)
                f, s = decoders.score(hier, g["L"], det_all, obs)
                n = max(hier.total_shots, 1)
                s1_rates.append(100.0 * (hier.zero_exits + hier.stage1_accepts) / n)
                unconv_rates.append(100.0 * hier.unconv_escalations / n)
                r_esc_rates.append(100.0 * hier.router_escalations / n)

            failed_cols.append(f)
            sec_cols.append(s)

        failed = np.column_stack(failed_cols)
        seconds = np.column_stack(sec_cols)
        decoders.print_score(name, failed, seconds, cycles=12)

        rate, err = decoders.per_cycle_rate(failed.any(axis=1), cycles=12)
        ms = seconds.sum(axis=1) * 1e3
        summary[name] = {
            "rate": rate,
            "err": err,
            "mean_ms": float(ms.mean()),
            "p99_ms": float(np.percentile(ms, 99)),
        }
        if name == "hier-ml":
            print(
                f"   -> fast Stage-1 accept: {np.mean(s1_rates):.1f}% | "
                f"unconverged esc: {np.mean(unconv_rates):.1f}% | "
                f"router esc: {np.mean(r_esc_rates):.1f}%"
            )

    return summary


def make_plots(all_results, p_values, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(all_results[p_values[0]].keys())
    colors = {"bposd": "#1f77b4", "relay-bp": "#2ca02c", "ml+relay": "#d62728", "hier-ml": "#9467bd"}
    markers = {"bposd": "o", "relay-bp": "s", "ml+relay": "^", "hier-ml": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for name in names:
        c = colors.get(name)
        m = markers.get(name, "o")
        rates = [all_results[p][name]["rate"] for p in p_values]
        errs = [all_results[p][name]["err"] for p in p_values]
        lat = [all_results[p][name]["mean_ms"] for p in p_values]
        lw = 2.6 if name == "hier-ml" else 1.8

        axes[0].errorbar(p_values, rates, yerr=errs, marker=m, color=c, lw=lw, capsize=4, label=name)
        axes[1].plot(p_values, lat, marker=m, color=c, lw=lw, label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Physical error rate p")
    axes[0].set_ylabel("Logical error rate per cycle")
    axes[0].set_title("Accuracy: [[144, 12, 12]] Gross Code")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Physical error rate p")
    axes[1].set_ylabel("Mean decoding time per X+Z pair (ms)")
    axes[1].set_title("Mean Decoding Latency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel 3: Accuracy vs. Latency trade-off at highest noise point (or p=0.004 if present)
    p_focus = 0.005 if 0.005 in all_results else p_values[-1]
    for name in names:
        r = all_results[p_focus][name]
        axes[2].errorbar(
            r["mean_ms"], max(r["rate"], 1e-6), yerr=r["err"],
            fmt=markers.get(name, "o"), color=colors.get(name),
            markersize=9 if name == "hier-ml" else 7, capsize=4, label=name,
        )
        axes[2].annotate(name, (r["mean_ms"], max(r["rate"], 1e-6)), textcoords="offset points", xytext=(7, 4), fontsize=9)

    axes[2].set_yscale("log")
    axes[2].set_xlabel("Mean latency per X+Z pair (ms)  [Lower is faster]")
    axes[2].set_ylabel("Logical error rate per cycle  [Lower is better]")
    axes[2].set_title(f"Accuracy vs. Latency Trade-Off (p = {p_focus})")
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"\nSaved V2 comparison plot to {out_path}")


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=300)
    parser.add_argument("--pre-tau", type=float, default=0.96)
    parser.add_argument("--route-tau", type=float, default=0.80)
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--skip-ml-relay", action="store_true")
    parser.add_argument("--plot", default="results/v2_comparison.png")
    args = parser.parse_args()

    all_results = {}
    for p in args.p:
        all_results[p] = eval_at_p(p, args.shots, args.pre_tau, args.route_tau, include_ml_relay=not args.skip_ml_relay)

    import json
    from plot_v2 import plot_results
    payload = {
        "shots": args.shots,
        "p_values": args.p,
        "data": {str(p): all_results[p] for p in args.p},
    }
    json_path = os.path.splitext(args.plot)[0] + ".json"
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved raw metrics to {json_path}")
    plot_results(payload, args.plot)


if __name__ == "__main__":
    main()
