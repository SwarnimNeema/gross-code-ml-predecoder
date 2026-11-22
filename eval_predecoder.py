"""Benchmark Learned Pre-Decoder + BP-OSD / Relay-BP against baselines on held-out test sets."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import circuit as kit_circuit
import decoders
from predecoder import LearnedPreDecoderWrapper, TannerPreDecoder, build_graph_data


def load_model(ckpt_path, graph):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = TannerPreDecoder(graph["H"], graph["H_cross"], dim=ckpt["dim"], num_layers=ckpt["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def has_relay():
    try:
        import relay_bp  # noqa: F401
        return True
    except ImportError:
        return False


def eval_at_p(p, shots, threshold, run_relay=True):
    tag = f"p{int(round(p * 1000)):03d}"
    datasets = {}
    graphs = {}
    models = {}

    for basis in ("X", "Z"):
        npz_path = f"data/test_{basis}_{tag}.npz"
        d = np.load(npz_path)
        datasets[basis] = (d["detectors"][:shots], d["observables"][:shots], d["basis_detectors"])
        c = kit_circuit.load_circuit(kit_circuit.GROSS_CIRCUITS[basis], p)
        graphs[basis] = build_graph_data(c, basis)
        models[basis] = load_model(f"models/predecoder_{basis}.pt", graphs[basis])

    n_params = sum(param.numel() for param in models["X"].parameters())
    print(f"\n=== p = {p} | {shots} shots/basis | tau = {threshold} | params/basis = {n_params:,} ({n_params*4/1024:.1f} KB) ===")

    summary = {}
    configs = [("bposd", False, "bposd"), ("ml+bposd", True, "bposd")]
    if run_relay and has_relay():
        configs += [("relay-bp", False, "relay"), ("ml+relay", True, "relay")]

    for name, use_ml, backend in configs:
        failed_cols = []
        sec_cols = []
        fast_rates = []
        red_rates = []

        for basis in ("X", "Z"):
            det_all, obs, b_ids = datasets[basis]
            g = graphs[basis]
            fb = (
                decoders.bposd_decoder(g["H"], g["priors"])
                if backend == "bposd"
                else decoders.relay_decoder(g["H"], g["priors"])
            )
            if use_ml:
                wrapper = LearnedPreDecoderWrapper(models[basis], g, fb, threshold=threshold)
                f, s = decoders.score(wrapper, g["L"], det_all, obs)
                fast_rates.append(100.0 * wrapper.fast_exits / max(wrapper.total_shots, 1))
                red = 100.0 * (1.0 - wrapper.res_weight_sum / max(wrapper.orig_weight_sum, 1))
                red_rates.append(red)
            else:
                f, s = decoders.score(fb.decode, g["L"], det_all[:, b_ids], obs)

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
        if use_ml:
            print(
                f"   -> fast-path exit: {np.mean(fast_rates):.1f}% of shots | "
                f"syndrome weight reduced by {np.mean(red_rates):.1f}%"
            )

    return summary


def make_plots(all_results, p_values, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(all_results[p_values[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in names:
        rates = [all_results[p][name]["rate"] for p in p_values]
        errs = [all_results[p][name]["err"] for p in p_values]
        lat = [all_results[p][name]["mean_ms"] for p in p_values]
        axes[0].errorbar(p_values, rates, yerr=errs, marker="o", capsize=4, label=name)
        axes[1].plot(p_values, lat, marker="o", label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Physical error rate p")
    axes[0].set_ylabel("Logical error rate per cycle")
    axes[0].set_title("Accuracy: Gross Code [[144, 12, 12]]")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Physical error rate p")
    axes[1].set_ylabel("Mean decoding time per X+Z pair (ms)")
    axes[1].set_title("Latency Comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"\nSaved comparison plot to {out_path}")


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=500, help="Use 500 for fast checks, 10000 for final run")
    parser.add_argument("--tau", type=float, default=0.80, help="Pre-decoder confidence threshold")
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--no-relay", action="store_true")
    parser.add_argument("--plot", default="results/predecoder_comparison.png")
    args = parser.parse_args()

    all_results = {}
    for p in args.p:
        all_results[p] = eval_at_p(p, args.shots, args.tau, run_relay=not args.no_relay)

    make_plots(all_results, args.p, args.plot)


if __name__ == "__main__":
    main()
