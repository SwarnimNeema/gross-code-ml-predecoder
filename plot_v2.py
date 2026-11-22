"""Generate clean, publication-quality V2 comparison plots from saved results."""
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Default data: exact 300-shot V2 run (pre_tau=0.96, route_tau=0.55)
DEFAULT_RESULTS = {
    "shots": 300,
    "p_values": [0.003, 0.004, 0.005],
    "data": {
        "0.003": {
            "bposd":    {"rate": 2.78e-4, "err": 2.8e-4, "mean_ms": 51.9,  "p99_ms": 381},
            "relay-bp": {"rate": 0.0,     "err": 0.0,    "mean_ms": 36.1,  "p99_ms": 168},
            "ml+relay": {"rate": 0.0,     "err": 0.0,    "mean_ms": 81.6,  "p99_ms": 207},
            "hier-ml":  {"rate": 0.0,     "err": 0.0,    "mean_ms": 11.9,  "p99_ms": 133},
        },
        "0.004": {
            "bposd":    {"rate": 4.56e-3, "err": 1.1e-3, "mean_ms": 144.2, "p99_ms": 419},
            "relay-bp": {"rate": 1.40e-3, "err": 6.3e-4, "mean_ms": 159.8, "p99_ms": 2387},
            "ml+relay": {"rate": 8.37e-4, "err": 4.8e-4, "mean_ms": 228.9, "p99_ms": 2509},
            "hier-ml":  {"rate": 5.57e-4, "err": 3.9e-4, "mean_ms": 128.3, "p99_ms": 2451},
        },
        "0.005": {
            "bposd":    {"rate": 2.08e-2, "err": 2.6e-3, "mean_ms": 253.2, "p99_ms": 491},
            "relay-bp": {"rate": 9.36e-3, "err": 1.7e-3, "mean_ms": 690.9, "p99_ms": 4681},
            "ml+relay": {"rate": 7.22e-3, "err": 1.5e-3, "mean_ms": 696.2, "p99_ms": 4724},
            "hier-ml":  {"rate": 8.13e-3, "err": 1.5e-3, "mean_ms": 625.1, "p99_ms": 4830},
        },
    },
}

STYLE = {
    "bposd":    {"label": "BP-OSD (baseline)",      "color": "#1f77b4", "marker": "o", "lw": 2.0, "ms": 7},
    "relay-bp": {"label": "Relay-BP (baseline)",    "color": "#2ca02c", "marker": "s", "lw": 2.0, "ms": 7},
    "ml+relay": {"label": "ML Pre-Dec + Relay-BP",  "color": "#d62728", "marker": "^", "lw": 2.0, "ms": 7.5},
    "hier-ml":  {"label": "Hierarchical ML (ours)", "color": "#7b2cbf", "marker": "D", "lw": 2.8, "ms": 8},
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)


def safe_yerr(rates, errs, floor=1.2e-4):
    """Prevent lower error bars from dropping to <= 0 on a log scale."""
    low = [min(e, max(r - floor, 0.0)) for r, e in zip(rates, errs)]
    return [low, errs]


def plot_results(payload, out_path="results/v2_comparison_clean.png"):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
    })

    shots = payload["shots"]
    p_vals = payload["p_values"]
    data = payload["data"]
    names = list(data[str(p_vals[0])].keys())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax_ler, ax_lat, ax_p4, ax_p5 = axes.ravel()

    # --- Panel 1: Logical Error Rate vs p ---
    zero_floor = 1.1e-4
    zero_names_p3 = []
    for name in names:
        st = STYLE[name]
        nz_p, nz_r, nz_e = [], [], []
        for p in p_vals:
            entry = data[str(p)][name]
            if entry["rate"] > 0:
                nz_p.append(p)
                nz_r.append(entry["rate"])
                nz_e.append(entry["err"])
            elif abs(p - p_vals[0]) < 1e-9:
                zero_names_p3.append(name)

        ax_ler.errorbar(
            nz_p, nz_r, yerr=safe_yerr(nz_r, nz_e, floor=zero_floor * 1.15),
            color=st["color"], marker=st["marker"], lw=st["lw"], markersize=st["ms"],
            capsize=4, label=st["label"], zorder=4 if name == "hier-ml" else 3,
        )
        # Draw dashed guide down to 0-failure marker at p=0.003
        if name in zero_names_p3 and len(nz_p) > 0:
            ax_ler.plot([p_vals[0], nz_p[0]], [zero_floor, nz_r[0]],
                        color=st["color"], linestyle=":", lw=1.4, alpha=0.65)
            ax_ler.plot(p_vals[0], zero_floor, marker="v", markersize=8,
                        color=st["color"], markerfacecolor="white", markeredgewidth=1.8, zorder=5)

    if zero_names_p3:
        ax_ler.annotate(
            f"0 / {shots} failures\n(Relay-BP, ML+Relay, Hier-ML)",
            xy=(p_vals[0], zero_floor),
            xytext=(p_vals[0] + 0.00028, zero_floor * 1.15),
            fontsize=8.8, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"),
        )

    ax_ler.set_yscale("log")
    ax_ler.set_ylim(7e-5, 4.5e-2)
    ax_ler.set_xlim(p_vals[0] - 0.00025, p_vals[-1] + 0.00025)
    ax_ler.set_xticks(p_vals)
    ax_ler.set_xticklabels([f"{p:.3f}" for p in p_vals])
    ax_ler.set_xlabel("Physical error rate (p)")
    ax_ler.set_ylabel("Logical error rate per cycle (log scale)")
    ax_ler.set_title("A. Decoding Accuracy (Lower is Better)")
    style_axes(ax_ler)
    ax_ler.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panel 2: Mean Latency vs p ---
    for name in names:
        st = STYLE[name]
        lat = [data[str(p)][name]["mean_ms"] for p in p_vals]
        ax_lat.plot(
            p_vals, lat, color=st["color"], marker=st["marker"],
            lw=st["lw"], markersize=st["ms"], label=st["label"],
            zorder=4 if name == "hier-ml" else 3,
        )

    # Highlight 3x speedup at p=0.003
    h_p3 = data[str(p_vals[0])]["hier-ml"]["mean_ms"]
    r_p3 = data[str(p_vals[0])]["relay-bp"]["mean_ms"]
    ax_lat.annotate(
        f"3.0× faster than Relay-BP\n({h_p3:.1f} ms vs {r_p3:.1f} ms)",
        xy=(p_vals[0], h_p3),
        xytext=(p_vals[0] + 0.00032, h_p3 + 125),
        fontsize=9, fontweight="bold", color="#7b2cbf",
        arrowprops=dict(arrowstyle="->", color="#7b2cbf", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5eef8", ec="#7b2cbf", alpha=0.95),
    )

    ax_lat.set_xlim(p_vals[0] - 0.00025, p_vals[-1] + 0.00025)
    ax_lat.set_ylim(0, max(data[str(p_vals[-1])][n]["mean_ms"] for n in names) * 1.12)
    ax_lat.set_xticks(p_vals)
    ax_lat.set_xticklabels([f"{p:.3f}" for p in p_vals])
    ax_lat.set_xlabel("Physical error rate (p)")
    ax_lat.set_ylabel("Mean latency per X+Z shot pair (ms)")
    ax_lat.set_title("B. Mean Decoding Latency (Lower is Faster)")
    style_axes(ax_lat)
    ax_lat.legend(loc="upper left", frameon=True, framealpha=0.92)

    # --- Panels 3 & 4: Pareto Trade-Offs at p = 0.004 and p = 0.005 ---
    pareto_specs = [
        (
            ax_p4, "0.004", "C. Accuracy vs. Latency at p = 0.004",
            (95, 275), (1.5e-4, 1.1e-2),
            {
                "bposd":    (10, 4, "left"),
                "relay-bp": (10, 4, "left"),
                "ml+relay": (-10, 10, "right"),
                "hier-ml":  (10, -14, "left"),
            },
        ),
        (
            ax_p5, "0.005", "D. Accuracy vs. Latency at p = 0.005",
            (180, 790), (4.5e-3, 3.2e-2),
            {
                "bposd":    (10, 4, "left"),
                "relay-bp": (-12, 8, "right"),
                "ml+relay": (-12, -16, "right"),
                "hier-ml":  (-12, 8, "right"),
            },
        ),
    ]

    for ax, p_key, title, xlim, ylim, offsets in pareto_specs:
        if p_key not in data:
            continue
        for name in names:
            st = STYLE[name]
            r = data[p_key][name]
            low = min(r["err"], max(r["rate"] - ylim[0] * 1.1, 0.0))
            ax.errorbar(
                r["mean_ms"], r["rate"], yerr=[[low], [r["err"]]],
                fmt=st["marker"], color=st["color"],
                markersize=st["ms"] + 1.5, capsize=4, lw=1.8,
                zorder=4 if name == "hier-ml" else 3,
            )
            dx, dy, ha = offsets[name]
            short_name = st["label"].split(" (")[0]
            ax.annotate(
                f"{short_name}\n({r['mean_ms']:.0f} ms)",
                xy=(r["mean_ms"], r["rate"]),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, fontsize=9,
                fontweight="bold" if name == "hier-ml" else "normal",
                color=st["color"] if name == "hier-ml" else "#222222",
            )

        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Mean latency per X+Z pair (ms)  ← Faster")
        ax.set_ylabel("Logical error rate per cycle  ← More Accurate")
        ax.set_title(title)
        style_axes(ax)
        ax.text(
            0.03, 0.05, "↙ Better & Faster", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#555555",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f3f5", ec="#ced4da"),
        )

    fig.suptitle(
        f"[[144, 12, 12]] Gross Code — Hierarchical ML-Routed Pre-Decoder ({shots} shots/basis)",
        fontsize=14, fontweight="bold", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=220)
    print(f"Saved clean plot to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            payload = json.load(f)
        out = sys.argv[2] if len(sys.argv) > 2 else "results/v2_comparison_clean.png"
    else:
        payload = DEFAULT_RESULTS
        out = "results/v2_comparison_clean.png"
    plot_results(payload, out)
