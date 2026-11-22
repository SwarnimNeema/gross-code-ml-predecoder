"""Train the Gross code Tanner-Graph Pre-Decoder on M4 MPS (or CPU)."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import circuit as kit_circuit
from predecoder import TannerPreDecoder, build_graph_data


def sample_dem_chunk(graph, shots, seed, verify=False):
    """Sample (s_basis, s_cross, e_true) directly from the Stim DEM."""
    sampler = graph["dem"].compile_sampler(seed=seed)
    det, obs, err = sampler.sample(shots, return_errors=True)
    raw_to_col = graph["raw_to_col"]
    n_v = graph["H"].shape[1]

    e_true = np.zeros((shots, n_v), dtype=np.uint8)
    r_idx, raw_c = np.nonzero(err)
    mapped = raw_to_col[raw_c]
    keep = mapped >= 0
    np.bitwise_xor.at(e_true, (r_idx[keep], mapped[keep]), 1)

    s_b = det[:, graph["basis_ids"]].astype(np.float32)
    s_c = det[:, graph["cross_ids"]].astype(np.float32)

    if verify:
        syn_check = (graph["H"] @ e_true.T).T % 2
        obs_check = (graph["L"] @ e_true.T).T % 2
        assert np.array_equal(syn_check, det[:, graph["basis_ids"]]), "Syndrome H*e mismatch!"
        assert np.array_equal(obs_check, obs), "Observable L*e mismatch!"

    return s_b, s_c, e_true.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", choices=["X", "Z"], required=True)
    parser.add_argument("--chunks", type=int, default=30, help="2000 shots per chunk (30 = 60,000 shots)")
    parser.add_argument("--chunk-shots", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    out_path = args.out or f"models/predecoder_{args.basis}.pt"

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Training basis={args.basis} on device={device}")

    p_list = [0.003, 0.004, 0.005]
    graphs = {}
    log_priors = {}
    for p in p_list:
        c = kit_circuit.load_circuit(kit_circuit.GROSS_CIRCUITS[args.basis], p)
        g = build_graph_data(c, args.basis)
        graphs[p] = g
        pr = np.clip(g["priors"], 1e-7, 1.0 - 1e-7)
        log_priors[p] = torch.tensor(np.log(pr / (1.0 - pr)), dtype=torch.float32, device=device)

    ref_g = graphs[0.004]
    model = TannerPreDecoder(ref_g["H"], ref_g["H_cross"], dim=args.dim, num_layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params * 4 / 1024:.1f} KB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.chunks * (args.chunk_shots // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    loss_fn = nn.BCEWithLogitsLoss()

    # Fixed validation set at p=0.004 (seed 99999)
    val_sb, val_sc, val_e = sample_dem_chunk(graphs[0.004], 1000, seed=99999, verify=True)
    print("DEM-to-H/L exact label verification PASSED.")

    best_val_loss = float("inf")
    t0 = time.perf_counter()

    for chunk in range(args.chunks):
        p = p_list[chunk % len(p_list)]
        s_b, s_c, e_true = sample_dem_chunk(graphs[p], args.chunk_shots, seed=1000 + chunk)

        model.train()
        perm = np.random.permutation(args.chunk_shots)
        running_loss = 0.0
        n_batches = 0

        for start in range(0, args.chunk_shots - args.batch_size + 1, args.batch_size):
            idx = perm[start : start + args.batch_size]
            xb = torch.from_numpy(s_b[idx]).to(device)
            xc = torch.from_numpy(s_c[idx]).to(device)
            yb = torch.from_numpy(e_true[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, xc, log_priors[p])
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        # Quick validation every 3 chunks (and final chunk)
        if (chunk + 1) % 3 == 0 or (chunk + 1) == args.chunks:
            model.eval()
            val_loss = 0.0
            tp = fp = fn = 0
            with torch.inference_mode():
                for start in range(0, len(val_sb), args.batch_size):
                    xb = torch.from_numpy(val_sb[start : start + args.batch_size]).to(device)
                    xc = torch.from_numpy(val_sc[start : start + args.batch_size]).to(device)
                    yb = torch.from_numpy(val_e[start : start + args.batch_size]).to(device)
                    logits = model(xb, xc, log_priors[0.004])
                    val_loss += loss_fn(logits, yb).item() * len(xb)
                    preds = torch.sigmoid(logits) >= 0.80
                    truth = yb > 0.5
                    tp += (preds & truth).sum().item()
                    fp += (preds & ~truth).sum().item()
                    fn += (~preds & truth).sum().item()
            val_loss /= len(val_sb)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            elapsed = time.perf_counter() - t0
            print(
                f"Chunk {chunk+1:02d}/{args.chunks} (p={p}) | "
                f"train_bce={running_loss/n_batches:.5f} | val_bce={val_loss:.5f} | "
                f"P@0.8={prec:.3f} R@0.8={rec:.3f} | {elapsed:.1f}s"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "dim": args.dim,
                        "layers": args.layers,
                        "basis": args.basis,
                    },
                    out_path,
                )

    print(f"Saved best model to {out_path} (val_bce={best_val_loss:.5f})")


if __name__ == "__main__":
    main()
