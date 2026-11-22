# Model Architecture & Technical Overview: V2 Hierarchical ML-Routed Pre-Decoder

## 1. Problem Formulation
The `[[144, 12, 12]]` Gross code is a bivariate bicycle (BB) quantum LDPC code running over 12 syndrome extraction cycles on a $12 \times 6$ periodic torus. Each circuit shot produces **1,728 detector bits**:
- **936 same-basis detectors** (forming the primary decoding matrix $H \in \mathbb{F}_2^{936 \times 8784}$)
- **792 opposite-basis detectors** (carrying correlated Pauli-Y and two-qubit gate fault signatures)

Baseline decoders (`BP-OSD` and `Relay-BP`) decode each basis independently using only the 936 same-basis detectors, discarding the 792 cross-basis bits. Meanwhile, `Relay-BP` achieves strong logical error rates by running up to 300 relay legs (`stop_nconv=5`), which creates steep mean and p99 tail latency even on easy shots.

---

## 2. V2 Architecture: 3-Stage Hierarchical Decoding Pipeline

V2 separates **easy shots** (resolved in milliseconds) from **hard or ambiguous shots** (which receive full cross-basis neural pre-decoding + full Relay-BP):

```text
All 1,728 Detectors (936 same-basis + 792 opposite-basis)
                         │
                         ▼
         [ Zero-Syndrome Fast Check ] ──(all 0)──► Return 0 (0 ms)
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Stage 1: Fast Relay-BP (~15 ms)              │
  │ pre_iter=40, num_sets=10, set_max_iter=30,   │
  │ stop_nconv=1 (same-basis 936 detectors)      │
  └──────────────────────┬───────────────────────┘
                         │
             ┌───────────┴───────────┐
      success == False        success == True
             │                       │
             │                       ▼
             │     ┌────────────────────────────────────────┐
             │     │ Stage 2: ML Confidence Router (<0.1ms) │
             │     │ 15 physics + posterior + cross-basis   │
             │     │ features -> calibrated failure risk    │
             │     └─────────────────┬──────────────────────┘
             │                       │
             │           ┌───────────┴───────────┐
             │     risk < tau_r            risk >= tau_r
             │           │                       │
             │           ▼                       │
             │   Accept Stage-1                  │
             │   (62%–99% of shots)              │
             │                                   │
             └───────────────┬───────────────────┘
                             ▼ (Escalated Hard Shots Only)
  ┌─────────────────────────────────────────────────────────┐
  │ Stage 3: Cross-Basis Tanner-Graph GNN Pre-Decoder       │
  │          (all 1,728 detectors, tau_pre = 0.96)          │
  │          + Safe Greedy Residual Peeling + Full Relay-BP │
  └─────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### Stage 1: Lightweight `fast-relay`
- Calls `relay_bp.RelayDecoderF64.decode_detailed` with capped search budget (`pre_iter=40`, `num_sets=10`, `set_max_iter=30`, `stop_nconv=1`).
- At `p = 0.004`, resolves **>93%** of shots in ~15 ms and flags non-convergent error chains directly via `res.success == False`.

### Stage 2: Sub-Millisecond ML Confidence Router (`router.py`)
- Evaluates converged Stage-1 solutions to catch rare **converged-but-wrong** logical failures before accepting them.
- Extracts **15 features** in pure NumPy (~5 µs/shot):
  1. **Syndrome & solution weights:** same-basis weight, opposite-basis weight, predicted error weight, weight-per-syndrome ratio.
  2. **Solver effort & DEM cost:** Stage-1 `iterations`, `iterations / max_iter`, normalized DEM negative log-odds cost of the proposed error, cost-per-syndrome ratio.
  3. **Cross-basis consistency (792 opposite-basis checks):** weighted opposite-basis check matches (`cross_match`), misses (`cross_miss`), and match ratio (`cross_ratio`) via sparse matrix $H_{\text{cross}}$.
  4. **Relay-BP soft margins:** minimum $|\text{posterior\_ratio}|$, count of low-confidence mechanisms ($|\text{pr}| < 2.0$), and min/mean $|\text{pr}|$ across predicted fault locations.
- **Rare-Event Training (`train_router.py`):** Class-balanced `LogisticRegression` trained on disjoint seeds (`5000+`, strictly separate from `seed=2` test sets) across `p = [0.004, 0.005, 0.006]`. Oversampling `p = 0.006` provides enough `converged_wrong` examples to overcome extreme class imbalance while preserving a 98.7% Stage-1 accept rate at `p = 0.003`.
- **Training AUROC (converged shots):** `0.9306` (X basis) and `0.9388` (Z basis).

### Stage 3: Cross-Basis Bipartite Tanner-Graph GNN Pre-Decoder (`predecoder.py`)
- Because the kit's `.stim` circuits tag detectors only by check basis (`[0.0]` for X, `[1.0]` for Z) without 3D grid coordinates, the GNN operates directly on the exact sparse Tanner graph built from Stim's Detector Error Model (DEM):
  - **Variable nodes (8,784):** error mechanisms matching `circuit.dem_matrices` column order.
  - **Same-basis check nodes (936):** connected via exact binary parity-check matrix $H$.
  - **Cross-basis check nodes (792):** connected via conditional coupling matrix $H_{\text{cross}}$ (edges retained where conditional coupling $\ge 0.05$).
- **Architecture:** 3 `BipartiteLayer` message-passing blocks (`dim=32`, `SiLU`, `LayerNorm`, residual connections). Predicts a logit shift $\Delta \ell$ added directly to exact DEM prior log-odds:
  $$\hat{p}_k = \sigma\left(\log\frac{p_k}{1 - p_k} + \Delta\ell_k\right)$$
  Zero-initializing the final head guarantees exact DEM prior behavior at step 0 and allows a single checkpoint to generalize across `p = [0.003, 0.004, 0.005]`.
- **Training (`train_predecoder.py`):** Trained on 60,000 on-the-fly DEM-sampled `(detectors, true_errors)` shots per basis across `p = [0.003, 0.004, 0.005]` on Apple Silicon MPS (`BCEWithLogitsLoss`, AdamW, cosine LR schedule), with exact $H e = s$ and $L e = o$ label verification.
- **Safe Greedy Residual Peeling:** Candidate mechanisms with $\hat{p}_k \ge \tau_{\text{pre}} = 0.96$ are peeled in descending confidence order **only if** flipping mechanism $k$ strictly reduces residual syndrome weight ($2 \cdot \text{hits} > \text{deg}(k)$). If the residual syndrome reaches zero, full `Relay-BP` is skipped; otherwise, full `Relay-BP` decodes the sparser residual $s_{\text{res}}$ and returns $e_{\text{pre}} \oplus e_{\text{fb}}$.

---

## 4. Parameter & Memory Footprint

| Component (per basis) | Parameters | Approx. Size |
| :--- | :---: | :---: |
| `TannerPreDecoder` (3-layer GNN, `dim=32`) | 57,985 | ~226.5 KB |
| `FastLinearRouter` (15 weights + intercept) | 16 | < 1 KB |
| **Total per basis** | **58,001** | **~227 KB** |

---

## 5. Evolution: V1 (`v1-predecoder`) → V2 (`v2-hierarchical-ml`)

1. **V1 (`tau = 0.80` on all shots):** Reduced syndrome weight by 61%–80% and beat `BP-OSD` accuracy at `p = 0.005`, but occasional false-positive peels injected synthetic errors into `s_res` (hurting `ml+relay`), and running the GNN on every shot added ~40 ms overhead.
2. **V2 Precision Fix (`tau_pre = 0.96`):** Raising the peeling threshold eliminated destructive false positives, cutting `ml+relay` LER at `p = 0.004` by 40% relative to baseline `Relay-BP` (`8.37e-04` vs. `1.40e-03`).
3. **V2 Latency Fix (Stage-1 + Stage-2 Router at `tau_route = 0.55`):** Accepting 98.7% (`p=0.003`), 88.0% (`p=0.004`), and 62.3% (`p=0.005`) of shots in Stage 1 delivered **3.0× faster mean latency than `Relay-BP` at `p = 0.003`** and **simultaneous accuracy + latency wins over both baselines at `p = 0.004`**.

---

## 6. Next Scope (V3)
- Scale held-out evaluation to 1,000–10,000 shots and sweep `route_tau` to map the full accuracy–latency Pareto frontier.
- Extend the pre-decoder and router to **neutral-atom heralded qubit loss** using `clifft`, feeding heralded erasure flags directly into mechanism priors and node features.
