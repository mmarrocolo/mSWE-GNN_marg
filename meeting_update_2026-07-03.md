# Progress update — since feedback of Jun 27

Baseline at feedback time: train loss ~0.05, val loss ~0.5, **val CSI@0.05 capped at ~0.6** on the
Ahr 2021 event (single simulation, warm start, 100 m SFINCS reference).

---

## 1. Feedback: "try MAE as an alternative to the shallow-WD weighting" → diagnosed first

Both suggestions attack the same suspected problem: under RMSE, squared errors let the few large
deep-water errors dominate the loss, so the model may neglect the small errors at shallow cells that
decide wet/dry classification. `shallow_weight` fixes that explicitly (upweight shallow cells);
MAE fixes it implicitly (linear errors → small errors count relatively more).

Before switching, I checked whether that suspected problem is actually what caps CSI
(notebook: `utils/compare_weighted_rmse_runs.ipynb`, 4 weighted-RMSE checkpoints on the same
full 118-step rollout):

- **CSI decays monotonically** from 1.0 (warm start) to ~0.35–0.45 → rollout error compounding (drift).
- **False positives dominate**: FP/real-wet grows steadily to 0.7–1.3 by the end of the rollout —
  the model progressively over-wets and never dries out on the recession.
  (spatial maps: scattered false alarms across the floodplain + dense cluster downstream)
- **Rising limb/peak**: one contiguous miss-band along the downstream reach → the model's flood wave
  lags SFINCS (front timing), recovers after the peak.
- **Fringe hypothesis rejected**: only 1–14 % of missed cells are within 5 cm of the 0.05 m threshold —
  misses are genuinely deep cells (the lagging front). False alarms also predict depths > 0.5 m.

**Conclusion:** the CSI-limiting errors are large and spatially coherent (drift + front timing), not
neglected small errors at shallow cells. So rebalancing toward shallow cells — by MAE *or* by more
shallow_weight — does not target the binding failure mode; MAE would even soften the penalty on
exactly the large errors that dominate it. Consistent with this, shallow_weight 3 vs 5 runs land in
the same CSI band.

Best current model: `best_sweep_valloss_weightedrmse` — masked RMSE 0.198 m, **CSI@0.05 = 0.674**.

## 2. Feedback: "add outflow ghost cells" → done, plus two important discoveries

Implemented (Option B): 87 outflow ghost cells mirroring the SFINCS msk==3 outlet cells,
free-evolving (no prescribed value), outward edges. Trained on HAL8.

While validating it I found two deeper issues:

**(a) The inflow BC was leaking ground truth.** The ghost-cell dataset accidentally prescribed the
*SFINCS solution's water depth* at the 7 source points instead of the discharge hydrograph
(fallback in the converter: no msk==2 cells exist, since inflow enters via .src point sources).
→ **Fixed**: converter rewritten to feed Q from sfincs.dis (type_BC=2), same convention as the
original datasets. Verified on HAL8.

**(b) Ablation: the model ignores its forcing entirely.** Re-ran the full rollout with the
boundary condition zeroed out:

| | real hydrograph | zero forcing |
|---|---|---|
| CSI@0.05 | 0.4745 | 0.4753 |

Removing the model's only time-varying input changes nothing — with one training simulation, the
model reproduces the memorized event from the warm-start state. It would predict (roughly) the same
flood for any hydrograph.

Also: the outflow ghost cells have one-directional edges (interior → ghost); the GNN's message
passing means the interior never receives information from them — they currently cannot influence
the prediction. A working alternative (absorbing WD=0 boundary with bidirectional edges) is
specified but deferred.

## 3. Feedback: "train longer" → done, with early stopping

Reruns with honest Q forcing: **2000 epochs, early stopping (patience 500), cosine LR** (3 runs on HAL8):

| run | CSI@0.05 | CSI@0.30 | rollout WD loss |
|---|---|---|---|
| 233531 | 0.593 | 0.612 | 0.257 |
| 233342 | 0.567 | 0.611 | 0.297 |
| 233547 | 0.533 | 0.588 | 0.259 |

Same band as all previous single-event runs (0.53–0.67). Removing the ground-truth leak cost
nothing — consistent with the ablation (the forcing wasn't being used anyway).

## Overall conclusion

Across loss weighting, forcing encoding, ghost cells, epochs and LR schedules, **single-event
training plateaus at CSI@0.05 ≈ 0.55–0.67 — a memorization ceiling**, and all current CSI numbers
are on the *training* event, so they partly measure memorization, not skill.

## Proposed next steps (in order)

1. **Generate additional SFINCS runs** on the same Ahr domain with scaled hydrographs
   (e.g. 0.5× / 0.75× / 1.25×). This is the one change that (a) forces the model to actually use the
   discharge input, (b) enables evaluation on a **held-out hydrograph** — the number that supports
   the thesis claim, and (c) directly serves the what-if-scenario end goal.
2. **Cheap parallel test**: increase training rollout_steps beyond 5 (the strongest run on W&B used a
   rollout-10 curriculum) — directly attacks the drift that dominates the CSI decay.
3. Later / optional: soft-CSI (Dice) loss term to suppress scattered false alarms; functional
   outflow boundary (absorbing ghost cells) to help recession drying.

## Figures to show

- `utils/compare_weighted_rmse_runs.ipynb` — CSI-over-time decay, FP/FN decomposition, spatial
  TP/FP/FN maps (rising limb / peak / recession), fringe histograms.
- W&B `mswe-gnn-ahr-finetune`: val_CSI_005 curves (note: train_loss is not comparable across runs
  with different shallow_weight; val metrics are).
- `database/create_dataset_inflow_outflow_gc.ipynb` (Step 4) — final BC configuration map:
  7 Q-inflow source nodes + 87 outflow ghost cells.
