# Thesis appendix plan — verification (A) vs characterization (B)

Working scaffold. Run numbers (#) and T/E labels refer to `thesis_runs_overview.md`;
figures live in `results/thesis_overview/`.

**Sorting rule used:** Appendix A answers right/wrong questions (something was *incorrect*
and was fixed — if the alternative had won, it would have been a bug). Appendix B answers
better/worse questions (both options were valid; a choice was measured and made).

**Main-text flag:** entries marked ★ carry a finding that shapes the thesis narrative —
they get a subchapter in the main body, with the appendix holding the extensive version.

---

# Appendix A — Implementation verification and fixes

**Entry format (use for every A-entry):**

> ### A.n — Title
> - **Status:** fixed on <date>, commit `<hash>` / file `<path>`
> - **Symptom:** what looked wrong (metric, figure, or impossibility that triggered the investigation)
> - **Diagnosis:** root cause, and how it was isolated (the experiment that proved it)
> - **Fix:** what was changed, exactly where (file/function/config)
> - **Verification:** before/after evidence — same metric, same protocol
> - **Downstream impact:** which earlier runs/conclusions were invalidated or re-run

### A.1 ★ Cold start vs warm start (model cannot create water)
- Evidence: runs #1–3, comparison T1. CSI = 0.000 at every threshold from a dry domain;
  0.408 with warm start, same checkpoint, same event.
- Notebooks: `visualize_inference_additionalsrc_100m_results.ipynb` (+ original-resolution twin).
- Figures: `visualize_inference_additionalsrc_results__c8_0.png`, `..._100m_results__c24_0.png`.
- ★ main text: the finding (autoregressive GNN propagates/amplifies water but cannot
  initiate flooding from point-source forcing) motivates the entire warmstart methodology.
  Appendix holds the full tables and both resolutions.

### A.2 — Multiscale scale-ordering fix (inverted meshes)
- Evidence: runs #4 vs #5, comparison T3. CSI@0.05 0.589 → 0.716 after making scale 0 =
  the SFINCS mesh, so BC nodes / loss / evaluation index the correct level.
- Fix location: `invert_scale_ordering` (`database/graph_creation.py`); templates rebuilt
  finest-first since.
- Figures: `visualize_finetuned_ahr_100m__c12_0.png` vs `visualize_finetuned_invertedmeshes__c12_0.png`.

### A.3 ★ Warm-start ground-truth quality (spinup length, old vs new GT)
- Evidence: comparison T2 (confounded — retrained): #12 (old GT, CSI@0.05 0.674, FN/wet 0.17)
  vs #20 (new GT, 0.741, FN/wet 0.05). Wet cells at t=0: 780 → 974 (river fully wetted).
- Diagnosis: the "flood-front lag" in the downstream ~7 km was an initialization artifact,
  not model drift — SFINCS rerun with 6-day spinup on 4 Jul.
- Planned clean split of data-vs-training contribution: cross-evaluation E1.
- ★ main text: warm-start *quality* matters as much as its presence (cross-cutting finding 1).

### A.4 — Boundary-condition leak and the honest-Q fix
- Symptom chain, three sub-findings (runs #14–18, comparisons T7 + T8-part-1):
  1. gc dataset fed SFINCS's own water depth at the 7 source cells (`type_BC=1` fallback)
     — ground truth leaking into the input;
  2. **zero-forcing ablation (the pre-fix hydrograph investigation): CSI 0.4745 (real Q)
     vs 0.4753 (Q≡0)** — the model ignored its only time-varying input entirely; this is
     the experiment that justified imposing the Q inflow;
  3. fix: discharge forcing `type_BC=2`, Q peak 450 m³/s (converters rebuilt Jul 2);
     honest forcing cost nothing (T7: 0.475 → 0.533–0.593).
- Fix location: `run_convert_warmstart_inflow_outflow_gc.py` / `build_output_data` path.

### A.5 — Outflow ghost cells: implemented, verified inert (Option B)
- Intent: let water leave the domain at open boundaries (SFINCS msk==3) to counter
  recession over-wetting; 87 ghost cells with one-way [interior → ghost] edges.
- Verification: no influence on interior possible by construction (no return edges);
  runs #14–18 confirm no benefit. Feature parked; functional alternative (absorbing
  WD=0, bidirectional edges) deferred to E6b.
- Data note: `template_100m_inflow_outflow_gc.pkl` retains the Option B structure.

### A.6 — Interpolation of dry cells (phantom-water / NaN fix)
- Symptom: ~15k fake wet cells (vs ~1–2.5k real) when SFINCS writes zs=NaN for dry cells
  and the converter interpolates between wet cells only.
- Fix: fill zs=NaN → zb before computing WD (`create_dataset_100m.ipynb` Step 3, 4 Jul;
  `database/convert_sfincs_to_pkl_marg.py` line ~274, 6 Jul).
- Verification: wet-cell trajectory sanity check now standard in every conversion
  (974 → 2474 @t=77 → 1822 for the new GT); 3-scale dataset verified bit-identical to
  4-scale at the finest level.

---

# Appendix B — Trials defining the network and training configuration

**Entry format (use for every B-entry):**

> ### B.n — Title
> - **Question:** the design choice being decided
> - **Setup:** what varied / what was held fixed (config, dataset, seed);
>   **clean or confounded** (if confounded: with what, and pointer to the planned clean rerun)
> - **Results:** table (RMSE WD / CSI@0.05 / CSI@0.30, full-rollout protocol)
> - **Decision:** option adopted, and where it lives now (config field / code)
> - **Caveats:** seed variance not yet quantified (E3), single-event scope, etc.

### B.1 — Finetuning from the pretrained dijkring checkpoint vs training from scratch
- Evidence: runs #3–4 (finetuned from `finetuned_dk15`) vs later from-scratch runs.
  **Confounded** (era changes: loss, GT, architecture) — flag for a clean rerun if it
  becomes a thesis claim; otherwise present as development history.
- Decision: training from scratch under the sweep config (`saved_model: ''`).

### B.2 — Mass-conservation penalty: on (0.01) vs off
- Evidence: T5 — direct A/B lives in W&B only; local #4 vs #9 is confounded with loss weighting.
- Decision: off (`conservation: 0`). Clean single-variable rerun planned (E2b).

### B.3 — Weighted vs unweighted RMSE (shallow_weight)
- Evidence: T6 — #4 → #9 RMSE 0.470 → 0.262, confounded with dropping conservation;
  sweep explored w ∈ {3,5} systematically (W&B).
- Decision: weighted, w=5, threshold 0.30 m (`shallow_weight`, `shallow_threshold`).
- Clean rerun planned (E2a); MAE-vs-RMSE variant (supervisor question) = E2c.

### B.4 — Loss on all scales vs scale 0 only
- Evidence: T4 (clean at the time): #4 vs #6 — CSI@0.05 0.589 → 0.626 at equal RMSE.
- Rationale: coarse scales are auxiliary message-passing levels, their WD values are not used.
- Decision: scale-0 loss (`training/loss.py`, `get_multiscale_loss` slices `node_ptr[0]:node_ptr[1]`).

### B.5 — Hyperparameter sweep (hid32 vs hid64, lr, rollout, batch)
- Evidence: runs #11–13; best sweep config ffphz2b4 (hid32, 193k params, lr 5.18e-4,
  rollout 5, batch 4) beat hid64 (795k). Supervisor-endorsed (Jun 27).
- Decision: `config_best_sweep.yaml` frozen as the reference config.

### B.6 — Mesh hierarchy: 4 scales vs 3 scales (coarsest 2000 m deleted)
- Setup: **clean by construction** — dataset verified identical at the finest scale
  (`create_dataset_3scales.ipynb` Step 4: allclose True on all 25,081 shared faces, same BC);
  only mesh hierarchy and K change (`[1,1,1,5,4,3,2]` → `[1,1,5,3,2]`, note the K-adaptation
  caveat: bottom receptive field halves from ~10 km to ~5 km per hop-block).
- Status: run pending on hal8 (`config_best_sweep_3scales.yaml` → `results/best_sweep_3scales.h5`).
- Results: *(fill in after the run)* vs #19/#20 as the 4-scale reference.

### B.7 — Checkpoint selection: best val_loss vs best val_CSI vs last epoch
- Evidence: T9 (clean, same run): #19/#20/#21 — CSI@0.05 0.680 / 0.741 / 0.629.
- Decision: dual checkpointing in `finetune_ahr.py`; select on the target metric
  (+0.06 CSI at negligible RMSE cost); never use the last epoch (#10, #13, #21 all drift).

### B.8 ★ — Forcing response after the Q fix (hydrograph scaling sensitivity)
- Evidence: T8-part-2 — BC × {0, 0.5, 1, 1.5, 2} on checkpoint #20: flood collapses at 0×
  but is essentially identical for any non-zero scaling (CSI ~0.74 at 0.5×–1.5×).
- Finding: forcing is used as a **binary switch, not a magnitude signal** — presence
  learned, amplitude not, exactly as expected from single-event training.
- ★ main text: this is the justification for multi-hydrograph training data and held-out
  evaluation (E4) — the thesis-critical experiment design. Appendix holds the full table,
  WD-sum diagnostics, and the persistence baseline. Should become a standard evaluation
  for every future model.
- Source: "Forcing sensitivity (T8)" section of `utils/visualize_finetuned_fixed_gt_warmstart.ipynb`.

### B.9 — (planned) Drift levers: rollout curriculum length, input noise
- E5: rollout 10 vs 5; Gaussian noise σ ≈ 0.5–1 cm on input depths. Placeholder — fill when run.

### B.10 — (planned) FP suppression: soft-CSI/Dice term, absorbing outflow BC
- E6: targets the recession over-wetting that dominates the current error. Placeholder.

---

## Notes for writing

- Every B-table should eventually carry seed-variance error bars (E3) — until then, add
  the caveat line "±0.03 CSI differences not interpretable without seed variance" wherever relevant.
- All metrics are on the training event (reproduction, not generalization) until E4 exists —
  state this once per appendix intro rather than per entry.
- The pre-fix ablation (A.4.2) and post-fix scaling test (B.8) use the same experimental
  tool; A.4 should end with a forward reference: "the same test applied after the fix
  reveals a data limitation rather than an implementation flaw → B.8 / main text".
