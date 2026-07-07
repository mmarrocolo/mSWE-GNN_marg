# Templates & dataset pkl overview

All pkl files live in `database/datasets/train/` (and `database/datasets/test/` for datasets — train and test copies are identical, saved by the same script).
Updated: 6 Jul 2026.

## Raw inputs (everything under `database/raw_datasets_ahr/`)

| Input | Used for |
|---|---|
| `Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon/` | **Template geometry source** for all `template_100m*` templates: `gis/region.geojson` (boundary polygon), `gis/dep.tif` (DEM), `sfincs_map.nc` (100 m SFINCS grid = finest mesh scale). Also the *cold-start* GT simulation. |
| `Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart/` | **Warmstart simulation** (initialised from `sfincs.20210714.000000.rst`): `sfincs_map.nc` (GT water depths/velocities — re-run 4 Jul for the fixed GT; the old map is kept as `sfincs_map_pre_rerun_20260528.nc`), `sfincs.src` + `sfincs.dis` (7 injection points + discharge hydrographs → BC). |
| *(nothing else)* | `database/raw_datasets_ahr/` now contains **only** the two simulation folders above. All legacy raw data was moved to `database/old_files/` on 6 Jul: `Polygon_1/2.pol` + `DEM_1/2.xyz` + `Hydrograph_1/2.txt` + `Simulations/output_1/2_map.nc` (the numbered quartet consumed by the legacy `create_mesh_dataset` D-Hydro pipeline in `graph_creation.py`), the **10 m original run** `additionalsrc_cutpolygon` (1.5 GB, source of Phase-1 runs #1–3), the non-velocity `additionalsrc_100m_cutpolygon`, and the unreferenced `inlet_points.geojson` (git-moved). The legacy notebook lives at `old_files/create_dataset_template_marg.ipynb`. |

## Templates

| Template | Created by | Inputs | Structure |
|---|---|---|---|
| `template_100m.pkl` (4 Jul, current main) | Step 1 of `database/create_dataset_100m.ipynb` (identical call in `build_template.py`) | region.geojson + dep.tif + non-warmstart `sfincs_map.nc` | SFINCS 100 m grid as finest scale (24,994 faces, exact cell match) + gmsh 500/1000/2000 m; 4 scales, ~30.2k nodes incl. ghost cells. Legacy BC structure (dummy `node_BC`, overridden by the converter with the 7 src nodes). |
| `template_100m_inflow_outflow_gc.pkl` (2 Jul 23:20) | `build_template_inflow_outflow_gc.py` or Step 1 of `database/create_dataset_inflow_outflow_gc.ipynb` (same call), Option B code enabled (commits `added_outflowBC` / `retrying_outflow_bc`) | same as `template_100m.pkl` | Same mesh as `template_100m` plus the **Option B ghost-cell structure**: 87 outflow ghost cells (SFINCS msk==3) with one-way [interior → ghost] edges. Experiments showed the outflow gc are inert (no path back to interior). The `_optionB_backup` copies of this template and its dataset were byte-identical duplicates — deleted 6 Jul. |
| `template_100m_3scales.pkl` (6 Jul, new) | `build_template_3scales.py` | same as `template_100m.pkl` | Same as `template_100m` **minus the coarsest 2000 m scale**: SFINCS 100 m + gmsh 500 + 1000 m; 3 scales, node counts 25,081 / 3,718 / 1,003. |

### Deleted on 6 Jul 2026 (cleanup)

- `template_marg.pkl` (17 Jun) — the original template. Started as 4 pure-gmsh scales from `Polygon_1.pol` + `DEM_1.xyz` (43,526 faces, finest not matching SFINCS cells); the on-disk version had later been rebuilt with the SFINCS-finest pipeline (30,166 nodes, structurally identical to `template_100m.pkl`), making it a redundant older duplicate. No dataset on disk was still built on it.
- `template_100m_warmstart_inflow_outflow_gc.pkl` (2 Jul 11:39) — orphan from an uncommitted morning-of-2-Jul iteration; no script in the repo or git history references it.
- `ahr_river_v03_marg_additionalsrc_velocity_100m_cutpolygon_warmstart_inflow_outflow_gc.pkl` (train + test, 2 Jul 11:41) — dataset built on that orphan template.
- `..._velocity_100m_cutpolygon_warmstart.pkl` (train + test, 4 Jul 20:54) — built by `run_convert_warmstart.py`; verified content-equivalent to `..._100m_warmstart.pkl` (identical wet-cell trajectory 974 → 2474 → 1822), just produced by the CLI converter under the older name. Was only referenced by the outdated `config_finetune_100m_small.yaml` and the cluster scripts `run_sweep_hal8.sh` / `run_finetune_delftblue.sh` (which rebuild it on the cluster if ever needed). Deleted permanently at user request.
- Scripts `run_convert_warmstart.py`, `run_convert_warmstart_94bc.py` (94-node WD-BC experiment — the "BC leak" approach), `patch_pkl_bc.py` (one-off Q-BC repair) — deleted 6 Jul; all superseded, recoverable from git history (`ca8dbbb`).

**Converter fix (6 Jul):** `database/convert_sfincs_to_pkl_marg.py` now fills `zs=NaN → zb` before computing WD (same fix as in `create_dataset_100m.ipynb`), so CLI-built datasets can't get phantom-water smearing if a future SFINCS map writes NaN for dry cells. The Jul-4 rerun map doesn't use NaN, so all Jul-4+ datasets were verified clean; the `..._3scales` dataset was rebuilt with the fixed converter anyway.

## Datasets

All converters interpolate WD/u/v from the simulation's `sfincs_map.nc` onto the template mesh and build the discharge BC (`type_BC=2`, Q at the 7 `sfincs.src` nodes) from `sfincs.src`/`sfincs.dis`.

| Dataset pkl | Template | Created by | Simulation | Used by |
|---|---|---|---|---|
| `..._velocity_100m_warmstart.pkl` (4 Jul) — **main training set** | `template_100m` | Step 3 of `database/create_dataset_100m.ipynb` (includes the NaN-dry-cell fix: `zs=NaN → zb` so dry cells enter interpolation as WD=0) | warmstart (new GT) | `config_best_sweep.yaml` → runs `best_sweep_new_gt`, `best_sweep_new_gt_bestCSI`; most `utils/visualize_*` notebooks |
| `..._velocity_100m_cutpolygon.pkl` (4 Jul) | `template_100m` | same notebook | cold-start | comparison/inference notebooks (`visualize_inference_additionalsrc_100m_results.ipynb`) |
| `..._velocity_100m_warmstart_inflow_outflow_gc.pkl` (2 Jul 23:26) | `template_100m_inflow_outflow_gc` | `run_convert_warmstart_inflow_outflow_gc.py` or Step 3 of `create_dataset_inflow_outflow_gc.ipynb` | warmstart (**pre-rerun GT**) | `config_best_sweep_outflow_gc.yaml` → HAL8 Q-forcing runs #16–18. **Contains the Option B structure** (87 outflow ghost cells in `node_BC_outflow`, found inert) with honest Q forcing (`type_BC=2`, peak 450 m³/s) — despite the notebook note claiming a legacy rebuild, that rebuild never replaced the file on disk. |
| `..._velocity_100m_warmstart_3scales.pkl` (6 Jul, new) | `template_100m_3scales` | `run_convert_warmstart_3scales.py` | warmstart (new GT) | `config_best_sweep_3scales.yaml` (coarse-mesh ablation run) |

## Key differences between the templates, in one paragraph each

- **`template_100m`** — the current standard: the finest scale *is* the SFINCS 100 m grid (no interpolation error at the finest level), plus three gmsh coarsenings at fixed 500/1000/2000 m.
- **`template_100m_inflow_outflow_gc`** — same mesh as `template_100m`; only the ghost-cell/BC structure differs. It exists for the outflow-boundary experiments; the current file on disk is structurally equivalent to `template_100m` (Option B reverted), the Option B variant lives in the `_optionB_backup` file.
- **`template_100m_3scales`** — same as `template_100m` with the 2000 m scale deleted (this week's ablation: does the coarsest mesh matter?). Model config needs `K` of length 5 instead of 7 (`config_best_sweep_3scales.yaml` uses `[1,1,5,3,2]`).

## Gotchas

- Datasets built on different templates are **not interchangeable**: total node counts and `num_scales` differ, and `num_scales` is read from the dataset at run time (the model architecture must match). The 7 src BC nodes land on the same finest-scale indices `[12555, 7287, 15, 368, 5453, 3676, 6427]` in both the 4-scale and 3-scale variants, since the SFINCS scale is unchanged and comes first.
- Everything `*.pkl` is gitignored — on hal8 the templates/datasets are rebuilt by the sbatch script (`run_finetune_hal8.sh` runs the build/convert scripts if the pkl is missing).
- The `inflow_outflow_gc` datasets were built **before** the 4 Jul GT re-run, so their ground truth is older than that of `..._100m_warmstart.pkl`.
