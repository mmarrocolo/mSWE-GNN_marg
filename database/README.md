# `database/` — how templates and dataset pkls are made

This folder contains the whole data pipeline of the project. Its end products — and the
**only** thing training ever loads — are the pickle files in:

```
database/datasets/train/<name>.pkl      ← loaded by finetune_ahr.py (training + validation)
database/datasets/test/<name>.pkl       ← loaded for rollout evaluation / notebooks
```

(Train and test copies are written by the same converter run and are identical; the split
exists for the loader's folder convention. All `*.pkl` are gitignored — on hal8 they are
rebuilt by `run_finetune_hal8.sh` if missing.)

A companion document at the repo root, `templates_datasets_overview.md`, tracks **which**
template/dataset files exist, which runs used them, and what was deleted when. This file
explains **how** they are produced.

## The two-step idea

1. **Template** (*"build the world"*): a pkl holding the multiscale mesh (SFINCS 100 m grid
   as finest scale + coarser gmsh meshes), the DEM on every cell, ghost cells, and
   placeholder time series/BC of the right shapes. Geometry only — reusable for any
   simulation on the same grid.
2. **Dataset** (*"pour a flood into the world"*): the template with the placeholders
   replaced by a real SFINCS run — interpolated water depth and velocities per cell per
   hour, plus the discharge boundary condition from `sfincs.src`/`sfincs.dis`
   (`node_BC` = 7 injection cells, `type_BC = 2`).

## File relations

```
LAYER 3 — entry points (you run these)
     notebooks (local, interactive)                scripts (hal8 auto-build / headless)
  create_dataset_100m.ipynb                     ../build_template.py
  create_dataset_inflow_outflow_gc.ipynb        ../build_template_inflow_outflow_gc.py
  create_dataset_3scales.ipynb                  ../build_template_3scales.py
                                                ../run_convert_warmstart_inflow_outflow_gc.py
                                                ../run_convert_warmstart_3scales.py
          │                                            │
          │        (identical function calls)          │
          ▼                                            ▼
LAYER 2 — the two engines (this folder)
  ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
  │ create_mesh_template_marg.py     │   │ convert_sfincs_to_pkl_marg.py    │
  │ "build the world"                │   │ "pour a flood into the world"    │
  │ create_mesh_template_pkl():      │   │ interpolate zs/u/v → mesh,       │
  │  polygon + DEM + SFINCS grid →   │──▶│ Q from src/dis → BC,             │
  │  datasets/train/template_*.pkl   │   │ NaN-dry-cell fix (zs=NaN → zb)   │
  └──────────────┬───────────────────┘   └──────────────┬───────────────────┘
                 │ imports Mesh, MultiscaleMesh,        │
                 │ create_gmesh, add_ghost_cells_mesh,  │
                 ▼ pool_multiscale_attributes, …        │
LAYER 1 — foundation library (this folder)             │
  ┌─────────────────────────────────────────────────┐  │
  │ graph_creation.py  (~2300 lines, from the       │  │
  │ original mSWE-GNN repo + local modifications)   │  │
  │ Mesh / MultiscaleMesh classes, gmsh/triangle    │  │
  │ mesh generators, ghost-cell machinery,          │  │
  │ multiscale pooling & interpolation, plotting    │  │
  │ + the LEGACY end-to-end pipeline                │  │
  │   (create_mesh_dataset, convert_mesh_to_pyg)    │  │
  └─────────────────────────────────────────────────┘  │
                                                       │
INPUTS (raw_datasets_ahr/Simulations/)                 │
  ..._velocity_100m_cutpolygon/                        │
     gis/region.geojson  (boundary polygon)            │
     gis/dep.tif         (DEM)                         │
     sfincs_map.nc       (grid → finest mesh scale;    │
                          also cold-start GT)          │
  ..._velocity_100m_cutpolygon_warmstart/              │
     sfincs_map.nc       (warmstart GT: zs, u, v)      │
     sfincs.src / sfincs.dis (7 sources + hydrographs) │
                                                       ▼
OUTPUTS ══════════════════════════════════════════════════════════════════
  datasets/train/template_100m*.pkl            (geometry only, ~30 MB)
  datasets/train/<dataset>.pkl   ┐  the ACTUAL training data (~67 MB):
  datasets/test/<dataset>.pkl    ┘  mesh + DEM + WD/VX/VY[cells × 119 h] + Q-BC
                                                       │
                                                       ▼
                                     finetune_ahr.py (via utils/dataset.py)
                                     config yaml selects the pkl by
                                     train_dataset_name / test_dataset_name
```

## Current template/dataset pairs (Jul 2026)

| Entry point | Template | Dataset | Purpose |
|---|---|---|---|
| `create_dataset_100m.ipynb` (script twin: `build_template.py`) | `template_100m.pkl` — SFINCS 100 m + gmsh 500/1000/2000 m (4 scales) | `..._velocity_100m_warmstart.pkl` (+ `..._cutpolygon.pkl` cold-start) | main setup — best runs |
| `create_dataset_inflow_outflow_gc.ipynb` (twins: `build_template_inflow_outflow_gc.py` + `run_convert_warmstart_inflow_outflow_gc.py`) | `template_100m_inflow_outflow_gc.pkl` — same + 87 outflow ghost cells (Option B, found inert) | `..._warmstart_inflow_outflow_gc.pkl` | outflow-BC experiments (parked) |
| `create_dataset_3scales.ipynb` (twins: `build_template_3scales.py` + `run_convert_warmstart_3scales.py`) | `template_100m_3scales.pkl` — 2000 m scale deleted (3 scales) | `..._warmstart_3scales.pkl` | coarse-mesh ablation |

Notes:
- `graph_creation.py` vs `create_mesh_template_marg.py`: the former is the generic
  **toolbox** (mesh classes and operations, domain-agnostic) plus the legacy numbered-file
  pipeline (`output_{i}_map.nc` + `DEM_{i}.xyz` + `Hydrograph_{i}.txt` + `Polygon_{i}.pol`,
  archived in `old_files/`); the latter is the Ahr/SFINCS-specific **orchestrator** that
  sequences those tools into a template.
- `convert_sfincs_to_pkl_marg.py` never touches mesh topology; it only fills a template.
  Unpickling any template/dataset requires `graph_creation.py` importable (the `Mesh`
  objects inside are defined there).
- `run_convert_warmstart_inflow_outflow_gc.py` re-implements the interpolation inline
  instead of calling the CLI converter, because it carries the extra `node_BC_outflow`
  field that the generic converter doesn't know about.
- `old_files/` holds everything superseded: the legacy D-Hydro raw family, the 10 m and
  non-velocity 100 m SFINCS runs, `create_dataset_template_marg.ipynb`, etc.
