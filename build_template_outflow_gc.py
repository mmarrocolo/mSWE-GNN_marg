"""Build the 100 m mesh template with OUTFLOW ghost cells (absorbing boundary).

Same inputs as build_template.py; graph_creation auto-detects msk==3 (outflow)
boundary cells and mirrors ghost cells across them. This SFINCS model has no
msk==2 cells, so no inflow ghost cells exist (inflow = discharge at the 7 .src
points, handled by the converter) — hence the plain "outflow" name.

Ghost edges are BIDIRECTIONAL (graph_creation.py, undirected_BC=True, Jul 2026):
required for the absorbing outflow (WD=0 clamp at ghosts in training/train.py).

Output:
  template_100m_outflow_gc.pkl
    data.node_BC_outflow = outflow ghost cells (msk==3), edges both directions
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from database.create_mesh_template_marg import create_mesh_template_pkl

SFINCS_DIR   = 'database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon'
TEMPLATE_PKL = 'database/datasets/train/template_100m_outflow_gc.pkl'

create_mesh_template_pkl(
    shapefile_path        = os.path.join(SFINCS_DIR, 'gis', 'region.geojson'),
    dem_tif_path          = os.path.join(SFINCS_DIR, 'gis', 'dep.tif'),
    output_pkl_path       = TEMPLATE_PKL,
    with_multiscale       = True,
    number_of_multiscales = 4,
    mesh_resolutions      = [2000, 1000, 500],
    sfincs_map_nc         = os.path.join(SFINCS_DIR, 'sfincs_map.nc'),
)
print('Done:', TEMPLATE_PKL)
