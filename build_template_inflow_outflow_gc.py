"""Build the 100 m mesh template with Option B ghost cells (inflow + outflow).

Same inputs as build_template.py; the new graph_creation code auto-detects msk==2
(inflow) and msk==3 (outflow) cells and creates both types of ghost cells.

Output:
  template_100m_inflow_outflow_gc.pkl
    data.node_BC         = inflow ghost cells (msk==2)   [ghost -> interior]
    data.node_BC_inflow  = same as node_BC
    data.node_BC_outflow = outflow ghost cells (msk==3)  [interior -> ghost]
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from database.create_mesh_template_marg import create_mesh_template_pkl

SFINCS_DIR   = 'database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon'
TEMPLATE_PKL = 'database/datasets/train/template_100m_inflow_outflow_gc.pkl'

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
