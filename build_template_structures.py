"""Build the 100 m mesh template with weirs/thin dams embedded as constrained
edges in the coarse meshes, using the SFINCS grid as the finest mesh level."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Import create_mesh_template_marg as a bare top-level module (database/ on
# sys.path directly), NOT as database.create_mesh_template_marg - the latter
# runs database/__init__.py first, which does `from .graph_creation import *`
# and loads torch immediately, defeating the lazy-import ordering that
# create_mesh_template_pkl relies on to spawn its gmsh subprocess torch-free.
sys.path.insert(0, os.path.join('.', 'database'))

from create_mesh_template_marg import create_mesh_template_pkl

SFINCS_DIR    = 'database/raw_datasets_ahr/Simulations/ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon'
TEMPLATE_PKL  = 'database/datasets/train/template_100m_structures.pkl'
STRUCTURES_GPKG = os.path.join(SFINCS_DIR, 'gis', 'structures.geojson')

create_mesh_template_pkl(
    shapefile_path        = os.path.join(SFINCS_DIR, 'gis', 'region.geojson'),
    dem_tif_path          = os.path.join(SFINCS_DIR, 'gis', 'dep.tif'),
    output_pkl_path       = TEMPLATE_PKL,
    with_multiscale       = True,
    number_of_multiscales = 4,
    mesh_resolutions      = [2000, 1000, 500],
    sfincs_map_nc         = os.path.join(SFINCS_DIR, 'sfincs_map.nc'),
    structures_gpkg_path  = STRUCTURES_GPKG,
)
print('Done:', TEMPLATE_PKL)
