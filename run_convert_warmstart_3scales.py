"""Wrapper: convert the warmstart (new GT) simulation onto the 3-scale template."""
import sys
import os

# Ensure project root is on the path so pickled database.* objects can be found
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SIM_DIR = os.path.join(
    PROJECT_ROOT,
    "database", "raw_datasets_ahr", "Simulations",
    "ahr_river_v03_Marg_additionalsrc_velocity_100m_cutpolygon_warmstart",
)

sys.argv = [
    "convert_sfincs_to_pkl_marg.py",
    "--sfincs-map",    os.path.join(SIM_DIR, "sfincs_map.nc"),
    "--template-pkl",  os.path.join(PROJECT_ROOT, "database", "datasets", "train", "template_100m_3scales.pkl"),
    "--dataset-name",  "ahr_river_v03_marg_additionalsrc_velocity_100m_warmstart_3scales",
    "--out-root",      os.path.join(PROJECT_ROOT, "database", "datasets"),
    "--vx-var", "u",
    "--vy-var", "v",
    "--src-file", os.path.join(SIM_DIR, "sfincs.src"),
    "--dis-file", os.path.join(SIM_DIR, "sfincs.dis"),
]

# Import and run the conversion
import importlib.util
spec = importlib.util.spec_from_file_location(
    "convert_sfincs_to_pkl_marg",
    os.path.join(PROJECT_ROOT, "database", "convert_sfincs_to_pkl_marg.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()
