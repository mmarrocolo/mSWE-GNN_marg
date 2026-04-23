"""
Create a mesh template (.pkl) from a shapefile boundary and DEM.
This is a standalone script that doesn't require D-Hydro simulations.

Usage:
    python create_mesh_template_marg.py <shapefile_path> <dem_tif_path> <output_pkl_path> [--multiscale]
    
Example:
    python create_mesh_template_marg.py C:\\data\\catchment.shp C:\\data\\dem.tif datasets/train/template_marg.pkl --multiscale

The output .pkl is a list containing one torch_geometric.Data object with:
    - Mesh topology (nodes, edges, faces)
    - DEM elevation interpolated to mesh
    - Dummy time series for WD, VX, VY (zeros, will be replaced with SFINCS data)
    - Boundary condition structure
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from pathlib import Path
from copy import copy

from copy import copy as _copy

from graph_creation import (
    create_mesh_dhydro,
    Mesh,
    MultiscaleMesh,
    save_polygon_to_file,
    find_face_BC,
    interpolate_BC_location_multiscale,
    pool_multiscale_attributes,
    update_ghost_cells_attributes,
    add_ghost_cells_mesh,
    add_ghost_cells_attributes
)


def tif_to_xyz(dem_tif_path, output_xyz_path=None):
    """
    Convert GeoTIFF DEM to XYZ format.
    
    Args:
        dem_tif_path: Path to .tif DEM file
        output_xyz_path: Optional path to save .xyz file
    
    Returns:
        output_xyz_path if provided (file path), else xyz array
    """
    try:
        import rasterio
    except ImportError:
        raise RuntimeError(
            "rasterio not found. Install with: pip install rasterio"
        )
    
    if not os.path.exists(dem_tif_path):
        raise FileNotFoundError(f"DEM file not found: {dem_tif_path}")
    
    with rasterio.open(dem_tif_path) as src:
        arr = src.read(1)
        rows, cols = np.where(~np.isnan(arr))
        xs, ys = src.xy(rows, cols)
        zs = arr[rows, cols]
        xyz = np.column_stack([np.array(xs), np.array(ys), zs])
    
    if output_xyz_path:
        np.savetxt(output_xyz_path, xyz, fmt='%.3f %.3f %.5f')
        print(f'  Saved DEM XYZ: {output_xyz_path}')
        return output_xyz_path
    else:
        return xyz


def shapefile_to_polygon_in_dem_crs(shapefile_path, dem_tif_path):
    """
    Convert catchment boundary to polygon and reproject it to DEM CRS.
    
    Args:
        shapefile_path: Path to vector file (.shp/.geojson)
        dem_tif_path: Path to DEM raster (.tif)
    
    Returns:
        geom: shapely Polygon object
    """
    try:
        import fiona
        from shapely.geometry import shape as geom_shape
        from shapely.ops import transform as shp_transform
        import rasterio
        from pyproj import Transformer
    except ImportError:
        raise RuntimeError(
            "Missing geospatial dependencies. Install: fiona shapely rasterio pyproj"
        )
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    with fiona.open(shapefile_path, 'r') as shp:
        feat = shp[0]
        geom = geom_shape(feat['geometry'])
        src_crs = shp.crs_wkt or shp.crs

    if src_crs is None:
        raise RuntimeError(
            "Input polygon has no CRS defined. Please define CRS in the source file before running this script."
        )

    with rasterio.open(dem_tif_path) as dem_src:
        dem_crs = dem_src.crs

    if dem_crs is None:
        raise RuntimeError("DEM raster has no CRS defined.")
    
    # Handle MultiPolygon, select largest
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda g: g.area)

    transformer = Transformer.from_crs(src_crs, dem_crs, always_xy=True)
    geom = shp_transform(transformer.transform, geom)
    
    return geom


def create_mesh_template_pkl(
    shapefile_path,
    dem_tif_path,
    output_pkl_path,
    with_multiscale=False,
    number_of_multiscales=4,
    simplify_tolerance=0,
    n_timesteps=10,
    sfincs_map_nc=None,
):
    """
    Create a mesh template pickle file from shapefile + DEM.

    Args:
        shapefile_path: Path to catchment boundary .shp
        dem_tif_path: Path to DEM .tif
        output_pkl_path: Output path for .pkl file
        with_multiscale: If True, create multiscale mesh (required for MSGNN)
        number_of_multiscales: Number of refinement levels
        simplify_tolerance: Simplify boundary to reduce nodes (0 = no simplification)
        n_timesteps: Number of time steps for dummy data (will be replaced by SFINCS converter)
        sfincs_map_nc: Optional path to a SFINCS sfincs_map.nc file. When provided the
            SFINCS structured grid is used as the finest mesh level and only
            number_of_multiscales-1 coarser meshkernel meshes are created.
    """
    
    print(f"\n=== Creating Mesh Template ===")
    print(f"Shapefile: {shapefile_path}")
    print(f"DEM: {dem_tif_path}")
    print(f"Output: {output_pkl_path}")
    print(f"Multiscale: {with_multiscale}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl_path)), exist_ok=True)
    
    # Step 1: Convert shapefile to polygon
    print("\n1. Converting shapefile to polygon...")
    geom = shapefile_to_polygon_in_dem_crs(shapefile_path, dem_tif_path)
    simplify_tol = simplify_tolerance
    if simplify_tol > 0:
        geom = geom.simplify(tolerance=simplify_tol, preserve_topology=True)
    print(f"   Polygon area: {geom.area:.2f} m²")
    print(f"   Boundary length: {geom.length:.2f} m")

    try:
        import rasterio
        with rasterio.open(dem_tif_path) as dem_src:
            dem_bounds = dem_src.bounds
            print(f"   Polygon bounds: {geom.bounds}")
            print(f"   DEM bounds: ({dem_bounds.left}, {dem_bounds.bottom}, {dem_bounds.right}, {dem_bounds.top})")
    except Exception:
        pass
    
    # Step 2: Save polygon to temporary .pol file
    print("\n2. Creating polygon file (.pol)...")
    temp_dir = os.path.dirname(os.path.abspath(output_pkl_path))
    polygon_file = os.path.join(temp_dir, 'temp_boundary.pol')
    save_polygon_to_file(geom, polygon_file)
    
    # Step 3: Convert DEM .tif to .xyz (or use existing .xyz)
    print("\n3. Processing DEM...")
    dem_path = dem_tif_path
    dem_xyz_path = None
    if dem_tif_path.lower().endswith('.tif'):
        dem_xyz_path = os.path.join(temp_dir, 'temp_dem.xyz')
        dem_path = tif_to_xyz(dem_tif_path, dem_xyz_path)  # Returns file path if output_xyz_path is provided
    # else dem_path is already the .xyz file path
    
    # Load and check DEM
    try:
        dem_data = np.loadtxt(dem_path)
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "The provided DEM file is not readable as plain text XYZ. "
            "SFINCS .dep files are often grid/binary-style files and cannot be read directly by this script. "
            "Use a .tif DEM/DTM or an .xyz file with 3 columns: x y z."
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Could not parse DEM file. Expected .tif or .xyz with 3 columns: x y z."
        ) from e

    if dem_data.ndim != 2 or dem_data.shape[1] < 3:
        raise RuntimeError(
            "DEM file does not have the expected XYZ structure. "
            "Expected at least 3 columns: x y z."
        )

    print(f"   DEM points: {dem_data.shape[0]}")
    print(f"   Elevation range: {dem_data[:, 2].min():.2f} to {dem_data[:, 2].max():.2f} m")
    
    # Step 4: Create mesh from polygon (+ optional SFINCS finest level)
    print("\n4. Creating mesh from polygon...")
    n_mk_scales = number_of_multiscales - 1 if sfincs_map_nc else number_of_multiscales
    mesh_list = create_mesh_dhydro(polygon_file, n_mk_scales, for_simulation=False)

    if sfincs_map_nc:
        if not os.path.exists(sfincs_map_nc):
            raise FileNotFoundError(f"SFINCS map file not found: {sfincs_map_nc}")
        print(f"   Loading SFINCS mesh from: {sfincs_map_nc}")
        sfincs_mesh = Mesh()
        sfincs_mesh._import_from_sfincs_map(sfincs_map_nc)
        mesh_list.append(sfincs_mesh)
        print(f"   SFINCS mesh faces: {sfincs_mesh.face_x.shape[0]}")

    finest_mesh = mesh_list[-1]
    print(f"   Finest mesh nodes: {finest_mesh.node_x.shape[0]}")
    print(f"   Finest mesh faces: {finest_mesh.face_x.shape[0]}")
    print(f"   Finest mesh edges: {finest_mesh.edge_index.shape[1]}")

    # Step 5: Import DEM into finest mesh
    print("\n5. Interpolating DEM to mesh...")
    finest_mesh._import_DEM(dem_path)
    print(f"   Interpolated DEM range (finest): {finest_mesh.DEM.min():.2f} to {finest_mesh.DEM.max():.2f} m")

    # Initialize BC edge/face attributes.
    # For a SFINCS mesh these are already set by _import_from_sfincs_map;
    # for a pure meshkernel mesh we derive them from boundary edges.
    if sfincs_map_nc:
        if finest_mesh.face_BC.size == 0:
            raise RuntimeError('SFINCS mesh has no open-boundary cells (msk==3). '
                               'Check the sfincs_map.nc mask.')
    else:
        if not hasattr(finest_mesh, 'boundary_edges') or len(finest_mesh.boundary_edges) == 0:
            raise RuntimeError('Could not infer boundary edges to initialize BC attributes.')

        finest_mesh.edge_index_BC = np.asarray(finest_mesh.boundary_edges, dtype=int)
        edge_bc_idx = []
        for edge in finest_mesh.edge_index_BC:
            idx = np.where(
                ((edge == finest_mesh.edge_index.T).sum(1) == 2)
                | ((edge[::-1] == finest_mesh.edge_index.T).sum(1) == 2)
            )[0]
            if len(idx) > 0:
                edge_bc_idx.append(idx[0])

        finest_mesh.edge_BC = np.asarray(edge_bc_idx, dtype=int)
        if hasattr(finest_mesh, 'edge_type') and finest_mesh.edge_BC.size > 0:
            finest_mesh.edge_type[finest_mesh.edge_BC] = 2

        finest_mesh.face_BC = find_face_BC(finest_mesh)
        if finest_mesh.face_BC.size == 0:
            raise RuntimeError('Could not infer boundary face for BC initialization.')

    # Step 6: Build single-scale or multiscale mesh with ghost cells
    print("\n6. Adding ghost cells for boundary conditions...")
    if with_multiscale:
        if sfincs_map_nc:
            edge_BC_mid = finest_mesh.face_xy[finest_mesh.face_BC].mean(0, keepdims=True)
        else:
            edge_BC_mid = finest_mesh.node_xy[finest_mesh.edge_index_BC].mean(1)
        mesh_list = interpolate_BC_location_multiscale(mesh_list, edge_BC_mid)
        mesh_list = [add_ghost_cells_mesh(m) for m in mesh_list]

        multiscale_mesh = MultiscaleMesh()
        multiscale_mesh.stack_meshes(mesh_list)
        mesh = multiscale_mesh

        # Create dummy attributes at finest scale and pool to all scales.
        finest_mesh_with_ghost = mesh_list[-1]
        n_faces_fine = finest_mesh_with_ghost.face_x.shape[0]
        WD_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        VX_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        VY_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        DEM_fine = finest_mesh_with_ghost.DEM.copy()

        DEM_fine, WD_fine, VX_fine, VY_fine = add_ghost_cells_attributes(
            finest_mesh_with_ghost, DEM_fine, WD_fine, VX_fine, VY_fine
        )

        DEM, WD, VX, VY = pool_multiscale_attributes(mesh, DEM_fine, WD_fine, VX_fine, VY_fine, reduce='mean')
        DEM = update_ghost_cells_attributes(mesh, DEM)[0]
        print(f"   Multiscale meshes: {mesh.num_meshes}")
        print(f"   Boundary nodes (ghost cells): {len(mesh.ghost_cells_ids)}")
    else:
        mesh = add_ghost_cells_mesh(finest_mesh)
        print(f"   Boundary nodes (ghost cells): {len(mesh.ghost_cells_ids)}")

        n_faces = mesh.face_x.shape[0]
        WD = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        VX = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        VY = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        DEM = mesh.DEM.copy()

        DEM, WD, VX, VY = add_ghost_cells_attributes(mesh, DEM, WD, VX, VY)

    # Step 7: Dummy time series summary
    print("\n7. Creating dummy time series...")
    print(f"   Shapes: WD={WD.shape}, VX={VX.shape}, VY={VY.shape}")
    
    # Step 8: Create boundary condition structure (dummy)
    print("\n8. Creating boundary condition structure...")
    n_bc_nodes = len(mesh.node_BC) if hasattr(mesh, 'node_BC') else len(mesh.ghost_cells_ids)
    BC_dummy = np.zeros((n_timesteps, 2), dtype=np.float32)  # [T, 2] → will be replicated per BC node
    print(f"   BC nodes: {n_bc_nodes}")
    print(f"   BC shape will be: [{n_bc_nodes}, {n_timesteps}, 2]")
    
    # Step 9: Create torch_geometric Data object
    print("\n9. Creating torch_geometric Data object...")
    data = Data()
    
    # Properties
    data.DEM = torch.FloatTensor(DEM)
    data.WD = torch.FloatTensor(WD)
    data.VX = torch.FloatTensor(VX)
    data.VY = torch.FloatTensor(VY)
    data.area = torch.FloatTensor(mesh.face_area)
    
    # Graph structure
    data.edge_index = torch.LongTensor(mesh.dual_edge_index)
    data.face_distance = torch.FloatTensor(mesh.dual_edge_length)
    data.face_relative_distance = torch.FloatTensor(mesh.face_relative_distance)
    data.edge_slope = (data.DEM[data.edge_index][0] - data.DEM[data.edge_index][1]) / (data.face_distance + 1e-6)
    
    data.num_nodes = mesh.face_x.shape[0]
    
    # Boundary conditions
    data.node_BC = torch.IntTensor(mesh.ghost_cells_ids)
    data.edge_BC_length = torch.FloatTensor(mesh.edge_length[mesh.edge_BC])
    if with_multiscale:
        data.node_ptr = torch.LongTensor(mesh.face_ptr)
        data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr)
        data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr)
        data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)

        # Keep BC at finest scale only, as expected by training/inference pipeline.
        n_bc_finest = len(mesh.meshes[-1].ghost_cells_ids)
        data.node_BC = data.node_BC[-n_bc_finest:]
        data.edge_BC_length = data.edge_BC_length[-n_bc_finest:]

    data.BC = torch.FloatTensor(BC_dummy).unsqueeze(0).repeat(len(data.node_BC), 1, 1)
    data.type_BC = torch.tensor(2, dtype=torch.int)  # 2 = discharge
    
    # Store mesh object for reference
    data.mesh = mesh
    
    print(f"   Data object keys: {list(data.keys())}")
    print(f"   Edge index shape: {data.edge_index.shape}")
    print(f"   BC shape: {data.BC.shape}")
    
    # Step 10: Save as pickle
    print("\n10. Saving to pickle...")
    dataset_list = [data]  # Load them back as list
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(dataset_list, f)
    print(f"   ✓ Saved: {output_pkl_path}")
    print(f"   File size: {os.path.getsize(output_pkl_path) / (1024**2):.2f} MB")
    
    # Step 11: Cleanup temporary files
    print("\n11. Cleaning up temporary files...")
    if os.path.exists(polygon_file):
        os.remove(polygon_file)
    if dem_xyz_path is not None and os.path.exists(dem_xyz_path):
        os.remove(dem_xyz_path)
    
    print("\n=== Template Created Successfully ===\n")
    
    return output_pkl_path


def create_mesh_template_from_pol(pol_path, xyz_path, output_pkl_path,
                                   with_multiscale=False, number_of_multiscales=4,
                                   n_timesteps=10, sfincs_map_nc=None):
    """Create a mesh template directly from an existing .pol boundary and .xyz DEM,
    bypassing the shapefile conversion step.

    Args:
        sfincs_map_nc: Optional path to a SFINCS sfincs_map.nc file. When provided the
            SFINCS structured grid is used as the finest mesh level and only
            number_of_multiscales-1 coarser meshkernel meshes are created.
    """
    import pickle

    print(f"\n=== Creating Mesh Template (from .pol + .xyz) ===")
    print(f"Polygon: {pol_path}")
    print(f"DEM:     {xyz_path}")
    print(f"Output:  {output_pkl_path}")
    print(f"Multiscale: {with_multiscale}")
    if sfincs_map_nc:
        print(f"SFINCS map: {sfincs_map_nc}")

    os.makedirs(os.path.dirname(os.path.abspath(output_pkl_path)), exist_ok=True)

    dem_data = np.loadtxt(xyz_path)
    print(f"   DEM points: {dem_data.shape[0]}")
    print(f"   Elevation range: {dem_data[:, 2].min():.2f} to {dem_data[:, 2].max():.2f} m")

    print("\n4. Creating mesh from polygon...")
    n_mk_scales = number_of_multiscales - 1 if sfincs_map_nc else number_of_multiscales
    mesh_list = create_mesh_dhydro(pol_path, n_mk_scales, for_simulation=False)

    if sfincs_map_nc:
        if not os.path.exists(sfincs_map_nc):
            raise FileNotFoundError(f"SFINCS map file not found: {sfincs_map_nc}")
        print(f"   Loading SFINCS mesh from: {sfincs_map_nc}")
        sfincs_mesh = Mesh()
        sfincs_mesh._import_from_sfincs_map(sfincs_map_nc)
        mesh_list.append(sfincs_mesh)
        print(f"   SFINCS mesh faces: {sfincs_mesh.face_x.shape[0]}")

    finest_mesh = mesh_list[-1]
    print(f"   Finest mesh faces: {finest_mesh.face_x.shape[0]}")

    print("\n5. Interpolating DEM to mesh...")
    finest_mesh._import_DEM(xyz_path)

    # Initialize BC attributes for finest mesh.
    # SFINCS mesh: already set by _import_from_sfincs_map.
    # Meshkernel mesh: derive from boundary edges.
    if not sfincs_map_nc:
        finest_mesh.edge_index_BC = np.asarray(finest_mesh.boundary_edges, dtype=int)
        edge_bc_idx = []
        for edge in finest_mesh.edge_index_BC:
            idx = np.where(
                ((edge == finest_mesh.edge_index.T).sum(1) == 2)
                | ((edge[::-1] == finest_mesh.edge_index.T).sum(1) == 2)
            )[0]
            if len(idx) > 0:
                edge_bc_idx.append(idx[0])
        finest_mesh.edge_BC = np.asarray(edge_bc_idx, dtype=int)
        if hasattr(finest_mesh, 'edge_type') and finest_mesh.edge_BC.size > 0:
            finest_mesh.edge_type[finest_mesh.edge_BC] = 2
        finest_mesh.face_BC = find_face_BC(finest_mesh)

    print("\n6. Adding ghost cells for boundary conditions...")
    if with_multiscale:
        if sfincs_map_nc:
            # SFINCS BC cells (msk==3) lie at the SFINCS grid boundary, which is in the
            # interior of the meshkernel polygon. Seed the coarser-mesh BC search from the
            # finest meshkernel mesh's polygon boundary edges instead.
            mk_finest = mesh_list[-2]
            bnd_mids = mk_finest.node_xy[mk_finest.boundary_edges].mean(1)
            valid = np.isfinite(bnd_mids).all(axis=1)
            bnd_mids = bnd_mids[valid]
            centroid = mk_finest.face_xy.mean(0)
            dists = np.linalg.norm(bnd_mids - centroid, axis=1)
            best = int(dists.argmax())
            edge_BC_mid = bnd_mids[best:best+1]
            interpolate_BC_location_multiscale(mesh_list[:-1], edge_BC_mid)
        else:
            all_bc_mids = finest_mesh.node_xy[finest_mesh.edge_index_BC].mean(1)
            centroid = finest_mesh.node_xy.mean(0)
            dists = np.linalg.norm(all_bc_mids - centroid, axis=1)
            best = int(dists.argmax())
            edge_BC_mid = all_bc_mids[best:best+1]
            mesh_list = interpolate_BC_location_multiscale(mesh_list, edge_BC_mid)
        mesh_list = [add_ghost_cells_mesh(m) for m in mesh_list]

        multiscale_mesh = MultiscaleMesh()
        multiscale_mesh.stack_meshes(mesh_list)
        mesh = multiscale_mesh

        finest_mesh_with_ghost = mesh_list[-1]
        n_faces_fine = finest_mesh_with_ghost.face_x.shape[0]
        WD_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        VX_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        VY_fine = np.zeros((n_faces_fine, n_timesteps), dtype=np.float32)
        DEM_fine = finest_mesh_with_ghost.DEM.copy()

        DEM_fine, WD_fine, VX_fine, VY_fine = add_ghost_cells_attributes(
            finest_mesh_with_ghost, DEM_fine, WD_fine, VX_fine, VY_fine
        )
        DEM, WD, VX, VY = pool_multiscale_attributes(mesh, DEM_fine, WD_fine, VX_fine, VY_fine, reduce='mean')
        DEM = update_ghost_cells_attributes(mesh, DEM)[0]
        print(f"   Multiscale meshes: {mesh.num_meshes}")
    else:
        mesh = add_ghost_cells_mesh(finest_mesh)
        n_faces = mesh.face_x.shape[0]
        WD = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        VX = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        VY = np.zeros((n_faces, n_timesteps), dtype=np.float32)
        DEM = mesh.DEM.copy()
        DEM, WD, VX, VY = add_ghost_cells_attributes(mesh, DEM, WD, VX, VY)

    data = Data()
    data.DEM = torch.FloatTensor(DEM)
    data.WD = torch.FloatTensor(WD)
    data.VX = torch.FloatTensor(VX)
    data.VY = torch.FloatTensor(VY)
    data.area = torch.FloatTensor(mesh.face_area)
    data.edge_index = torch.LongTensor(mesh.dual_edge_index)
    data.face_distance = torch.FloatTensor(mesh.dual_edge_length)
    data.face_relative_distance = torch.FloatTensor(mesh.face_relative_distance)
    data.edge_slope = (data.DEM[data.edge_index][0] - data.DEM[data.edge_index][1]) / (data.face_distance + 1e-6)
    data.num_nodes = mesh.face_x.shape[0]
    data.node_BC = torch.IntTensor(mesh.ghost_cells_ids)
    data.edge_BC_length = torch.FloatTensor(mesh.edge_length[mesh.edge_BC])

    if with_multiscale:
        data.node_ptr = torch.LongTensor(mesh.face_ptr)
        data.edge_ptr = torch.LongTensor(mesh.dual_edge_ptr)
        data.intra_edge_ptr = torch.LongTensor(mesh.intra_edge_ptr)
        data.intra_mesh_edge_index = torch.LongTensor(mesh.intra_mesh_dual_edge_index)
        n_bc_finest = len(mesh.meshes[-1].ghost_cells_ids)
        data.node_BC = data.node_BC[-n_bc_finest:]
        data.edge_BC_length = data.edge_BC_length[-n_bc_finest:]

    BC_dummy = np.zeros((n_timesteps, 2), dtype=np.float32)
    data.BC = torch.FloatTensor(BC_dummy).unsqueeze(0).repeat(len(data.node_BC), 1, 1)
    data.type_BC = torch.tensor(2, dtype=torch.int)
    data.mesh = mesh

    with open(output_pkl_path, 'wb') as f:
        pickle.dump([data], f)
    print(f"\n=== Template saved: {output_pkl_path} ===")
    print(f"   WD shape: {tuple(data.WD.shape)}")
    return output_pkl_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a mesh template (.pkl) from shapefile + DEM"
    )
    parser.add_argument(
        'shapefile', nargs='?', default=None,
        help='Path to catchment boundary shapefile (.shp) — omit if using --pol'
    )
    parser.add_argument(
        'dem', nargs='?', default=None,
        help='Path to DEM file (.tif or .xyz) — omit if using --pol'
    )
    parser.add_argument(
        'output', nargs='?', default=None,
        help='Output path for template pickle file (.pkl)'
    )
    parser.add_argument(
        '--pol',
        default=None,
        help='Use existing .pol boundary file directly (skips shapefile conversion)'
    )
    parser.add_argument(
        '--xyz',
        default=None,
        help='Use existing .xyz DEM file directly (required with --pol)'
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output .pkl path (alternative to positional argument)'
    )
    parser.add_argument(
        '--multiscale',
        action='store_true',
        help='Create multiscale mesh (required for MSGNN checkpoint)'
    )
    parser.add_argument(
        '--num-scales',
        type=int,
        default=4,
        help='Number of multiscale refinement levels (default: 4)'
    )
    parser.add_argument(
        '--simplify',
        type=float,
        default=0,
        help='Simplify boundary to reduce nodes (tolerance in meters, 0=no simplification)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10,
        help='Number of dummy time steps (will be replaced by SFINCS converter)'
    )

    args = parser.parse_args()

    output = args.out or args.output

    # --pol / --xyz shortcut: skip shapefile conversion
    if args.pol is not None:
        if args.xyz is None:
            print("Error: --xyz is required when using --pol")
            return 1
        if output is None:
            print("Error: output path required (positional arg or --out)")
            return 1
        for p, label in [(args.pol, '--pol'), (args.xyz, '--xyz')]:
            if not os.path.exists(p):
                print(f"Error: {label} file not found: {p}")
                return 1
        try:
            create_mesh_template_from_pol(
                pol_path=args.pol,
                xyz_path=args.xyz,
                output_pkl_path=output,
                with_multiscale=args.multiscale,
                number_of_multiscales=args.num_scales,
                n_timesteps=args.timesteps,
            )
            return 0
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Original shapefile path
    if args.shapefile is None or args.dem is None or output is None:
        parser.print_help()
        return 1

    if not os.path.exists(args.shapefile):
        print(f"Error: Shapefile not found: {args.shapefile}")
        return 1

    if not os.path.exists(args.dem):
        print(f"Error: DEM file not found: {args.dem}")
        return 1

    try:
        create_mesh_template_pkl(
            shapefile_path=args.shapefile,
            dem_tif_path=args.dem,
            output_pkl_path=output,
            with_multiscale=args.multiscale,
            number_of_multiscales=args.num_scales,
            simplify_tolerance=args.simplify,
            n_timesteps=args.timesteps,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
