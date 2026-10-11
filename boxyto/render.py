"""PyVista surface extraction and reproducible publication rendering.

Density arrays always use (y, x, z); meshes and boundary points use physical
(x, y, z). PyVista is imported lazily so headless optimization does not need VTK.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is required for rendering. Use an existing environment with "
            "pyvista and vtk, or install the optional render dependencies in your "
            "own environment. Optimization does not require them."
        ) from exc
    return pv


def _density_and_extent(rho_yxz, extent=None):
    rho = np.asarray(rho_yxz, dtype=np.float64)
    if rho.ndim != 3 or any(s < 1 for s in rho.shape):
        raise ValueError("density must be a nonempty 3D array with shape (y, x, z)")
    if not np.isfinite(rho).all():
        raise ValueError("density contains NaN or infinity")
    if np.any(rho < 0) or np.any(rho > 1):
        raise ValueError("density values must lie in [0, 1]")
    ny, nx, nz = rho.shape
    dims = np.asarray((nx, ny, nz) if extent is None else extent, dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("extent must contain three positive finite xyz lengths")
    return rho, dims


def density_to_surface(rho_yxz, *, extent=None, cutoff: float = 0.5):
    """Extract a closed isosurface in physical CAD xyz coordinates.

    Samples are FE cell centers. Zero padding closes surfaces on the boundary.
    If supersampling a domain, pass its original xyz extent to retain dimensions.
    The Top3d row-zero-at-y-max convention is converted to physical y-up.
    """
    pv = _pyvista()
    rho, dims = _density_and_extent(rho_yxz, extent)
    if not np.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError("cutoff must lie strictly between zero and one")
    ny, nx, nz = rho.shape
    spacing = dims / np.asarray((nx, ny, nz), dtype=float)
    # VTK point scalars flatten with x varying fastest. The source is y,x,z.
    xyz = np.pad(rho[::-1, :, :].transpose(1, 0, 2), 1, constant_values=0)
    grid = pv.ImageData(
        dimensions=xyz.shape,
        spacing=tuple(spacing),
        origin=tuple(-0.5 * spacing),
    )
    grid.point_data["density"] = xyz.ravel(order="F")
    mesh = grid.contour([float(cutoff)], scalars="density").triangulate()
    if mesh.n_points:
        mesh = mesh.clean().compute_normals(
            consistent_normals=True, auto_orient_normals=True, inplace=False,
        )
    return mesh


def _attach_curvature(mesh, curvature):
    if curvature is None or mesh.n_points == 0:
        return
    values = np.asarray(curvature(mesh.points) if callable(curvature) else curvature).reshape(-1)
    if len(values) != mesh.n_points:
        raise ValueError("curvature must provide one value per extracted mesh vertex")
    if not np.isfinite(values).all():
        raise ValueError("curvature contains NaN or infinity")
    mesh.point_data["absolute_gaussian_curvature"] = np.abs(values)


def _add_topology(plotter, mesh, curvature, curvature_limit, pv):
    if mesh.n_points == 0:
        return
    kwargs = dict(
        smooth_shading=True, split_sharp_edges=False, show_edges=False,
        ambient=0.25, diffuse=0.75, specular=0.18, specular_power=25,
    )
    if curvature is None:
        plotter.add_mesh(mesh, color="#dadbdd", **kwargs)
    else:
        if "absolute_gaussian_curvature" not in mesh.point_data:
            _attach_curvature(mesh, curvature)
        rgba = np.full((256, 4), 255, dtype=np.uint8)
        rgba[:, 1:3] = np.round(255 * (1 - np.linspace(0, 1, 256)))[:, None]
        lut = pv.LookupTable()
        lut.values = rgba
        lut.scalar_range = (0, float(curvature_limit))  # PyVista LUT objects override add_mesh(clim).
        plotter.add_mesh(
            mesh, scalars="absolute_gaussian_curvature", cmap=lut,
            clim=(0, curvature_limit), show_scalar_bar=False, **kwargs,
        )


def _points(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3))
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite array with shape (N, 3)")
    return arr


def _subsample_indices(n: int, maximum: int):
    return np.unique(np.linspace(0, n - 1, min(n, maximum)).round().astype(int))


def _add_bc(plotter, bc, dims, pv, *, design_box=True):
    if bc is None:
        data = {}
    elif isinstance(bc, Mapping):
        data = bc
    elif hasattr(bc, "to_render_dict"):
        data = bc.to_render_dict()
    elif all(hasattr(bc, k) for k in ("supports", "loads", "forces")):
        data = {"fixed_points_xyz": bc.supports, "fixed_components": getattr(bc, "constrained", None),
                "load_points_xyz": bc.loads, "load_vectors_xyz": bc.forces}
    else:
        raise TypeError("bc must be a mapping or expose physical supports/loads/forces")
    scale = float(np.max(dims))
    if design_box:
        box = pv.Box(bounds=(0, dims[0], 0, dims[1], 0, dims[2]))
        plotter.add_mesh(box, color="#b9bdc2", opacity=0.18, lighting=False)
        plotter.add_mesh(box, style="wireframe", color="#b2b6bb", opacity=0.45, line_width=1)
    fixed = _points(data.get("fixed_points_xyz", []), "fixed_points_xyz")
    if len(fixed):
        # Filled support patches are reconstructed only for planar axis-aligned
        # rectangular supports. Other supports remain exact red node glyphs.
        span = np.ptp(fixed, axis=0)
        flat_axes = np.flatnonzero(span < 1e-10)
        if len(flat_axes) == 1 and len(fixed) >= 4:
            axis = int(flat_axes[0])
            active = [a for a in range(3) if a != axis]
            axes_full = all(len(np.unique(fixed[:, a])) >= 2 for a in active)
            counts = np.prod([len(np.unique(fixed[:, a])) for a in active])
            if axes_full and counts == len(np.unique(fixed, axis=0)):
                center = np.mean([fixed.min(axis=0), fixed.max(axis=0)], axis=0)
                # Build the exact patch so the Plane automatic local frame cannot
                # swap widths for unusual dimensions.
                corners = []
                for a, b in [(0,0),(1,0),(1,1),(0,1)]:
                    p = center.copy()
                    p[active[0]] = fixed[:, active[0]].max() if a else fixed[:, active[0]].min()
                    p[active[1]] = fixed[:, active[1]].max() if b else fixed[:, active[1]].min()
                    corners.append(p)
                face = pv.PolyData(np.array(corners), np.array([4,0,1,2,3]))
                plotter.add_mesh(face, color="#e71b23", ambient=0.5, diffuse=0.5)
        idx = _subsample_indices(len(fixed), 300)
        plotter.add_mesh(pv.PolyData(fixed[idx]), color="#e71b23",
            render_points_as_spheres=True, point_size=16)
    loads = _points(data.get("load_points_xyz", data.get("loads_points_xyz", [])), "load_points_xyz")
    vectors = _points(data.get("load_vectors_xyz", []), "load_vectors_xyz")
    if len(loads) != len(vectors):
        raise ValueError("load points and vectors must have equal lengths")
    if len(loads):
        lengths = np.linalg.norm(vectors, axis=1)
        valid = np.flatnonzero(lengths > 0)
        if len(valid):
            idx = valid[_subsample_indices(len(valid), 60)]
            unit = vectors[idx] / lengths[idx, None]
            arrow_length = scale * 0.13
            # Arrowheads end at the loaded node and show the signed physical force.
            cloud = pv.PolyData(loads[idx] - unit * arrow_length)
            cloud.point_data["force_direction"] = unit
            glyph = cloud.glyph(orient="force_direction", scale=False,
                factor=arrow_length, geom=pv.Arrow(tip_length=.3, tip_radius=.08, shaft_radius=.025))
            plotter.add_mesh(glyph, color="#22cd35", ambient=0.35, diffuse=0.65)
            plotter.add_mesh(pv.PolyData(loads[idx]), color="#22cd35",
                render_points_as_spheres=True, point_size=8)


def _prepare_plotter(pv, dims, size, *, camera=None):
    if len(size) != 2 or any(int(s) < 64 for s in size):
        raise ValueError("size must be two pixel dimensions >= 64")
    plotter = pv.Plotter(off_screen=True, window_size=tuple(int(s) for s in size))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    center = dims / 2
    distance = float(np.linalg.norm(dims)) * 2.6
    direction = np.array([0.487, 0.542, 1.0])
    direction /= np.linalg.norm(direction)
    plotter.camera_position = camera if camera is not None else [
        tuple(center + direction * distance), tuple(center), (0, 1, 0),
    ]
    plotter.enable_parallel_projection()
    return plotter


def _save(plotter, output, *, transparent, dpi):
    output = Path(output).resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("figure output must have a .png suffix")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.reset_camera(render=False)
    plotter.camera.zoom(1.10)
    plotter.show(auto_close=False)
    plotter.screenshot(output, transparent_background=transparent, return_img=False)
    # PyVista controls pixel dimensions. Add physical DPI metadata without
    # changing pixels, for drop-in use in publication layouts.
    try:
        from PIL import Image
        with Image.open(output) as img:
            img.save(output, dpi=(dpi, dpi))
    except ImportError:
        pass
    return str(output)


def _cylinder_preview(rho, dims, cylinders, pv, cutoff):
    # Notebook assembly preview; edited display mesh and separate gray collars.
    values = np.asarray(cylinders, dtype=float)
    if values.size == 0:
        return density_to_surface(rho, extent=dims, cutoff=cutoff), []
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("cylinders must be finite rows [center_x,center_y,diameter,height]")
    if np.any(values[:,2] <= 3) or np.any(values[:,3] <= 0):
        raise ValueError("preview collar diameter must exceed 3 and height be positive")
    ny, nx, nz = rho.shape
    yy = dims[1] - (np.arange(ny) + .5) * dims[1] / ny
    xx = (np.arange(nx) + .5) * dims[0] / nx
    zz = (np.arange(nz) + .5) * dims[2] / nz
    y, x, z = np.meshgrid(yy, xx, zz, indexing="ij")
    preview = rho.copy()
    collars = []
    theta = np.linspace(0, 2*np.pi, 129)[:-1]
    n = len(theta)
    for cx, cy, diameter, height in values:
        trim_radius = (diameter - 1) / 2
        trim = ((x-cx)**2 + (y-cy)**2 < trim_radius**2) & (z <= height)
        preview[trim] = 0
        outer, inner = diameter/2, (diameter-3)/2
        points = []
        for level, radius in [(0,outer),(height,outer),(0,inner),(height,inner)]:
            points.extend(np.column_stack((cx + radius*np.cos(theta),
                cy + radius*np.sin(theta), np.full(n,level))))
        faces = []
        for i in range(n):
            j=(i+1)%n
            faces.extend([4,i,j,n+j,n+i])
            faces.extend([4,2*n+j,2*n+i,3*n+i,3*n+j])
            faces.extend([4,i,2*n+i,2*n+j,j])
            faces.extend([4,n+j,3*n+j,3*n+i,n+i])
        collar=pv.PolyData(np.asarray(points), np.asarray(faces)).triangulate()
        collars.append(collar.compute_normals(auto_orient_normals=True, split_vertices=True,
            feature_angle=45, consistent_normals=True, inplace=False))
    return density_to_surface(preview, extent=dims, cutoff=cutoff), collars


def render_density(
    rho_yxz, output, *, extent=None, cutoff=0.5, curvature=None, bc=None,
    size=(1500, 1500), transparent=True, curvature_limit=0.04,
    mesh_output=None, camera=None, dpi=300, cylinders=None,
):
    """Render paper-style topology and optional analytical curvature/BCs.

    ``curvature`` is either a callback receiving physical xyz mesh vertices, or
    values matching those vertices. Use the neural field Hessian for paper
    curvature; discrete mesh curvature is deliberately not substituted.
    ``bc`` accepts the keys described in ``docs/VISUALIZATION.md``.
    ``cylinders`` optionally adds the notebook illustrative collar assembly to
    the image only; mesh_output retains the exact optimized-density surface.
    Returns a small report containing output paths and exact surface counts.
    """
    pv = _pyvista()
    rho, dims = _density_and_extent(rho_yxz, extent)
    if not np.isfinite(curvature_limit) or curvature_limit <= 0:
        raise ValueError("curvature_limit must be positive and finite")
    mesh = density_to_surface(rho, extent=dims, cutoff=cutoff)
    _attach_curvature(mesh, curvature)
    display_mesh, collars = (mesh, []) if cylinders is None else _cylinder_preview(rho, dims, cylinders, pv, cutoff)
    plotter = _prepare_plotter(pv, dims, size, camera=camera)
    try:
        _add_topology(plotter, display_mesh, curvature, curvature_limit, pv)
        for collar in collars:
            _add_topology(plotter, collar, None, curvature_limit, pv)
        if bc is not None:
            _add_bc(plotter, bc, dims, pv)
        result = _save(plotter, output, transparent=transparent, dpi=dpi)
        if mesh_output is not None:
            mesh_path = Path(mesh_output).resolve()
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(mesh_path)
        return {
            "figure": result, "points": int(mesh.n_points), "triangles": int(mesh.n_cells),
            "extent_xyz": dims.tolist(), "cutoff": float(cutoff),
            "curvature_limit": float(curvature_limit), "dpi": int(dpi),
            "illustrative_collar_count": len(collars),
        }
    finally:
        plotter.close()


def render_boundary_conditions(dims_xyz, bc, output, *, size=(1500,1500),
                               transparent=True, camera=None, dpi=300):
    """Render exact support nodes (red), load vectors (green), and design box."""
    pv = _pyvista()
    dims = np.asarray(dims_xyz, dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("dims_xyz must be three positive finite lengths")
    plotter = _prepare_plotter(pv, dims, size, camera=camera)
    try:
        _add_bc(plotter, bc, dims, pv)
        result = _save(plotter, output, transparent=transparent, dpi=dpi)
        return {"figure": result, "extent_xyz": dims.tolist(), "dpi": int(dpi)}
    finally:
        plotter.close()
