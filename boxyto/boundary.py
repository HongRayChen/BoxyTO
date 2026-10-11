"""Structured-grid boundary conditions and the four original BoxyTO problems.

Public coordinates are physical integer grid nodes (x, y, z). Density arrays
have shape (ny, nx, nz), with their first row adjacent to physical y=ny.
There is no implicit z reflection. The node/element layout is exactly Top3d.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int

    def __post_init__(self):
        for name in ("nx", "ny", "nz"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v or v < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(v))

    @property
    def dims(self):
        return (self.nx, self.ny, self.nz)

    @property
    def shape(self):
        return (self.ny, self.nx, self.nz)

    @property
    def nele(self):
        return self.nx * self.ny * self.nz

    @property
    def nnodes(self):
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def ndof(self):
        return 3 * self.nnodes

    def node_ids(self, xyz):
        points = _points(xyz, "node coordinates")
        if not np.allclose(points, np.rint(points), rtol=0, atol=1e-10):
            raise ValueError("Boundary coordinates must be integer grid nodes")
        p = np.rint(points).astype(np.int64)
        if np.any(p < 0) or np.any(p > np.array(self.dims)):
            raise ValueError(f"Boundary coordinates are outside grid bounds {self.dims}")
        return p[:, 2] * (self.nx + 1) * (self.ny + 1) + p[:, 0] * (self.ny + 1) + self.ny - p[:, 1]

    def nodes_xyz(self):
        z, x, row = np.meshgrid(np.arange(self.nz + 1), np.arange(self.nx + 1), np.arange(self.ny + 1), indexing="ij")
        return np.stack((x.ravel(), self.ny - row.ravel(), z.ravel()), axis=1)

    def cell_centers_xyz(self):
        row, x, z = np.meshgrid(np.arange(self.ny) + .5, np.arange(self.nx) + .5, np.arange(self.nz) + .5, indexing="ij")
        return np.stack((x, self.ny - row, z), axis=-1)


def _points(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        return np.empty((0, 3), dtype=float)
    if a.ndim == 1 and a.shape == (3,):
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite (N, 3) array")
    return a


@dataclass
class BoundaryConditions:
    loads: np.ndarray
    forces: np.ndarray
    supports: np.ndarray
    constrained: np.ndarray

    def __post_init__(self):
        self.loads = _points(self.loads, "loads")
        self.forces = _points(self.forces, "forces")
        self.supports = _points(self.supports, "supports")
        c = np.asarray(self.constrained)
        if c.ndim == 1 and c.shape == (3,):
            c = np.broadcast_to(c, self.supports.shape).copy()
        if c.shape != self.supports.shape or not np.all(np.isin(c, [0, 1, False, True])):
            raise ValueError("constrained must contain three Boolean DOF flags per support")
        self.constrained = c.astype(bool)
        if self.loads.shape != self.forces.shape:
            raise ValueError("Each load point needs exactly one three-component force")
        if not len(self.loads) or not np.any(self.forces):
            raise ValueError("At least one nonzero force is required")
        if not len(self.supports) or not np.any(self.constrained):
            raise ValueError("At least one constrained support is required")

    def dof_data(self, grid: Grid):
        load_ids = grid.node_ids(self.loads)
        support_ids = grid.node_ids(self.supports)
        f = np.zeros(grid.ndof, dtype=float)
        np.add.at(f, (3 * load_ids[:, None] + np.arange(3)).ravel(), self.forces.ravel())
        fixed = np.unique((3 * support_ids[:, None] + np.arange(3))[self.constrained])
        if len(fixed) == grid.ndof:
            raise ValueError("All DOFs are fixed; there is no structural equilibrium to solve")
        if np.any(np.abs(f[fixed]) > 1e-14):
            raise ValueError("Nonzero force applied to a constrained DOF; move the load or release that DOF")
        if not np.any(f):
            raise ValueError("Loads cancel exactly; assembled nodal force vector is zero")
        # Check all six rigid motions analytically before attempting a sparse solve.
        center = np.array(grid.dims, dtype=float) / 2
        p = self.supports - center
        skew = np.zeros((len(p), 3, 3))
        skew[:, 0, 1] = -p[:, 2]; skew[:, 0, 2] = p[:, 1]
        skew[:, 1, 0] = p[:, 2]; skew[:, 1, 2] = -p[:, 0]
        skew[:, 2, 0] = -p[:, 1]; skew[:, 2, 1] = p[:, 0]
        rigid = np.concatenate((np.broadcast_to(np.eye(3), skew.shape), -skew), axis=2)
        if np.linalg.matrix_rank(rigid[self.constrained]) < 6:
            raise ValueError("Supports leave an unconstrained rigid-body motion (rank < 6)")
        return f, fixed

    def to_dict(self):
        return {"loads": self.loads.tolist(), "forces": self.forces.tolist(),
                "supports": self.supports.tolist(), "constrained": self.constrained.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["loads"], data["forces"], data["supports"], data.get("constrained", [True, True, True]))


@dataclass
class ProblemSpec:
    grid: Grid
    bc: BoundaryConditions
    volume_fraction: float
    void_mask: np.ndarray | None = None
    solid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.volume_fraction) or not 0 < self.volume_fraction <= 1:
            raise ValueError("volume_fraction must be in (0, 1]")
        self.void_mask = self._mask(self.void_mask, "void_mask")
        self.solid_mask = self._mask(self.solid_mask, "solid_mask")
        if np.any(self.void_mask & self.solid_mask):
            raise ValueError("Forced solid and forced void masks overlap")
        if self.solid_mask.mean() > self.volume_fraction + 1e-12:
            raise ValueError("Forced solid region exceeds the requested total-domain volume fraction")
        if (~self.void_mask).mean() < self.volume_fraction - 1e-12:
            raise ValueError("Forced void region makes the requested volume fraction infeasible")
        self.bc.dof_data(self.grid)

    def _mask(self, value, name):
        if value is None:
            return np.zeros(self.grid.shape, dtype=bool)
        a = np.asarray(value)
        if a.shape != self.grid.shape or not np.all(np.isin(a, [0, 1, False, True])):
            raise ValueError(f"{name} must be a Boolean array with shape {self.grid.shape}")
        return a.astype(bool)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def dims(self):
        return self.grid.dims

    def apply_masks(self, density):
        a = np.asarray(density)
        if a.shape != self.shape:
            raise ValueError(f"density must have shape {self.shape}")
        return np.where(self.void_mask, 0., np.where(self.solid_mask, 1., a))

    def apply_masks_torch(self, density):
        import torch
        if tuple(density.shape) != self.shape:
            raise ValueError(f"density must have shape {self.shape}")
        void = torch.as_tensor(self.void_mask, device=density.device)
        solid = torch.as_tensor(self.solid_mask, device=density.device)
        return torch.where(void, torch.zeros_like(density), torch.where(solid, torch.ones_like(density), density))


def select_nodes(grid: Grid, box):
    """Select nodes in an inclusive [[xmin,xmax],[ymin,ymax],[zmin,zmax]] box."""
    b = np.asarray(box, dtype=float)
    if b.shape != (3, 2) or not np.all(np.isfinite(b)) or np.any(b[:, 0] > b[:, 1]):
        raise ValueError("box must be finite (3, 2) inclusive coordinate bounds")
    xyz = grid.nodes_xyz()
    return xyz[np.all((xyz >= b[:, 0]) & (xyz <= b[:, 1]), axis=1)]


def cylinder_shell_nodes(grid: Grid, center, diameter, thickness=1.0):
    """Original notebook cylindrical support/load shell, including every z layer."""
    xyz = grid.nodes_xyz()
    r = np.linalg.norm(xyz[:, :2] - np.asarray(center), axis=1)
    mask = (r >= diameter / 2 - thickness / 2) & (r <= diameter / 2 + thickness / 2)
    if not np.any(mask):
        raise ValueError("Cylinder shell selects no grid nodes")
    return xyz[mask]


def sphere_mask(grid: Grid, center=None, radius=None):
    center = np.array(grid.dims, dtype=float) / 2 if center is None else np.asarray(center, dtype=float)
    radius = grid.ny / 4 if radius is None else float(radius)
    if center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("sphere center must be finite xyz and radius must be positive")
    return np.sum((grid.cell_centers_xyz() - center) ** 2, axis=-1) <= radius ** 2


def box_mask(grid: Grid, bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (3, 2) or not np.all(np.isfinite(b)) or np.any(b[:, 0] > b[:, 1]):
        raise ValueError("mask bounds must be finite (3, 2)")
    xyz = grid.cell_centers_xyz()
    return np.all((xyz >= b[:, 0]) & (xyz <= b[:, 1]), axis=-1)


def _distributed(points, total):
    if not len(points):
        raise ValueError("Load selector contains no nodes")
    return np.broadcast_to(np.asarray(total, dtype=float) / len(points), points.shape).copy()


def paper_case(name: str, dims=None, no_design=False):
    """Reproduce four objective_tuning BCs; dims scales geometry for quick demos.

    Default resolutions and forces are exact. Fractional scaled cylindrical
    centers are allowed, while BC nodes always remain integer grid locations.
    Render-only collars are recorded in metadata, not inserted into the FE mesh.
    """
    defaults = {"mbb": (40, 20, 8, .3), "mount": (40, 20, 8, .2),
                "link": (40, 20, 8, .25), "pedal": (50, 20, 8, .2)}
    if name not in defaults:
        raise ValueError(f"Unknown paper case {name!r}; choose {tuple(defaults)}")
    bx, by, bz, vf = defaults[name]
    grid = Grid(*(dims if dims is not None else (bx, by, bz)))
    sx, sy = np.array(grid.dims[:2]) / [bx, by]
    sr = min(sx, sy)
    def cyl(cx, cy, dia):
        return cylinder_shell_nodes(grid, (cx * sx, cy * sy), dia * sr, thickness=max(1., sr))
    collars = []
    if name == "mbb":
        loads = select_nodes(grid, [[grid.nx, grid.nx], [0, 0], [0, grid.nz]])
        forces = _distributed(loads, [0, 1, 0])
        supports = select_nodes(grid, [[0, 0], [0, grid.ny], [0, grid.nz]])
    elif name == "mount":
        loads = select_nodes(grid, [[0, grid.nx], [0, 0], [0, grid.nz]])
        forces = _distributed(loads, [0, 1, 0]); supports = cyl(17, 14, 10)
        collars = [[17*sx, 14*sy, 10*sr, grid.nz]]
    elif name == "link":
        a = cyl(5, 15, 6); b = cyl(35, 5, 6)
        loads = np.concatenate((a, b)); forces = np.concatenate((_distributed(a, [2, 0, 0]), _distributed(b, [0, 1, 0])))
        supports = cyl(11, 5, 6)
        collars = [[5*sx, 15*sy, 8*sr, grid.nz], [35*sx, 5*sy, 8*sr, grid.nz], [11*sx, 5*sy, 8*sr, grid.nz]]
    else:
        xa = int(round((bx-10)*sx)); ya = int(round((by-1)*sy))
        loads = select_nodes(grid, [[xa, grid.nx], [ya, ya], [0, grid.nz]])
        forces = _distributed(loads, [0, 1, 0]); supports = cyl(6, 10, 10)
        collars = [[6*sx, 10*sy, 11*sr, grid.nz]]
    bc = BoundaryConditions(loads, forces, supports, np.ones_like(supports, dtype=bool))
    return ProblemSpec(grid, bc, vf, sphere_mask(grid) if no_design else None,
                       metadata={"case": name, "source": "boxy_objective_tuning.ipynb", "scaled": dims is not None,
                                 "render_only_cylinders": collars, "default_machining_directions": ["z-"],
                                 "mbb_note": "Original notebook label; BC is a clamped cantilever" if name == "mbb" else ""})


def random_case(grid: Grid | tuple, seed=0, volume_fraction=.1, max_loads=4, support_style="points", no_design=False):
    """Seeded native random BCs with all rigid motions removed, no external FE code.

    Like the source generator, this uses 1..4 point loads, corner/edge/surface/
    interior positions, and point supports by default. Exact pyFANTOM seeds are
    not replayed: NumPy PCG64 is explicitly used and BCs are saved in each run.
    """
    if not isinstance(grid, Grid):
        grid = Grid(*grid)
    if int(max_loads) != max_loads or max_loads < 1:
        raise ValueError("max_loads must be a positive integer")
    if support_style not in ("points", "patch", "face"):
        raise ValueError("support_style must be points, patch, or face")
    rng = np.random.Generator(np.random.PCG64(seed))
    axis = int(rng.integers(3)); side = int(rng.integers(2))
    dim = np.array(grid.dims)
    clamp = side * dim[axis]
    if support_style == "points":
        other = [i for i in range(3) if i != axis]
        corners = np.zeros((4, 3)); corners[:, axis] = clamp
        corners[:, other[0]] = np.array([0, 0, 1, 1]) * dim[other[0]]
        corners[:, other[1]] = np.array([0, 1, 0, 1]) * dim[other[1]]
        supports = corners[rng.choice(4, 3, replace=False)]
    else:
        box = np.column_stack((np.zeros(3), dim)).astype(float)
        box[axis] = clamp
        if support_style == "patch":
            for a in range(3):
                if a != axis:
                    width = max(1, int(round(dim[a] * rng.uniform(.25, .65))))
                    start = int(rng.integers(0, dim[a] - width + 1))
                    box[a] = [start, start + width]
        supports = select_nodes(grid, box)
    fixed_set = set(grid.node_ids(supports).tolist())
    count = int(rng.integers(1, int(max_loads) + 1))
    loads, forces = [], []
    for _ in range(count):
        for attempt in range(100):
            point = np.array([rng.integers(0, int(d)+1) for d in dim])
            kind = int(rng.choice(4, p=[.25, .25, .25, .25]))
            if kind == 0:
                point = rng.integers(0, 2, 3) * dim
            elif kind in (1, 2):
                axes = rng.choice(3, 2 if kind == 1 else 1, replace=False)
                point[axes] = rng.integers(0, 2, len(axes)) * dim[axes]
            if int(grid.node_ids(point)[0]) not in fixed_set:
                break
        else:
            raise RuntimeError("Failed to sample a free load node")
        force = rng.normal(size=3); force /= np.linalg.norm(force)
        loads.append(point); forces.append(force / count)
    bc = BoundaryConditions(loads, forces, supports, [True, True, True])
    return ProblemSpec(grid, bc, volume_fraction, sphere_mask(grid) if no_design else None,
                       metadata={"case": "random", "seed": int(seed), "generator": "numpy.PCG64", "support_style": support_style,
                                 "source_compatibility": "Original distributions adapted; pyFANTOM seeds are not replayed"})


def custom_case(data: dict):
    """Read explicit BC arrays or inclusive node-box selectors and mask primitives.

    Required dims are [nx,ny,nz]. boundary may contain explicit arrays. Otherwise
    load_regions entries contain box, force, and optional total (default True);
    support_regions entries contain box and axes (default [True,True,True]).
    loads/supports may instead group explicit points with vectors/components.
    regions are sphere/box primitives with kind=void/solid; no_design and
    forced_solid accept the same primitives as separate lists.
    """
    grid = Grid(*data["dims"])
    if "boundary" in data:
        bc = BoundaryConditions.from_dict(data["boundary"])
    elif "loads" in data or "supports" in data:
        loads, forces, supports, constrained = [], [], [], []
        for r in data.get("loads", []):
            points = _points(r["points"], "load points")
            vectors = _points(r["vectors"], "load vectors")
            if len(vectors) == 1:
                vectors = np.broadcast_to(vectors, points.shape)
            loads.extend(points); forces.extend(vectors)
        for r in data.get("supports", []):
            points = _points(r["points"], "support points")
            components = np.asarray(r.get("components", [True, True, True]))
            if components.ndim == 1:
                components = np.broadcast_to(components, points.shape)
            supports.extend(points); constrained.extend(components)
        bc = BoundaryConditions(loads, forces, supports, constrained)
    else:
        loads, forces, supports, constrained = [], [], [], []
        for r in data.get("load_regions", []):
            p = select_nodes(grid, r["box"])
            loads.extend(p)
            forces.extend(_distributed(p, r["force"]) if r.get("total", True) else np.broadcast_to(r["force"], p.shape))
        for r in data.get("support_regions", []):
            p = select_nodes(grid, r["box"]); supports.extend(p)
            constrained.extend(np.broadcast_to(r.get("axes", [True, True, True]), p.shape))
        bc = BoundaryConditions(loads, forces, supports, constrained)
    def region_mask(regions):
        out = np.zeros(grid.shape, dtype=bool)
        for region in regions:
            if region["type"] == "sphere":
                out |= sphere_mask(grid, region["center"], region["radius"])
            elif region["type"] == "box":
                out |= box_mask(grid, region["bounds"])
            else:
                raise ValueError(f"Unknown mask primitive {region['type']!r}")
        return out
    regions = data.get("regions", [])
    if any(r.get("kind", "void") not in ("void", "solid") for r in regions):
        raise ValueError("region kind must be void or solid")
    void_regions = data.get("no_design", []) + [r for r in regions if r.get("kind", "void") == "void"]
    solid_regions = data.get("forced_solid", []) + [r for r in regions if r.get("kind", "void") == "solid"]
    return ProblemSpec(grid, bc, data.get("volume_fraction", .3), region_mask(void_regions),
                       region_mask(solid_regions), metadata={"case": "custom", "name": data.get("name", "custom")})
