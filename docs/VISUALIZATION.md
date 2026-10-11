# Publication figures with PyVista

All geometry and images are generated on the computation host. `boxyto.render`
uses PyVista/VTK for isosurface extraction, lighting, support glyphs, force arrows,
and offscreen PNG output. Optimization imports do not load PyVista. No Matplotlib
rendering code is used.

## Style and manuscript reference

The authors' manuscript archive contains `main.tex` and the original figure
assets. That source archive remains on the authors' workstation; the public
repository includes the [paper PDF](../BoxyTO_arxiv-2.pdf). The figure `pic_bc_compare.png` was visually inspected
against the renderer style. Its main conventions are:

- White or transparent background, no axes or grid, and an oblique orthographic
  view that exposes the front, right, and upper surfaces.
- Smooth, pale gray geometry with modest specular lighting.
- White-to-red absolute Gaussian curvature: low-curvature patches are white,
  high-curvature seams are red. The default saturation is 0.04, matching the
  nonnegative half of the original notebook `bwr` range of [-0.04, 0.04].
- Fixed-displacement supports are red; physical force vectors are green; the
  design envelope is translucent gray.
- 1500 by 1500 pixels with 300 DPI metadata, corresponding to a five-inch image.

These conventions reproduce the visual language of the paper. They do not imply
that a new optimization run reproduces the manuscript geometry or compliance.
PyVista lighting differs from the notebook face-color shading, and individual
paper panels may need their camera tuned for matching layout.

## Coordinates and scalar data

An FE density array has shape `(ny, nx, nz)`. Row zero is at the **maximum physical
y coordinate**, following the Top3d notebook convention. Public boundary points
and exported mesh vertices use physical CAD `(x, y, z)` coordinates, with the
design domain spanning `[0,nx] x [0,ny] x [0,nz]` in unit element lengths.

The renderer reverses the density y rows, transposes to xyz, and flattens with
Fortran order for VTK. It applies no z reflection. An asymmetric anisotropic
single-voxel regression test checks this mapping; cubic grids cannot reveal all
axis mistakes.

Density samples lie at voxel centers. With `extent=(nx,ny,nz)`, an array sampled
at four times the FE resolution still produces a mesh in the original physical
extent. One layer of zero padding closes surfaces that meet the domain edge.
The mesh is the interpolated density level set, rather than a union of voxel
boxes. The default level is 0.5. Always record the cutoff when reporting volume
or comparing structural evaluation.

The neural coordinates used by the port are ordered `(y,x,z)` and centered at
the domain midpoint with a factor of two:

```python
# points_xyz are the physical mesh points returned by PyVista.
field_coords_yxz = 2 * (
    points_xyz[:, [1, 0, 2]] - np.asarray([ny, nx, nz]) / 2
)
```

Curvature supplied to `render_density` is either one value per extracted mesh
vertex, or a callable receiving those physical xyz mesh vertices. For the paper
curvature plot, evaluate the trained field Gaussian curvature through its
analytical gradient and Hessian. The renderer takes the absolute value and keeps
all finite values in the exported mesh; the color saturation clips only their
display. Discrete mesh Gaussian curvature is not substituted for neural curvature.
The scale 0.04 follows the notebook coordinate convention; it is not a universal
physical curvature threshold.

## Renderer API

```python
from boxyto.render import render_density, render_boundary_conditions

# rho_yxz may be the FE density or a denser neural field sample.
render_density(
    rho_yxz, "outputs/example/gray.png", extent=(nx, ny, nz),
    size=(1500, 1500), transparent=True,
    mesh_output="outputs/example/surface.vtp",
)

# curvature_fn receives physical xyz and returns one scalar per point.
render_density(
    rho_yxz, "outputs/example/curvature.png", extent=(nx, ny, nz),
    curvature=curvature_fn, curvature_limit=0.04,
    mesh_output="outputs/example/surface_curvature.vtp",
)

# BoundaryConditions instances are accepted directly.
render_boundary_conditions((nx, ny, nz), boundary_conditions,
                           "outputs/example/boundary_conditions.png")
```

Paths are examples relative to the remote project. Output paths must have a
`.png` suffix. `surface.vtp` preserves topology and point scalars for inspection
in ParaView; STL/PLY export is also supported through PyVista `mesh_output`.
Exported meshes are not smoothed, decimated, or given decorative collars, so an
export is not silently altered for appearance.

A plain boundary-condition mapping is also accepted:

```python
bc = {
    "fixed_points_xyz": supports,          # (N,3) coordinates
    "fixed_components": constrained,      # (N,3) booleans, optional
    "load_points_xyz": loads,              # (M,3) coordinates
    "load_vectors_xyz": forces,            # (M,3) physical force vectors
}
```

The alias `loads_points_xyz` is accepted. The native boundary object exposes
`supports`, `constrained`, `loads`, and `forces`. Dense supports and force arrows
are subsampled only for the figure: all FE conditions remain unchanged. A full
rectangular planar support receives a filled red patch; other supports use exact
node locations. Red marks indicate constrained nodes; partial support components
must be read from the saved BC data. Force arrows point in the signed force
direction and end at the loaded node. Arrow display length is normalized for
legibility, so read magnitudes from the BC file.

The original paper depicts cylindrical collars on the mount, pedal, and link.
Some of those notebook collars were added only during plotting. This renderer
does not add material absent from the optimized density by default. For an
explicit manuscript assembly illustration, `cylinders=[[cx,cy,diameter,height],...]`
creates gray annular collars in the image and trims the display density around
them. As in the notebook, inner diameter is `diameter-3` and the topology trim
diameter is `diameter-1`, in element units. This is a cosmetic assembly preview:
`mesh_output` still contains the unmodified optimized density surface. Use a
separate `paper.png` name for this preview and retain `gray.png`/`curvature.png`
for the evaluated topology. With collars and curvature, provide a callback so
curvature can be evaluated at both exact and preview vertices. Use explicit fixed-solid
and fixed-void masks when a bearing collar must be part of the evaluated design.

## Camera and export

The default camera uses y-up, a front/right/top direction, and parallel projection.
Its direction (0.487,0.542,1.0) reproduces the notebook view at elevation 26
and azimuth -64 after converting notebook display axes into physical xyz.
For a matching individual paper panel, pass `camera=[position_xyz, target_xyz,
up_vector_xyz]`. Choose one camera and one pixel size per comparison figure,
and retain those values with the run metadata. Transparent PNGs can be composed
in a vector editor without a white rectangular background. Use `transparent=False`
for a white-background image.

The existing `mlfa4` environment on f80 includes PyVista 0.47.3 and VTK with EGL.
Offscreen rendering was exercised there without package installation or a local
server. A first-use `bad X server connection` warning is followed by EGL fallback
and does not indicate a failed screenshot. Use the returned output path and
inspect the actual image. On another host, a working OpenGL/EGL or display setup
is required; no automatic dependency installation is performed.

## Paper experiment settings

The manuscript table reports MBB `(40,20,8)` with target volume 0.30; bearing mount
`(40,20,8)` with 0.20; pedal `(50,20,8)` with 0.20; and link `(40,20,8)` with 0.25.
These are four structural cases, distinct from the four objective combinations:
compliance only; compliance plus accessibility; developability for 5-axis; and
cutter-surface alignment for 3-axis.

The notebook uniform feature-scale sweep is `0.5,0.6,0.7,0.8,0.9,1.0` across all
three fields. Its anisotropic sweep scales the z input of fields 2 and 3 across
the same values, retaining field 1. In the manuscript 3-axis examples, z scaling
is reduced to 0.5 and the alignment loss is enabled. A 5-axis example omits the
fixed cutter-surface alignment loss while retaining developability and the
specified accessibility directions. This is a geometry objective, not a complete
5-axis collision-free CAM toolpath generator.

The paper random-BC comparison uses target volume 0.10, varied domain aspect
ratios, and either z-minus access or two-sided z-plus/z-minus access. Its example
feature-scale figures illustrate how larger input scaling permits denser seams;
FE resolution ultimately limits resolved details. Increasing rendering sampling
alone cannot recover features unresolved by the FE optimization.

## Verification

`tests/test_render.py` checks:

- Full-domain closed-surface bounds and positive orientation.
- A single voxel in an asymmetric grid with unequal xyz element dimensions,
  including the row-zero-at-y-max conversion.
- Empty surfaces and invalid density/cutoff inputs.
- Actual offscreen RGBA screenshots, visible curvature colors, scalar-preserving
  surface export, and boundary-condition output.
- Optional illustrative collars preserve all exact exported mesh points, faces,
  and analytical curvature scalars.
- A constant curvature of 0.04 renders saturated red, catching PyVista lookup-table
  ranges that override the requested color limits.

Run from the remote project in a PyVista-enabled environment:

```bash
python -m unittest discover -s tests -p 'test_render.py' -v
```

## Validation figures and interpretation

All eight 400-epoch paper-case runs were visually reviewed against the manuscript.
Gray geometry, red analytical-curvature seams, white background, and the derived
camera match its visual conventions. The 3-axis examples have broader aligned
surfaces than the corresponding 5-axis examples. They are new port results;
original manuscript geometries and numerical values are not claimed reproduced.

The 300-DPI plate results/validation/paper_cases.png has MBB, mount, pedal, and link
rows, with boundary conditions, 3-axis, and 5-axis columns. Caption:
**Analyzed-density geometry with analytical absolute Gaussian curvature; white
to red spans [0,0.04]. Cosmetic collars omitted.** Tight crops improve layout;
read physical dimensions from configs, not pixel lengths.

Exact curvature figures omit decorative collars. Separate paper.png assembly
previews add them and can conceal missing structural contact. At threshold 0.5,
the original paper-configuration validation runs have these loaded-node contacts:

| Run | Loaded nodes without adjacent density > 0.5 | Total loaded nodes |
| --- | ---: | ---: |
| Mount 3-axis | 0 | 369 |
| Mount 5-axis | 14 | 369 |
| Pedal 3-axis | 15 | 99 |
| Pedal 5-axis | 12 | 99 |
| Link 3-axis | 48 | 288 |
| Link 5-axis | 13 | 288 |

For example, mount 5-axis has continuous compliance 1.5442 and binary compliance
315.3064. Missing thresholded load contact is a plausible cause: forces still act
through minimum-stiffness void. A visually clean assembly preview does not
establish binary structural validity. Inspect saved FE residuals and contact
diagnostics; use explicit physical solid/void masks for separately labeled
bearing-assembly studies.

The separate paper_previews.png plate adds cosmetic collars for mount, pedal,
and link; its caption states that FE results and exported meshes retain the
exact density. Use paper_cases.png for structural interpretation.

The feature_scale.png plate compares all-field scaling 0.5, all-field scaling
1.0, and all-field scaling 1.0 with local z scales (1.0,0.5,0.5) for fields
1/2/3. All use the same MBB grid, seed, epochs, and 5-axis objective. These
settings change representable detail; no fixed physical feature-size
guarantee is implied. All local x/y scales stay at 1.0.

## Regenerate comparison plates

The straight-line composer reads saved PNGs and run metadata; it does not rerun
optimization or FE. Run from the project in an existing Pillow-enabled environment:

    python scripts/render_comparisons.py --root results/validation

The root defaults to results/validation. It rewrites paper_cases.png,
paper_previews.png, and feature_scale.png with white backgrounds and 300 DPI
metadata. The script keeps exact geometry separate from cosmetic collars and
reads feature-scale labels from the saved configs. All named runs must already
have their figures; feature comparisons require matching case, grid, objective
mode, epoch count, seed, and volume target.
