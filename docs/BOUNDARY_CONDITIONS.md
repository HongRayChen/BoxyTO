# Define boundary conditions and fixed regions

All public points use physical **x,y,z** coordinates. Nodes are integer coordinates in inclusive bounds [0,nx] x [0,ny] x [0,nz]. Density arrays internally reverse y; the public API handles that conversion. No hidden z reflection is applied.

## Paper and random cases

```json
{"problem": {"type":"paper","name":"mbb"}, "optimization":{"mode":"5axis"}}
```

Paper names are mbb, mount, pedal, and link. Optional `dims` resamples a paper case by proportional scaling and rounded node selection; original dimensions are recommended for comparisons. Optional `volume_fraction` overrides the default target.

```json
{"problem": {"type":"random","dims":[40,24,12],"seed":300,"volume_fraction":0.1},
 "optimization":{"mode":"5axis","directions":["z+","z-"],"regularization_weight":0}}
```

The random generator uses a local NumPy PCG64 seed, full surface-grid candidate nodes, validated constraints, and checked load/support separation. Its saved `boundary.json` records realized points, forces, and DOF constraints. Random seeds are reproducible within this generator; they do not promise the identical pyFANTOM notebook seed-to-geometry mapping. The public physical coordinate convention replaces implicit notebook z flips.

## Custom conditions

Use `examples/custom.json` as a starting point. A custom problem provides dimensions, target volume, and loads/supports. A load at a point has an xyz force vector; a support has an xyz Boolean list of constrained displacement components. True means zero displacement, false means free displacement.

A load box uses [[xmin,xmax],[ymin,ymax],[zmin,zmax]] and selects every integer node inside its inclusive bounds. With `total:true`, the given force is divided across selected nodes; with `total:false`, that force is applied to every selected node. Supports can constrain only selected components, provided the final structure constrains all rigid motions.

The API rejects coordinates outside the grid, noninteger node coordinates, nonzero loads applied to fixed DOFs, and supports that leave rigid-body modes. Duplicate nodal forces add rather than overwrite.

## Fixed material

Every problem can include a `regions` list. A region has `kind:"void"` (density exactly zero) or `kind:"solid"` (density exactly one), and either a `type:"sphere"` with physical xyz `center` and `radius`, or `type:"box"` with bounds [[xmin,xmax],[ymin,ymax],[zmin,zmax]]. Regions are evaluated at FE cell centers. Void/solid masks may not overlap, and the volume target must be feasible for the prescribed masks.

```json
{"type":"sphere","center":[20,10,4],"radius":3,"kind":"void"}
```

For smoother analytic fixed-region boundaries at higher resolution, extend the region evaluator explicitly; the default renderer repeats FE masks exactly and does not silently change the analyzed domain.

## Python API

```python
from boxyto.boundary import Grid, BoundaryConditions, ProblemSpec
from boxyto.fea import Top3dFE
import numpy as np

grid = Grid(12, 6, 4)
supports = np.array([[0,y,z] for y in range(7) for z in range(5)])
bc = BoundaryConditions(
    loads=np.array([[12,0,2]]),
    forces=np.array([[0,1,0]]),
    supports=supports,
    constrained=np.ones_like(supports, dtype=bool),
)
fe = Top3dFE(grid, bc, backend="cpu")
result = fe.evaluate(np.full(grid.shape, 0.3), penal=3)
print(result.compliance, result.residual)
```

Physical length unit is one element edge; force and modulus units must be mutually consistent. This reference implementation uses uniform unit cubes and isotropic elasticity.
