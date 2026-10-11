# Method and notebook provenance

## Source mapping

| Source notebook | Material retained |
| --- | --- |
| `boxy_objective_tuning.ipynb` | Four paper problems, three-field representation, optimizer schedules, curvature/intersection/alignment losses |
| `boxy_45_3field_update_rendering.ipynb` | Supersampled neural geometry and absolute-curvature figure conventions |
| `boxy_45_3field_no_design.ipynb` | No-design-region workflow |
| `boxy_random_bc_flipz_gen1_1.ipynb` | Seeded randomized support/load distributions and selectable machining directions |

Source notebooks are unchanged. The handoff package uses PyTorch analytic spatial derivatives and a bundled H8 FE implementation instead of TensorFlow and pyFANTOM. The optional independent reference check uses existing TensorFlow only for validation, not runtime.

## Representation and objective

Each of three cosine feature fields has a sigmoid density. The physical density is their product. Spectral kernels and output weights are trainable; the initial kernel multipliers are 1, 0.95, and 0.9. Defaults use the objective-tuning notebook's four bands (512 features per field), spectral extent 0.6, zero weights, and Adam learning rate 0.002 with epsilon 1e-7.

The objective combines normalized SIMP compliance, a volume-fraction penalty, weight regularization, sampled inaccessible void, Gaussian curvature away from seamlines, intersection-angle quality, and optional cutter alignment. SIMP exponent increases from 2 to 4 at 0.01/epoch; the volume coefficient rises from 1 to 100 at 0.5/epoch. Accessibility and alignment begin after epoch 20, curvature after epoch 40, and intersection-angle tightening after epoch 60. Continuation ramps span 300 epochs; the angle caps are 80 and 90 degrees. The notebooks and the supplied manuscript have some inconsistent tabulated ramp values; runnable defaults follow the objective-tuning notebook implementation.

Five-axis mode keeps developability/intersection/accessibility losses and disables alignment to a fixed cutter axis. Three-axis mode adds alignment and the supplied examples also scale the z coordinates of fields 2 and 3 by 0.5. This formulation controls surface developability and selected accessibility directions; it does not implement a complete five-axis toolpath planner.

Other modes `compliance` and `accessibility` allow the paper's baseline objectives to be explored with the same problem specification.

## Coordinates and fixed regions

Physical coordinates are CAD-style xyz with ranges [0,nx], [0,ny], [0,nz]. FE density arrays are (ny,nx,nz), with row zero adjacent to y=ny and elements flattened in Fortran order for Top3d assembly. Neural coordinates are centered (y,x,z), two units per FE cell. Physical xyz maps to neural coordinates as:

```text
(2*(y-ny/2), 2*(x-nx/2), 2*(z-nz/2))
```

Optimization samples FE cell centers, consistent with the paper's stated sampling scheme. Original notebook linspace endpoints were replaced to avoid assigning cell centers to domain boundaries. Geometry losses use fresh uniformly jittered samples within those cells.

Forced void/solid masks enter the FE density, the volume fraction, the accessibility field, and the final rendering. The original no-design notebook masked only FE while counting raw neural density for volume/accessibility; that mismatch is repaired. A field penalty also teaches the continuous representation to respect the fixed regions. Curvature/seam/alignment optimization omits fixed cells.

The accessibility multiplier is updated as a detached numerical scalar. This avoids retaining a computation graph across optimization iterations; it differs from the notebook's in-tape augmented-multiplier assignment and is documented rather than claimed to preserve its exact gradient.

## Feature scale controls

`global_frequency` sets the initial spectral extent. `global_scale` multiplies all input coordinates; larger values permit finer topology and more seams, while smaller values favor larger features. `field_scales` gives three independent [y,x,z] anisotropic scale vectors. For the paper's targeted z-scale example, use [[1,1,1],[1,1,0.5],[1,1,0.5]].

Kernels remain trainable. These settings influence feature complexity but are not certified feature-size bounds. The FE grid limits what stiffness/accessibility can resolve, even if a very fine render looks detailed.

## FE and limitations

The solver uses isotropic unit H8 hexahedra, unit-cell Young's modulus 1, Poisson ratio 0.3, minimum modulus 1e-6, and SIMP interpolation. Forces at duplicate nodes are summed by actual DOF indices. A rigid-motion rank check rejects underconstrained supports, and a residual check rejects unconverged solves rather than silently using invalid sensitivities.

The CPU backend uses SciPy sparse direct solves; the CuPy backend uses GPU sparse conjugate gradients with diagonal preconditioning. The sensitivity is the exact SIMP adjoint -p*(E0-Emin)*rho**(p-1)*element_strain_energy. Automatic differentiation passes this through the neural model.

Binary evaluation thresholds the FE density at 0.5 with the same minimum modulus. Continuous and binary compliance can differ substantially. A visually smooth supersampled surface is not independently certified structurally; its FE density and residual evidence are saved separately.
