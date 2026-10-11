# Completed validation — September 29, 2026

All 14 configurations completed 400 epochs on the two f80 GPUs. Both validation workers exited normally, and the GPUs were idle at final inspection. The final test suite passed all 34 tests, including GPU FE/field parity and real PyVista screenshot/color regressions.

## Environment and numerical checks

Existing workstation environment: Python 3.11, Torch 2.11.0+cu130, CuPy 14.0.1, NumPy 2.4.3, SciPy 1.17.1, PyVista 0.47.3, and two NVIDIA RTX PRO 6000 Blackwell GPUs. A process-only CUDA 13 library path was used; no packages were installed and no environment was created or replaced.

Checks include independent Gaussian-integration H8 stiffness, six rigid modes, asymmetric node/element orientation, DOF load summation, constraint stability, CPU/GPU equilibrium and energy, adjoint finite differences, Torch double gradcheck, analytic spatial/parameter derivative checks, mask orientation/gradients, infeasible region rejection, and real offscreen meshes/images. The unchanged original TensorFlow field methods agreed with the port within 4.33e-15 in a float64 CPU reference probe at nonzero weights. TensorFlow is not a runtime dependency.

Every saved run was checked for 400 finite history rows, finite bounded density with the expected FE shape, summary/density volume agreement, exact prescribed mask values, nonempty checkpoints/meshes, and valid 1500×1500 figures.

Maximum training FE residual: 9.9997e-09. Maximum final continuous residual: 9.9891e-09. Maximum binary residual: 2.8525e-08; this is below the solver’s explicitly checked residual acceptance bound of 10×rtol=1e-7. Maximum absolute continuous volume-target error: 0.003814.

## Full experiments

GPU numbers below are physical f80 GPUs. Each process uses its selected physical GPU as local cuda:0. Compliance is dimensionless for unit modulus and unit cells. Binary compliance uses density >= 0.5 with Emin=1e-6. Optimization runtime excludes rendering.

| Example | GPU | Target V | Final V | Continuous c | Binary c | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- |
| mbb_3axis | 0 | 0.300 | 0.301990 | 29.8196 | 18.5737 | 193.2 |
| mbb_5axis | 1 | 0.300 | 0.301040 | 26.4681 | 20.5770 | 194.6 |
| mount_3axis | 1 | 0.200 | 0.203058 | 1.6084 | 0.8331 | 143.4 |
| mount_5axis | 0 | 0.200 | 0.202097 | 1.5442 | 315.3064 | 134.3 |
| pedal_3axis | 0 | 0.200 | 0.200855 | 48.5350 | 3076.5651 | 185.3 |
| pedal_5axis | 1 | 0.200 | 0.200410 | 43.2983 | 2238.1889 | 196.1 |
| link_3axis | 1 | 0.250 | 0.252614 | 16.2648 | 3639.2473 | 156.1 |
| link_5axis | 0 | 0.250 | 0.252494 | 13.5507 | 798.3516 | 159.6 |
| random_single | 0 | 0.100 | 0.099778 | 4.9271 | 4.7550 | 284.4 |
| random_two_sided | 1 | 0.100 | 0.100386 | 26.7228 | 8896.8445 | 499.4 |
| global_scale_coarse | 0 | 0.300 | 0.300704 | 24.8065 | 20.4737 | 155.7 |
| local_z_scale | 1 | 0.300 | 0.301851 | 25.0976 | 20.4730 | 196.3 |
| no_design | 0 | 0.300 | 0.299858 | 33.3641 | 23.0369 | 183.7 |
| custom | 1 | 0.300 | 0.303814 | 116.8386 | 62.1934 | 74.0 |

The global-scale 1.0 example is the identical configuration to mbb_5axis and is validated by that run, avoiding a duplicate experiment. Random cases use boundary seeds 300 and 301 respectively, both with optimization seed 42; they are different boundary problems, so the table does not isolate a causal one-versus-two-direction effect.

The no-design sphere contains 136 FE cells; all saved densities there equal zero. The custom example prescribes eight solid FE cells; all equal one. These values were verified in density.npz, not inferred from image appearance.

## Interpretation and limitations

The source mathematics and executable workflows pass the stated checks. This is finite validation, not a guarantee that arbitrary configurations have no bugs. The port intentionally repairs voxel-center sampling and fixed-region bookkeeping; see [METHOD.md](METHOD.md). It does not claim exact reproduction of the paper’s numerical table.

Binary fixture attachment is a substantive modeling limitation in several original-style examples. In mount_5axis, 14 of 369 loaded nodes have no neighboring FE cell at density >0.5; continuous compliance is 1.5442 while binary compliance is 315.3064. Pedal/link and random_two_sided also show large continuous-to-binary increases. The small Emin lets these binary equilibria be solved; solver convergence alone does not establish a useful solid design. Prescribe physical solid fixture/load regions and re-evaluate the resulting geometry before manufacturing or assigning binary structural performance. The custom example demonstrates fixed-solid material; it does not guarantee every arbitrary fixture layout.

Paper-style cosmetic collars are separately labeled previews. They neither repair binary contacts nor enter compliance, saved FE density, or the exported exact mesh. The exact comparison plate deliberately omits them. Feature scaling is a representation control and supplies no strict minimum-size guarantee; sampled accessibility is not full CAM collision verification.

## Evidence and figures

- [Full regression output](../results/validation/tests.log)
- GPU workers and per-case stdout: ../results/validation/gpu{0,1}.log and <example>.log
- Saved config, boundary points, density, history, checkpoint, summary, and exact mesh: ../results/validation/<example>/
- [Exact paper-case comparison](../results/validation/paper_cases.png)
- [Explicit cosmetic paper previews](../results/validation/paper_previews.png)
- [Feature-scale comparison](../results/validation/feature_scale.png)
- [Visual review and figure controls](VISUALIZATION.md)

The short smoke runs, two aborted compatibility probes, camera preview, Python bytecode caches, and redundant extracted paper assets were removed after validation. All 14 full-run outputs, logs, comparison figures, and demo source files are included in this repository, together with the supplied paper PDF. The original paper-source ZIP and research notebooks remain in the authors' workstation project and are not bundled here.

## Publication verification — October 10, 2026

The public checkout passed all 34 existing regression tests, including CUDA/CuPy parity and actual offscreen rendering, using the existing workstation environment. See [publication test output](publication-tests.log). The saved 400-epoch experiments above were retained unchanged; this publication check did not rerun them.
