# Example configurations

Run from the repository root with `python -m boxyto optimize examples/NAME.json --output results/NEW_NAME`. Use `--device cpu` on CPU machines or `--fe-backend cupy` for GPU FE. Each output folder must be new or empty.

Paper cases use `mbb`, `mount`, `pedal`, and `link`. The `3axis` files enable fixed-cutter alignment and scale z of fields 2/3 by 0.5; the `5axis` files leave orientation free while enforcing developability. Defaults run 400 epochs.

`global_scale_coarse` and `global_scale_fine` set uniform coordinate scale 0.5 and 1.0. The fine example is identical to `mbb_5axis`; the validation matrix runs it once under the paper-case name. `local_z_scale` scales only z of fields 2/3.

`random_single` and `random_two_sided` use seeds 300/301 with one/two z access directions and target volume 0.1. Saved boundary points are authoritative; old notebook seed mapping is not promised.

`no_design` prescribes a void sphere. `custom` shows distributed loads, component constraints, and a fixed-solid load pad. Use fixed-solid regions when attachment to load/support fixtures must survive density thresholding; paper-style cosmetic collars do not supply structural material.

See [boundary-condition schema](../docs/BOUNDARY_CONDITIONS.md), [method](../docs/METHOD.md), and [validation](../docs/VALIDATION.md).
