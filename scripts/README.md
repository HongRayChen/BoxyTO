# Validation commands

Use an existing environment; scripts do not install dependencies or restart failed cases.

```bash
python scripts/validate_gpu.py --gpu 0 --output-root results/new_validation
python scripts/validate_gpu.py --gpu 1 --output-root results/new_validation
```

Run these in separate terminals to use both physical GPUs. Each worker sets CUDA_VISIBLE_DEVICES and executes its seven assigned examples sequentially. The default matrix runs 400 epochs per example. `--epochs 80` is a smoke run and should not be treated as the full validation.

The output root must have no prior case folders/logs. Logs use exclusive creation, and the optimizer refuses nonempty output directories. Use a new root to repeat the matrix. The report for the completed authors' workstation run is [VALIDATION.md](../docs/VALIDATION.md).

Run numerical/visual unit checks with `BOXYTO_TEST_GPU=1 python -m unittest discover -s tests -v`; omit that variable to skip CUDA checks. Your existing PyTorch/CuPy environment must be able to find its matching CUDA runtime libraries; see the environment versions in [VALIDATION.md](../docs/VALIDATION.md).
