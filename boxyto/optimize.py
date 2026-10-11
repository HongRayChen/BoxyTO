"""Optimization, configuration, and reproducible saved outputs."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .boundary import Grid, custom_case, paper_case, random_case
from .fea import Top3dFE
from .fields import TrimmedNeuralField, shadows


@dataclass
class Options:
    mode: str = "5axis"
    epochs: int = 400
    seed: int = 42
    learning_rate: float = 0.002
    device: str = "cuda:0"
    fe_backend: str = "cpu"
    fe_rtol: float = 1e-8
    fe_maxiter: int = 20000
    global_frequency: float = 0.6
    global_scale: float = 1.0
    field_scales: tuple = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    bands: int = 4
    directions: tuple = ("z-",)
    tool_axis_xyz: tuple = (0, 0, 1)
    curvature_weight: float = 1000.0
    intersection_weight: float = 1000.0
    alignment_weight: float = 1000.0
    regularization_weight: float = 10.0
    mask_weight: float = 100.0
    accessibility_start: int = 20
    curvature_start: int = 40
    intersection_start: int = 60
    alignment_start: int = 20
    ramp_epochs: int = 300
    log_every: int = 20
    render_samples: int = 4
    render_size: int = 1500

    def validate(self):
        if self.mode not in ("compliance", "accessibility", "3axis", "5axis"):
            raise ValueError("mode must be compliance, accessibility, 3axis, or 5axis")
        for key in ("epochs", "bands", "ramp_epochs", "log_every", "render_samples", "render_size", "fe_maxiter"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "global_frequency", "global_scale", "fe_rtol"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "curvature_weight", "intersection_weight", "alignment_weight", "regularization_weight", "mask_weight",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        scale = np.asarray(self.field_scales, dtype=float)
        if scale.shape != (3, 3) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("field_scales must be positive 3x3 values in neural y,x,z order")
        axis = np.asarray(self.tool_axis_xyz, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError("tool_axis_xyz must be a nonzero finite xyz vector")
        if not self.directions or any(d not in ("x+", "x-", "y+", "y-", "z+", "z-") for d in self.directions):
            raise ValueError("directions must contain at least one named axis direction")
        if self.fe_backend not in ("cpu", "cupy"):
            raise ValueError("fe_backend must be cpu or cupy")


def read_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) - {"problem", "optimization", "description"}:
        raise ValueError("Unknown top-level configuration keys")
    return config


def build_problem(config):
    p = dict(config["problem"])
    kind = p.pop("type", "paper")
    if kind == "paper":
        spec = paper_case(p.pop("name"), dims=p.pop("dims", None))
    elif kind == "random":
        spec = random_case(Grid(*p.pop("dims")), seed=p.pop("seed", 300))
    elif kind == "custom":
        spec = custom_case(p)
        p = {}
    else:
        raise ValueError(f"Unknown problem type {kind}")
    if "volume_fraction" in p:
        spec.volume_fraction = float(p.pop("volume_fraction"))
    regions = p.pop("regions", [])
    # Region parsing is shared with custom problems, so masks follow one coordinate convention.
    if regions:
        custom = custom_case({
            "dims": list(spec.grid.dims),
            "volume_fraction": spec.volume_fraction,
            "loads": [{"points": spec.bc.loads.tolist(), "vectors": spec.bc.forces.tolist()}],
            "supports": [{"points": spec.bc.supports.tolist(), "components": spec.bc.constrained.tolist()}],
            "regions": regions,
        })
        spec.void_mask = custom.void_mask
        spec.solid_mask = custom.solid_mask
        spec.metadata["regions"] = regions
    if p:
        raise ValueError(f"Unknown problem keys: {sorted(p)}")
    if not 0 < spec.volume_fraction < 1:
        raise ValueError("volume_fraction must lie strictly between zero and one")
    if not np.any(~(spec.void_mask | spec.solid_mask)):
        raise ValueError("At least one design cell is required")
    if np.mean(spec.solid_mask) >= spec.volume_fraction or 1 - np.mean(spec.void_mask) <= spec.volume_fraction:
        raise ValueError("Target volume fraction is incompatible with fixed regions")
    return spec


def field_coordinates(dims, samples=1):
    """FE cell centers, in neural (y,x,z) coordinates; density rows descend in physical y, two units per cell."""
    nx, ny, nz = dims
    y, x, z = np.meshgrid(
        (np.arange(ny*samples)+0.5)/samples,
        (np.arange(nx*samples)+0.5)/samples,
        (np.arange(nz*samples)+0.5)/samples, indexing="ij",
    )
    return np.stack((-2*(y-ny/2), 2*(x-nx/2), 2*(z-nz/2)), axis=-1).reshape(-1, 3)


def physical_density(raw, spec):
    void = torch.as_tensor(spec.void_mask, dtype=torch.bool, device=raw.device)
    solid = torch.as_tensor(spec.solid_mask, dtype=torch.bool, device=raw.device)
    return torch.where(solid, torch.ones_like(raw), torch.where(void, torch.zeros_like(raw), raw))


def _ramp(epoch, start, length):
    return max(0.0, min((epoch-start)/length, 1.0))


def optimize(config, output, options_override=None, render=True):
    spec = build_problem(config)
    args = dict(config.get("optimization", {}))
    args.update(options_override or {})
    known = {f.name for f in fields(Options)}
    if set(args) - known:
        raise ValueError(f"Unknown optimization keys: {sorted(set(args)-known)}")
    opts = Options(**args)
    opts.validate()
    device = torch.device(opts.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    torch.set_num_threads(4)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    resolved = {"problem": config["problem"], "optimization": asdict(opts)}
    (out / "config.json").write_text(json.dumps(resolved, indent=2)+"\n")
    scales = np.asarray(opts.field_scales)*opts.global_scale
    model = TrimmedNeuralField(max(spec.grid.dims), opts.global_frequency, opts.bands, scales).to(device)
    fe = Top3dFE(
        spec.grid, spec.bc, backend=opts.fe_backend, rtol=opts.fe_rtol, maxiter=opts.fe_maxiter,
        gpu_device=device.index if device.type == "cuda" else None,
    )
    coords = torch.as_tensor(field_coordinates(spec.grid.dims), dtype=torch.float32, device=device)
    design = torch.as_tensor(~(spec.void_mask | spec.solid_mask), device=device).reshape(-1)
    fixed_void = torch.as_tensor(spec.void_mask, device=device).reshape(-1)
    fixed_solid = torch.as_tensor(spec.solid_mask, device=device).reshape(-1)
    axis = np.asarray(opts.tool_axis_xyz, dtype=float)
    axis /= np.linalg.norm(axis)
    # Physical XYZ -> neural (Y,X,Z).
    axis = (axis[1], axis[0], axis[2])
    initial = np.full(spec.shape, spec.volume_fraction)
    initial[spec.void_mask] = 0
    initial[spec.solid_mask] = 1
    c0 = fe.evaluate(initial, penal=2.0).compliance
    adam = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, eps=1e-7)
    history = []
    multiplier = 1.0
    started = time.perf_counter()
    for epoch in range(1, opts.epochs+1):
        adam.zero_grad(set_to_none=True)
        raw = model.density(coords).reshape(spec.shape)
        rho = physical_density(raw, spec)
        penal = min(2.0+0.01*epoch, 4.0)
        alpha = min(1.0+0.5*epoch, 100.0)
        compliance = fe.torch_compliance(rho, penal=penal)
        vf = rho.mean()
        accessibility = curvature = intersection = alignment = mask_loss = torch.zeros((), device=device)
        sv = _ramp(epoch, opts.accessibility_start, opts.ramp_epochs)
        gc = _ramp(epoch, opts.curvature_start, opts.ramp_epochs)
        ic = _ramp(epoch, opts.intersection_start, opts.ramp_epochs)
        ac = _ramp(epoch, opts.alignment_start, opts.ramp_epochs)
        if opts.mode != "compliance" and sv > 0:
            accessibility = shadows(rho, opts.directions).mean()
            multiplier += sv*float(accessibility.detach())
        if opts.mode in ("3axis", "5axis") and (gc > 0 or ic > 0 or (opts.mode == "3axis" and ac > 0)):
            points = (coords + torch.rand_like(coords)*2-1)[design]
            geom = model.geometry_losses(
                points, ic*80 if ic > 0 and opts.intersection_weight > 0 else None,
                ac*90 if opts.mode == "3axis" and ac > 0 and opts.alignment_weight > 0 else None,
                axis,
            )
            if gc > 0 and opts.curvature_weight > 0:
                curvature = geom["curvature"]
            intersection = geom["intersection"]
            alignment = geom["alignment"]
        raw_flat = raw.reshape(-1)
        if fixed_void.any():
            mask_loss = mask_loss + raw_flat[fixed_void].mean()
        if fixed_solid.any():
            mask_loss = mask_loss + (1-raw_flat[fixed_solid]).mean()
        loss = (compliance/c0 + alpha*(vf/spec.volume_fraction-1).square()
                + opts.regularization_weight*model.regularization()
                + gc*opts.curvature_weight*curvature
                + opts.intersection_weight*intersection
                + opts.alignment_weight*alignment
                + sv*accessibility.square() + multiplier*accessibility
                + opts.mask_weight*mask_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite or missing parameter gradients at epoch {epoch}")
        adam.step()
        row = dict(epoch=epoch, compliance=float(compliance.detach()), volume_fraction=float(vf.detach()),
                   objective=float(loss.detach()), curvature=float(curvature.detach()),
                   intersection=float(intersection.detach()),
                   alignment=float(alignment.detach()), accessibility=float(accessibility.detach()),
                   mask_loss=float(mask_loss.detach()),
                   fe_residual=fe.last_result.residual, fe_iterations=fe.last_result.iterations, penal=penal)
        history.append(row)
        if epoch == 1 or epoch % opts.log_every == 0 or epoch == opts.epochs:
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        raw = model.density(coords).reshape(spec.shape)
        rho = physical_density(raw, spec).cpu().numpy()
    final = fe.evaluate(rho, penal=min(2+0.01*opts.epochs, 4))
    binary = (rho >= 0.5).astype(float)
    binary_fe = fe.evaluate(binary, penal=4.0)
    summary = {
        "case": spec.metadata.get("case", config["problem"].get("name", config["problem"]["type"])),
        "mode": opts.mode, "seed": opts.seed, "device": str(device), "fe_backend": opts.fe_backend,
        "dims_xyz": list(spec.grid.dims), "epochs": opts.epochs,
        "target_volume_fraction": spec.volume_fraction,
        "volume_fraction": float(rho.mean()), "binary_volume_fraction": float(binary.mean()),
        "initial_normalization_compliance": c0, "compliance": final.compliance,
        "binary_compliance": binary_fe.compliance,
        "fe_residual": final.residual, "binary_fe_residual": binary_fe.residual,
        "max_training_fe_residual": max(row["fe_residual"] for row in history),
        "void_max_density": float(rho[spec.void_mask].max()) if spec.void_mask.any() else None,
        "solid_min_density": float(rho[spec.solid_mask].min()) if spec.solid_mask.any() else None,
        "runtime_seconds": time.perf_counter()-started,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    with (out/"history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(out/"density.npz", density=rho, void_mask=spec.void_mask, solid_mask=spec.solid_mask)
    torch.save({"state_dict": model.state_dict(), "options": asdict(opts), "config": resolved}, out/"model.pt")
    (out/"boundary.json").write_text(json.dumps({
        "dims": list(spec.grid.dims), "loads_points_xyz": spec.bc.loads.tolist(),
        "load_vectors_xyz": spec.bc.forces.tolist(), "fixed_points_xyz": spec.bc.supports.tolist(),
        "constrained_components": spec.bc.constrained.tolist(), "metadata": spec.metadata,
    }, indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if render:
        render_run(out, device=opts.device)
    print(json.dumps({"state": "complete", **summary}), flush=True)
    return summary


def render_run(output, device="cpu", samples=None, size=None):
    """Render saved neural fields at arbitrary resolution; FE results remain on the FE grid."""
    from .render import render_boundary_conditions, render_density
    out = Path(output)
    saved = torch.load(out/"model.pt", map_location=device, weights_only=False)
    opts = Options(**saved["options"])
    spec = build_problem(saved["config"])
    model = TrimmedNeuralField(
        max(spec.grid.dims), opts.global_frequency, opts.bands, np.asarray(opts.field_scales)*opts.global_scale,
    ).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    ss = samples if samples is not None else opts.render_samples
    image_size = size if size is not None else opts.render_size
    if ss < 1 or image_size < 1:
        raise ValueError("Render samples and size must be positive")
    def evaluate(coords, curvature=False):
        result = []
        with torch.no_grad():
            for chunk in np.array_split(coords, max(1, (len(coords)+2047)//2048)):
                tensor = torch.as_tensor(chunk.copy(), dtype=torch.float32, device=device)
                value = model.gaussian_curvature(tensor).abs() if curvature else model.density(tensor)
                result.append(value.cpu().numpy().reshape(-1))
        return np.concatenate(result)
    rho = evaluate(field_coordinates(spec.grid.dims, ss)).reshape(tuple(n*ss for n in spec.shape))
    # Fixed FE-cell masks repeat onto the high-resolution render grid, exactly.
    for original, value in ((spec.void_mask, 0.0), (spec.solid_mask, 1.0)):
        mask = original
        for axis in range(3):
            mask = np.repeat(mask, ss, axis=axis)
        rho[mask] = value
    bc = json.loads((out/"boundary.json").read_text())
    def curvature(xyz):
        nx, ny, nz = spec.grid.dims
        coords = np.stack((2*(xyz[:,1]-ny/2), 2*(xyz[:,0]-nx/2), 2*(xyz[:,2]-nz/2)), axis=1)
        return evaluate(coords, curvature=True)
    kwargs = {"extent": spec.grid.dims, "size": (image_size,image_size)}
    report = render_density(rho, out/"curvature.png", curvature=curvature, mesh_output=out/"mesh.vtp", **kwargs)
    render_density(rho, out/"geometry.png", **kwargs)
    collars = spec.metadata.get("render_only_cylinders")
    if collars:
        render_density(rho, out/"paper.png", curvature=curvature, cylinders=collars, **kwargs)
    render_boundary_conditions(spec.grid.dims, bc, out/"boundary.png", size=(image_size,image_size))
    (out/"render_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
