"""Faithful PyTorch port of the notebooks' three trimmed cosine fields.

Spatial coordinates and scale vectors use historical (y,x,z) order, in
centered notebook units with two units per voxel. The optimizer owns the
coordinate grid and no-design-region masks.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn

# Signs retain the original notebook's ray traversal convention.
DIRECTION_AXES = {
    "y+": (0, True), "y-": (0, False), "x+": (1, False),
    "x-": (1, True), "z+": (2, False), "z-": (2, True),
}


def adjugate(matrix: Tensor) -> Tensor:
    """Polynomial adjugate; valid for singular 3x3 Hessians too."""
    return torch.stack(
        (torch.linalg.cross(matrix[..., 1, :], matrix[..., 2, :]),
         torch.linalg.cross(matrix[..., 2, :], matrix[..., 0, :]),
         torch.linalg.cross(matrix[..., 0, :], matrix[..., 1, :])), dim=-1)


def shadows(density: Tensor, directions: Sequence[str | int]) -> Tensor:
    """Smooth void inaccessible from all selected axis-aligned tool directions.

    Uses the original product of sigmoid(10*(cumsum(rho)-.5))*(1-rho).
    A six-item zero/one flag sequence uses y+,y-,x+,x-,z+,z- order.
    This is the notebook accessibility surrogate, not a finite-tool collision
    checker or a guarantee of manufacturability.
    """
    if density.ndim != 3:
        raise ValueError("density must have shape (nely,nelx,nelz)")
    directions = tuple(directions)
    if len(directions) == 6 and all(isinstance(x, (int, bool, np.integer)) for x in directions):
        if any(int(x) not in (0, 1) for x in directions):
            raise ValueError("direction flags must be zero or one")
        directions = tuple(name for name, flag in zip(DIRECTION_AXES, directions) if flag)
    if not directions:
        raise ValueError("at least one machining direction is required")
    if len(set(directions)) != len(directions):
        raise ValueError("machining directions must be unique")
    result = torch.ones_like(density)
    for direction in directions:
        if direction not in DIRECTION_AXES:
            raise ValueError(f"unknown machining direction: {direction!r}")
        axis, reverse = DIRECTION_AXES[direction]
        ray = density.flip((axis,)) if reverse else density
        cumulative = ray.cumsum(axis)
        if reverse:
            cumulative = cumulative.flip((axis,))
        result = result * torch.sigmoid(10.0 * (cumulative - 0.5)) * (1.0 - density)
    return result


machining_shadow = shadows


def _curvature(gradient: Tensor, hessian: Tensor) -> Tensor:
    numerator = torch.einsum("ni,nij,nj->n", gradient, adjugate(hessian), gradient)
    return (numerator / (gradient.square().sum(dim=1).square() + 0.0001))[:, None]


def _seams(logits: Tensor) -> Tensor:
    gaussians = torch.exp(-0.5 * logits.square())
    pairs = gaussians[:, 0] * gaussians[:, 1] + gaussians[:, 0] * gaussians[:, 2] + gaussians[:, 1] * gaussians[:, 2]
    return torch.sigmoid(20.0 * (pairs - 0.2))[:, None]


def _intersection(gradients: list[Tensor], angle: float) -> Tensor:
    result = torch.zeros_like(gradients[0][:, :1])
    cutoff = math.cos(float(angle) * 0.0174533)  # Original notebook degree conversion.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        first, second = gradients[i], gradients[j]
        nfirst = first / torch.sqrt(first.square().sum(dim=1, keepdim=True) + 0.01)
        nsecond = second / torch.sqrt(second.square().sum(dim=1, keepdim=True) + 0.01)
        dot = (first * second).sum(dim=1, keepdim=True)
        ndot = (nfirst * nsecond).sum(dim=1, keepdim=True).abs()
        result = result + (torch.sigmoid(20.0 * (ndot - cutoff - 0.2)) * dot).abs()
    return result


def _alignment(gradient: Tensor, angle: float, direction: Sequence[float]) -> Tensor:
    axis = torch.as_tensor(direction, dtype=gradient.dtype, device=gradient.device).reshape(-1)
    if axis.shape != (3,) or not torch.isfinite(axis).all() or axis.norm() == 0:
        raise ValueError("direction must be a nonzero finite (y,x,z) vector")
    axis = axis / axis.norm()
    normal = gradient / torch.sqrt(gradient.square().sum(dim=1, keepdim=True) + 0.001)
    cosine = (gradient * axis).sum(dim=1, keepdim=True).abs()
    ncosine = (normal * axis).sum(dim=1, keepdim=True).abs()
    sine = torch.sqrt(torch.linalg.cross(gradient, axis.expand_as(gradient)).square().sum(dim=1, keepdim=True) + 0.0001)
    nsine = torch.sqrt(torch.linalg.cross(normal, axis.expand_as(normal)).square().sum(dim=1, keepdim=True) + 0.0001)
    cutoff = math.cos(float(angle) * 0.0174533)  # Original notebook degree conversion.
    return torch.sigmoid(20.0 * (ncosine - cutoff)) * cosine * torch.sigmoid(20.0 * (nsine - cutoff)) * sine


class TrimmedNeuralField(nn.Module):
    """Product of three sigmoid cosine fields with trainable Fourier kernels.

    global_frequency is the initial spectral extent (high_band); kernels
    remain trainable. field_scales supplies three per-field (y,x,z) coordinate
    scales (scalingX1,2,3), controlling feature sizes through anisotropic
    trimming. These are representation controls, not strict feature bounds.
    """
    def __init__(self, nelm: int | float, global_frequency: float = 0.6,
                 bands: int = 4, field_scales: Sequence[Sequence[float]] | None = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if not math.isfinite(float(nelm)) or nelm <= 0:
            raise ValueError("nelm must be finite and positive")
        if not math.isfinite(float(global_frequency)) or global_frequency <= 0:
            raise ValueError("global_frequency must be finite and positive")
        if isinstance(bands, bool) or not isinstance(bands, int) or bands < 2:
            raise ValueError("bands must be an integer of at least two")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.nelm = float(nelm)
        self.global_frequency = float(global_frequency)
        self.bands = bands
        # Preserve duplicate zero frequencies in the original initialization.
        axis = np.linspace([-global_frequency, 0.0], [0.0, global_frequency], bands).flatten()
        grids = np.meshgrid(axis, axis, axis, indexing="ij")
        kernel = np.stack([x.reshape(-1) for x in grids], axis=0)
        kernels = np.stack([kernel, kernel * 0.95, kernel * 0.9], axis=0)
        self.kernels = nn.Parameter(torch.as_tensor(kernels, dtype=dtype))
        self.weights = nn.Parameter(torch.zeros((3, kernel.shape[1], 1), dtype=dtype))
        scales = torch.ones((3, 3), dtype=dtype) if field_scales is None else torch.as_tensor(field_scales, dtype=dtype)
        if scales.shape != (3, 3) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("field_scales must contain three positive finite (y,x,z) scales")
        self.register_buffer("field_scales", scales.clone())

    def _coords(self, coords: Tensor) -> Tensor:
        coords = torch.as_tensor(coords, dtype=self.kernels.dtype, device=self.kernels.device)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N,3), ordered (y,x,z)")
        return coords

    def _basis(self, field: int) -> Tensor:
        return self.kernels[field] * self.field_scales[field, :, None] * (50.0 / (2.0 * self.nelm))

    def logits(self, coords: Tensor) -> Tensor:
        coords = self._coords(coords)
        return torch.cat([torch.cos(coords @ self._basis(i) + 1.0) @ self.weights[i] for i in range(3)], dim=1)

    def components(self, coords: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(coords))

    def density(self, coords: Tensor) -> Tensor:
        components = self.components(coords)
        return components[:, 0:1] * components[:, 1:2] * components[:, 2:3]

    def forward(self, coords: Tensor) -> Tensor:
        return self.density(coords)

    def _evaluate(self, coords: Tensor, second: bool = True):
        coords = self._coords(coords)
        logits, values, gradients, hessians = [], [], [], []
        for i in range(3):
            basis = self._basis(i)
            phase = coords @ basis + 1.0
            cosine = torch.cos(phase)
            weights = self.weights[i, :, 0]
            logit = cosine @ weights
            dlogit = -(torch.sin(phase) * weights) @ basis.T
            value = torch.sigmoid(logit)
            slope = value * (1.0 - value)
            logits.append(logit)
            values.append(value)
            gradients.append(slope[:, None] * dlogit)
            if second:
                hlogit = -torch.einsum("nm,am,bm->nab", cosine * weights, basis, basis)
                outer = dlogit[:, :, None] * dlogit[:, None, :]
                hessians.append(slope[:, None, None] * (hlogit + (1.0 - 2.0 * value[:, None, None]) * outer))
        return torch.stack(logits, dim=1), values, gradients, hessians

    @staticmethod
    def _product(values, gradients, hessians):
        density = (values[0] * values[1] * values[2])[:, None]
        gradient = sum(gradients[i] * (values[(i + 1) % 3] * values[(i + 2) % 3])[:, None] for i in range(3))
        if not hessians:
            return density, gradient, None
        hessian = sum(hessians[i] * (values[(i + 1) % 3] * values[(i + 2) % 3])[:, None, None] for i in range(3))
        for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            cross = gradients[i][:, :, None] * gradients[j][:, None, :]
            hessian = hessian + (cross + cross.transpose(-1, -2)) * values[k][:, None, None]
        return density, gradient, hessian

    def derivatives(self, coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Exact spatial derivatives, differentiable in model parameters."""
        _, values, gradients, hessians = self._evaluate(coords)
        return self._product(values, gradients, hessians)

    def gradient(self, coords: Tensor) -> Tensor:
        _, values, gradients, hessians = self._evaluate(coords, second=False)
        return self._product(values, gradients, hessians)[1]

    def hessian(self, coords: Tensor) -> Tensor:
        return self.derivatives(coords)[2]

    def gaussian_curvature(self, coords: Tensor) -> Tensor:
        _, gradient, hessian = self.derivatives(coords)
        return _curvature(gradient, hessian)

    def seams(self, coords: Tensor) -> Tensor:
        return _seams(self.logits(coords))

    def intersection(self, coords: Tensor, angle: float) -> Tensor:
        _, _, gradients, _ = self._evaluate(coords, second=False)
        return _intersection(gradients, angle)

    intersect = intersection

    def axis_alignment(self, coords: Tensor, angle: float,
                       direction: Sequence[float] = (0.0, 0.0, 1.0)) -> Tensor:
        """Fixed-axis normal penalty; use zero objective weight for five axes.

        For Cartesian axes this equals the original abs-gradient equation.
        The tool vector is normalized and treated as an unoriented axis.
        """
        return _alignment(self.gradient(coords), angle, direction)

    alignment = axis_alignment

    def geometry_losses(self, coords: Tensor, intersection_angle: float | None,
                        alignment_angle: float | None = None,
                        direction: Sequence[float] = (0.0, 0.0, 1.0)) -> dict[str, Tensor]:
        """Mean notebook geometry penalties with one shared derivative evaluation.

        Pass None for inactive intersection/alignment penalties. Curvature and
        alignment are masked by (1-seams), matching the objective notebook.
        """
        logits, values, gradients, hessians = self._evaluate(coords)
        _, gradient, hessian = self._product(values, gradients, hessians)
        seam_mask = 1.0 - _seams(logits)
        zero = gradient.sum() * 0.0
        return {
            "curvature": (_curvature(gradient, hessian).abs() * seam_mask).mean(),
            "intersection": _intersection(gradients, intersection_angle).mean() if intersection_angle is not None else zero,
            "alignment": (_alignment(gradient, alignment_angle, direction) * seam_mask).mean() if alignment_angle is not None else zero,
        }

    def regularization(self) -> Tensor:
        return self.weights.square().mean(dim=(1, 2)).sum()

