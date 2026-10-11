"""Bundled Top3d-style H8 elasticity, SIMP compliance, and adjoint gradients.

No pyFANTOM dependency is required. CPU SciPy is the reference implementation;
CuPy sparse CG is an optional GPU backend. Both use the identical node layout,
element stiffness, material interpolation, and checked solver residual.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

from .boundary import Grid, BoundaryConditions


def lk_h8(nu=.3):
    """Top3d unit-cube stiffness for E=1, 8 nodes x 3 displacement components.

    Closed-form coefficients from the original BoxyTO notebooks / Top3d.
    Node order is (000,100,110,010,001,101,111,011), in physical xyz.
    """
    if not np.isfinite(nu) or not -1 < nu < .5:
        raise ValueError("Poisson ratio must lie strictly between -1 and 0.5")
    a = np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],
                  [-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]], dtype=float)
    k = a.T @ np.array([1., nu]) / 144.
    k1 = k[[0,1,1,2,4,4,1,0,1,3,5,6,1,1,0,3,6,5,2,3,3,0,7,7,4,5,6,7,0,1,4,6,5,7,1,0]].reshape(6,6)
    k2 = k[[8,7,11,5,3,6,7,8,11,4,2,4,9,9,12,6,3,5,5,4,10,8,1,9,3,2,4,1,8,11,10,3,5,11,9,12]].reshape(6,6)
    k3 = k[[5,6,3,8,11,7,6,5,3,9,12,9,4,4,2,7,11,8,8,9,1,5,10,4,11,12,9,10,5,3,1,11,8,3,4,2]].reshape(6,6)
    k4 = k[[13,10,10,12,9,9,10,13,10,11,8,7,10,10,13,11,7,8,12,11,11,13,6,6,9,8,7,6,13,10,9,7,8,6,10,13]].reshape(6,6)
    k5 = k[[0,1,7,2,4,3,1,0,7,3,5,10,7,7,0,4,10,5,2,3,4,0,7,1,4,5,10,7,0,7,3,10,5,1,7,0]].reshape(6,6)
    k6 = k[[13,10,6,12,9,11,10,13,6,11,8,1,6,6,13,9,1,8,12,11,9,13,6,10,9,8,1,6,13,6,11,1,8,10,6,13]].reshape(6,6)
    return np.block([[k1,k2,k3,k4], [k2.T,k5,k6,k3.T],
                     [k3.T,k6,k5.T,k2.T], [k4,k3,k2,k1]]) / ((1+nu)*(1-2*nu))


def element_dofs(grid: Grid):
    """Integer Top3d DOFs, element order matching density.flatten(order='F')."""
    row, x, z = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), np.arange(grid.nz), indexing="ij")
    first = (z*(grid.nx+1)*(grid.ny+1) + x*(grid.ny+1) + row + 1).ravel(order="F")
    layer = (grid.nx+1)*(grid.ny+1)
    node_offsets = np.array([0, grid.ny+1, grid.ny, -1, layer, layer+grid.ny+1, layer+grid.ny, layer-1])
    nodes = first[:, None] + node_offsets
    dofs = (3*nodes[:, :, None] + np.arange(3)).reshape(grid.nele, 24)
    if grid.ndof > np.iinfo(np.int32).max:
        raise ValueError("Grid exceeds supported sparse integer index range")
    return dofs.astype(np.int32)


@dataclass
class FEResult:
    compliance: float
    gradient: np.ndarray
    displacement: np.ndarray
    residual: float
    iterations: int
    element_energy: np.ndarray
    backend: str


class SolverError(RuntimeError):
    """Failure of a finite-element solve or its numerical residual check."""


class Top3dFE:
    def __init__(self, grid: Grid, bc: BoundaryConditions, backend="cpu", E0=1., Emin=1e-6,
                 nu=.3, rtol=1e-8, maxiter=10000, solver="direct", gpu_device=None):
        if not isinstance(grid, Grid):
            grid = Grid(*grid)
        if backend not in ("cpu", "cupy"):
            raise ValueError("backend must be cpu or cupy")
        if solver not in ("direct", "cg"):
            raise ValueError("solver must be direct or cg")
        if not (np.isfinite(E0) and np.isfinite(Emin) and 0 < Emin < E0):
            raise ValueError("Material moduli must satisfy 0 < Emin < E0")
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("rtol must lie in (0,1)")
        if int(maxiter) != maxiter or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        self.grid, self.bc, self.backend = grid, bc, backend
        self.E0, self.Emin, self.nu = float(E0), float(Emin), float(nu)
        self.rtol, self.maxiter, self.solver = float(rtol), int(maxiter), solver
        self.force, self.fixed = bc.dof_data(grid)
        self.free = np.setdiff1d(np.arange(grid.ndof), self.fixed)
        self.ke = lk_h8(nu)
        self.edof = element_dofs(grid)
        self.rows = np.repeat(self.edof, 24, axis=1).ravel()
        self.cols = np.tile(self.edof, (1,24)).ravel()
        self.last_result = None
        self._previous = None
        self._gpu = None
        if backend == "cupy":
            try:
                import cupy as cp
                import cupyx.scipy.sparse as csparse
                import cupyx.scipy.sparse.linalg as clinalg
            except ImportError as exc:
                raise ImportError("CuPy is needed only for backend='cupy'; use backend='cpu' otherwise") from exc
            self.gpu_device = cp.cuda.runtime.getDevice() if gpu_device is None else int(gpu_device)
            with cp.cuda.Device(self.gpu_device):
                self._gpu = (cp, csparse, clinalg)
                self._cg_tol_keyword = "rtol" if "rtol" in inspect.signature(clinalg.cg).parameters else "tol"
                self._grows = cp.asarray(self.rows); self._gcols = cp.asarray(self.cols)
                self._gedof = cp.asarray(self.edof); self._gke = cp.asarray(self.ke)
                self._gfree = cp.asarray(self.free); self._gforce = cp.asarray(self.force)
        else:
            self.gpu_device = None

    def _density(self, density, penal):
        if not np.isfinite(penal) or penal < 1:
            raise ValueError("SIMP penal must be finite and at least 1")
        if hasattr(density, "detach"):
            density = density.detach().cpu().numpy()
        a = np.asarray(density, dtype=np.float64)
        if a.shape != self.grid.shape:
            raise ValueError(f"density must have shape {self.grid.shape}, received {a.shape}")
        if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
            raise ValueError("density must be finite and lie within [0,1]")
        return a.ravel(order="F")

    def stiffness(self, density, penal=3.):
        """Assemble CPU global stiffness; useful for independent verification."""
        rho = self._density(density, penal)
        moduli = self.Emin + (self.E0-self.Emin)*rho**penal
        values = (moduli[:, None]*self.ke.ravel()[None, :]).ravel()
        k = sparse.coo_matrix((values, (self.rows, self.cols)), shape=(self.grid.ndof, self.grid.ndof)).tocsr()
        return k

    def evaluate(self, density, penal=3.):
        rho = self._density(density, penal)
        if self.backend == "cupy":
            with self._gpu[0].cuda.Device(self.gpu_device):
                result = self._evaluate_gpu(rho, penal)
        else:
            result = self._evaluate_cpu(rho, penal)
        self.last_result = result
        return result

    def _evaluate_cpu(self, rho, penal):
        moduli = self.Emin + (self.E0-self.Emin)*rho**penal
        values = (moduli[:, None]*self.ke.ravel()[None, :]).ravel()
        k = sparse.coo_matrix((values, (self.rows, self.cols)), shape=(self.grid.ndof,self.grid.ndof)).tocsr()
        kf = k[self.free][:,self.free]
        f = self.force[self.free]
        iterations = 0
        if self.solver == "direct":
            with warnings.catch_warnings():
                warnings.simplefilter("error", linalg.MatrixRankWarning)
                try:
                    uf = linalg.spsolve(kf.tocsc(), f)
                except linalg.MatrixRankWarning as exc:
                    raise SolverError("Singular stiffness matrix") from exc
            iterations = 1
        else:
            diag = kf.diagonal()
            if np.any(diag <= 0):
                raise SolverError("Nonpositive stiffness diagonal")
            pre = sparse.diags(1/diag)
            def callback(_):
                nonlocal iterations
                iterations += 1
            uf, info = linalg.cg(kf, f, x0=self._previous, rtol=self.rtol, atol=0,
                                maxiter=self.maxiter, M=pre, callback=callback)
            if info != 0:
                raise SolverError(f"CPU CG failed to converge: info={info}, iterations={iterations}")
        residual = float(np.linalg.norm(kf @ uf-f)/np.linalg.norm(f))
        self._check_solution(uf, residual)
        self._previous = uf
        u = np.zeros(self.grid.ndof); u[self.free] = uf
        ue = u[self.edof]; ce = np.einsum("ei,ij,ej->e",ue,self.ke,ue)
        compliance = float(self.force @ u)
        energy = float(moduli @ ce)
        if not np.isclose(compliance, energy, rtol=max(1e-6, 100*self.rtol), atol=1e-8):
            raise SolverError(f"Work/strain-energy mismatch: work={compliance}, energy={energy}")
        gradient = -penal*(self.E0-self.Emin)*rho**(penal-1)*ce
        return FEResult(compliance, gradient.reshape(self.grid.shape,order="F"), u,
                        residual, iterations, ce.reshape(self.grid.shape,order="F"), self.backend)

    def _evaluate_gpu(self, rho, penal):
        cp, csparse, clinalg = self._gpu
        grho = cp.asarray(rho)
        moduli = self.Emin + (self.E0-self.Emin)*grho**penal
        values = (moduli[:, None]*self._gke.ravel()[None, :]).ravel()
        k = csparse.coo_matrix((values, (self._grows,self._gcols)), shape=(self.grid.ndof,self.grid.ndof)).tocsr()
        kf = k[self._gfree][:,self._gfree]
        f = self._gforce[self._gfree]
        diag = kf.diagonal()
        if bool(cp.any(diag <= 0)):
            raise SolverError("Nonpositive GPU stiffness diagonal")
        pre = csparse.diags(1/diag)
        iterations = 0
        def callback(_):
            nonlocal iterations
            iterations += 1
        # CuPy 14 renamed tol to rtol; select once when the backend is initialized.
        uf, info = clinalg.cg(kf, f, x0=self._previous, atol=0, maxiter=self.maxiter,
                             M=pre, callback=callback, **{self._cg_tol_keyword: self.rtol})
        if info != 0:
            raise SolverError(f"GPU CG failed to converge: info={info}, iterations={iterations}")
        residual = float((cp.linalg.norm(kf @ uf-f)/cp.linalg.norm(f)).get())
        self._check_solution(cp.asnumpy(uf), residual)
        self._previous = uf
        u = cp.zeros(self.grid.ndof); u[self._gfree] = uf
        ue = u[self._gedof]; ce = cp.einsum("ei,ij,ej->e",ue,self._gke,ue)
        compliance = float((self._gforce @ u).get())
        energy = float((moduli @ ce).get())
        if not np.isclose(compliance, energy, rtol=max(1e-6,100*self.rtol), atol=1e-8):
            raise SolverError(f"GPU work/strain-energy mismatch: work={compliance}, energy={energy}")
        gradient = -penal*(self.E0-self.Emin)*grho**(penal-1)*ce
        return FEResult(compliance, cp.asnumpy(gradient).reshape(self.grid.shape,order="F"), cp.asnumpy(u),
                        residual, iterations, cp.asnumpy(ce).reshape(self.grid.shape,order="F"), self.backend)

    def _check_solution(self, displacement, residual):
        if not np.all(np.isfinite(displacement)) or not np.isfinite(residual):
            raise SolverError("Finite-element solution contains NaN or infinity")
        limit = max(10*self.rtol,1e-7)
        if residual > limit:
            raise SolverError(f"Relative equilibrium residual {residual:.3e} exceeds {limit:.3e}")

    def torch_compliance(self, density, penal=3.):
        """PyTorch scalar compliance with exact first-order adjoint sensitivity.

        The sparse solve uses float64. The returned scalar and derivative retain
        the input tensor's dtype and device. Higher derivatives are unsupported.
        """
        import torch
        from torch.autograd.function import once_differentiable
        if not isinstance(density, torch.Tensor) or not density.is_floating_point():
            raise TypeError("density must be a floating point Torch tensor")
        fe = self
        class _Compliance(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = fe.evaluate(x, penal=penal)
                grad = torch.as_tensor(result.gradient.copy(), dtype=x.dtype, device=x.device)
                ctx.save_for_backward(grad)
                return x.new_tensor(result.compliance)

            @staticmethod
            @once_differentiable
            def backward(ctx, upstream):
                (grad,) = ctx.saved_tensors
                return upstream * grad
        return _Compliance.apply(density)
