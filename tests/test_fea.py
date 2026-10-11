"""Independent H8 integration, equilibrium, finite differences, Torch, and CG."""
import os
import unittest
import numpy as np
from boxyto.boundary import Grid,paper_case,random_case
from boxyto.fea import Top3dFE,lk_h8,element_dofs,SolverError


def gauss_h8(nu):
    """Independent 2x2x2 integration; physical cube and engineering shear B."""
    nodes=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=float)
    signs=2*nodes-1
    lam=nu/((1+nu)*(1-2*nu)); mu=1/(2*(1+nu))
    d=np.zeros((6,6)); d[:3,:3]=lam
    d[np.arange(3),np.arange(3)]+=2*mu
    d[3:,3:]=mu*np.eye(3)
    k=np.zeros((24,24))
    for a in [-1/np.sqrt(3),1/np.sqrt(3)]:
        for b in [-1/np.sqrt(3),1/np.sqrt(3)]:
            for c in [-1/np.sqrt(3),1/np.sqrt(3)]:
                q=np.array([a,b,c])
                grad=np.empty((8,3))
                for axis in range(3):
                    other=[i for i in range(3) if i!=axis]
                    grad[:,axis]=signs[:,axis]*np.prod(1+signs[:,other]*q[other],axis=1)/4
                bm=np.zeros((6,24))
                for n,(dx,dy,dz) in enumerate(grad):
                    bm[0,3*n]=dx; bm[1,3*n+1]=dy; bm[2,3*n+2]=dz
                    bm[3,3*n:3*n+3]=[dy,dx,0]
                    bm[4,3*n:3*n+3]=[0,dz,dy]
                    bm[5,3*n:3*n+3]=[dz,0,dx]
                k+=bm.T@d@bm/8
    return k


class FiniteElementTests(unittest.TestCase):
    def test_element_stiffness_independent_integration(self):
        for nu in (.0,.2,.3,.45):
            k=lk_h8(nu)
            np.testing.assert_allclose(k,gauss_h8(nu),rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(k,k.T,atol=1e-15)
            eigen=np.linalg.eigvalsh(k)
            self.assertEqual(np.count_nonzero(np.abs(eigen)<1e-10),6)
            self.assertGreater(eigen[6],0)

    def test_element_node_orientation(self):
        grid=Grid(2,2,1)
        nodes=grid.nodes_xyz()
        first=element_dofs(grid)[0].reshape(8,3)[:,0]//3
        np.testing.assert_array_equal(nodes[first],[[0,1,0],[1,1,0],[1,2,0],[0,2,0],
                                                   [0,1,1],[1,1,1],[1,2,1],[0,2,1]])

    def test_equilibrium_energy_and_stiffness(self):
        p=paper_case("mbb",dims=(4,3,2))
        fe=Top3dFE(p.grid,p.bc,E0=2,Emin=1e-5)
        rho=np.random.default_rng(9).uniform(.3,.9,p.shape)
        r=fe.evaluate(rho,penal=2.7); k=fe.stiffness(rho,2.7)
        np.testing.assert_allclose(k.toarray(),k.T.toarray(),atol=1e-14)
        np.testing.assert_allclose(r.displacement[fe.fixed],0)
        np.testing.assert_allclose((k@r.displacement)[fe.free],fe.force[fe.free],atol=1e-11)
        self.assertLess(r.residual,1e-10)
        self.assertGreater(r.compliance,0)
        self.assertAlmostEqual(r.compliance,r.displacement@k@r.displacement,places=9)
        self.assertTrue(np.all(r.gradient<=0))

    def test_adjoint_finite_differences(self):
        # Random asymmetric BCs and heterogeneous densities expose axis/order bugs.
        p=random_case(Grid(3,2,2),seed=137)
        fe=Top3dFE(p.grid,p.bc,E0=3.2,Emin=.007)
        rho=np.random.default_rng(1).uniform(.25,.9,p.shape)
        penal=2.4; analytic=fe.evaluate(rho,penal).gradient
        numeric=np.zeros(p.shape); eps=1e-5
        for index in np.ndindex(p.shape):
            plus=rho.copy(); minus=rho.copy()
            plus[index]+=eps; minus[index]-=eps
            numeric[index]=(fe.evaluate(plus,penal).compliance-fe.evaluate(minus,penal).compliance)/(2*eps)
        np.testing.assert_allclose(analytic,numeric,rtol=3e-6,atol=2e-7)

    def test_torch_autograd_and_upstream(self):
        import torch
        p=paper_case("mbb",dims=(2,2,2))
        fe=Top3dFE(p.grid,p.bc)
        x=torch.tensor(np.random.default_rng(0).uniform(.3,.8,p.shape),dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a:fe.torch_compliance(a,2.5),(x,),eps=1e-5,atol=1e-5,rtol=1e-4))
        loss=3.7*fe.torch_compliance(x,2.5); loss.backward()
        np.testing.assert_allclose(x.grad.numpy(),3.7*fe.last_result.gradient)
        self.assertEqual(loss.dtype,x.dtype)

    def test_cg_matches_direct_and_reports_failure(self):
        p=paper_case("mbb",dims=(6,3,2))
        rho=np.random.default_rng(2).uniform(.4,.9,p.shape)
        direct=Top3dFE(p.grid,p.bc).evaluate(rho)
        cg=Top3dFE(p.grid,p.bc,solver="cg",rtol=1e-10).evaluate(rho)
        self.assertGreater(cg.iterations,1)
        self.assertLess(cg.residual,1e-9)
        self.assertAlmostEqual(cg.compliance/direct.compliance,1,places=9)
        np.testing.assert_allclose(cg.gradient,direct.gradient,rtol=1e-8,atol=1e-8)
        with self.assertRaisesRegex(SolverError,"failed to converge"):
            Top3dFE(p.grid,p.bc,solver="cg",maxiter=1).evaluate(rho)

    def test_invalid_inputs_and_zero_density(self):
        p=paper_case("mbb",dims=(2,2,1))
        fe=Top3dFE(p.grid,p.bc)
        for invalid in [np.full(p.shape,np.nan),np.full(p.shape,-.1),np.full(p.shape,1.1)]:
            with self.assertRaises(ValueError):
                fe.evaluate(invalid)
        with self.assertRaises(ValueError):
            fe.evaluate(np.ones((2,2,2)))
        r=fe.evaluate(np.zeros(p.shape),penal=3)
        self.assertGreater(r.compliance,0)
        np.testing.assert_array_equal(r.gradient,np.zeros(p.shape))

    @unittest.skipUnless(os.environ.get("BOXYTO_TEST_GPU")=="1","set BOXYTO_TEST_GPU=1 for GPU parity")
    def test_gpu_matches_cpu(self):
        p=random_case(Grid(6,4,3),seed=93)
        rho=np.random.default_rng(3).uniform(.3,.9,p.shape)
        cpu=Top3dFE(p.grid,p.bc,E0=2,Emin=1e-4).evaluate(rho,2.7)
        gpu=Top3dFE(p.grid,p.bc,backend="cupy",E0=2,Emin=1e-4,rtol=1e-10).evaluate(rho,2.7)
        self.assertLess(gpu.residual,1e-9)
        self.assertAlmostEqual(gpu.compliance/cpu.compliance,1,places=8)
        np.testing.assert_allclose(gpu.gradient,cpu.gradient,rtol=1e-7,atol=1e-8)

if __name__=="__main__":
    unittest.main()
