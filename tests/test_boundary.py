"""Boundary conventions, exact paper BCs, feasible masks, and random stability."""
import unittest
import numpy as np
from boxyto.boundary import (Grid, BoundaryConditions, ProblemSpec, paper_case,
                             random_case, custom_case, sphere_mask, box_mask)


class BoundaryTests(unittest.TestCase):
    def test_node_ids_and_cell_coordinates(self):
        grid = Grid(3, 2, 2)
        np.testing.assert_array_equal(grid.node_ids(grid.nodes_xyz()), np.arange(grid.nnodes))
        np.testing.assert_allclose(grid.cell_centers_xyz()[0, 0, 0], [.5, 1.5, .5])
        np.testing.assert_allclose(grid.cell_centers_xyz()[-1, -1, -1], [2.5, .5, 1.5])
        with self.assertRaises(ValueError):
            grid.node_ids([[.5, 0, 0]])
        with self.assertRaises(ValueError):
            grid.node_ids([[4, 0, 0]])

    def test_original_paper_boundary_conditions(self):
        expected = {"mbb": ((40,20,8),.3,9,189,[0,1,0]),
                    "mount": ((40,20,8),.2,369,252,[0,1,0]),
                    "link": ((40,20,8),.25,288,144,[2,1,0]),
                    "pedal": ((50,20,8),.2,99,252,[0,1,0])}
        for name, (dims,vf,nload,nfixed,resultant) in expected.items():
            p = paper_case(name)
            self.assertEqual(p.dims, dims)
            self.assertEqual(p.volume_fraction, vf)
            self.assertEqual(len(p.bc.loads),nload)
            self.assertEqual(len(p.bc.supports),nfixed)
            np.testing.assert_allclose(p.bc.forces.sum(0),resultant)
            self.assertFalse(p.void_mask.any())
            self.assertFalse(p.solid_mask.any())
        p=paper_case("mbb")
        self.assertTrue(np.all(p.bc.loads[:,0]==40))
        self.assertTrue(np.all(p.bc.loads[:,1]==0))
        self.assertTrue(np.all(p.bc.supports[:,0]==0))
        p=paper_case("pedal")
        self.assertEqual(set(p.bc.loads[:,0]),set(range(40,51)))
        self.assertTrue(np.all(p.bc.loads[:,1]==19))

    def test_duplicate_loads_accumulate(self):
        grid=Grid(2,2,1)
        p=paper_case("mbb",dims=grid.dims)
        bc=BoundaryConditions(np.repeat(p.bc.loads,2,axis=0),np.repeat(p.bc.forces,2,axis=0),
                              p.bc.supports,p.bc.constrained)
        f,_=bc.dof_data(grid); f0,_=p.bc.dof_data(grid)
        np.testing.assert_allclose(f,2*f0)
        roundtrip=BoundaryConditions.from_dict(bc.to_dict())
        np.testing.assert_array_equal(roundtrip.constrained,bc.constrained)

    def test_invalid_rigid_support_and_fixed_load(self):
        grid=Grid(2,2,2)
        bc=BoundaryConditions([[2,2,2]],[[0,1,0]],[[0,0,0]],[True,True,True])
        with self.assertRaisesRegex(ValueError,"rigid-body"):
            bc.dof_data(grid)
        p=paper_case("mbb",dims=grid.dims)
        bc=BoundaryConditions([[0,0,0]],[[1,0,0]],p.bc.supports,p.bc.constrained)
        with self.assertRaisesRegex(ValueError,"constrained DOF"):
            bc.dof_data(grid)
        bc=BoundaryConditions([[2,2,2]],[[0,1,0]],[[0,0,0],[0,2,0],[0,0,2]],
                              [[1,1,1],[1,0,1],[1,0,0]])
        bc.dof_data(grid)  # A minimally constrained, stable six-DOF fixture.

    def test_random_deterministic_and_stable(self):
        grid=Grid(8,6,4)
        for style in ("points","patch","face"):
            for seed in range(20):
                p=random_case(grid,seed=seed,support_style=style)
                q=random_case(grid,seed=seed,support_style=style)
                self.assertEqual(p.bc.to_dict(),q.bc.to_dict())
                p.bc.dof_data(grid)
                self.assertLessEqual(len(p.bc.loads),4)
                self.assertEqual(p.metadata["generator"],"numpy.PCG64")
        self.assertNotEqual(random_case(grid,1).bc.to_dict(),random_case(grid,2).bc.to_dict())

    def test_custom_schemas_and_mask_orientation(self):
        data={"dims":[4,4,2],"volume_fraction":.3,
              "load_regions":[{"box":[[4,4],[0,0],[0,2]],"force":[0,1,0]}],
              "support_regions":[{"box":[[0,0],[0,4],[0,2]]}],
              "no_design":[{"type":"box","bounds":[[2,4],[3,4],[0,2]]}]}
        p=custom_case(data)
        self.assertTrue(p.void_mask[0,2:,:].all())
        self.assertFalse(p.void_mask[1:,:,:].any())
        d={"dims":list(p.dims),"volume_fraction":.3,
           "loads":[{"points":p.bc.loads.tolist(),"vectors":p.bc.forces.tolist()}],
           "supports":[{"points":p.bc.supports.tolist(),"components":p.bc.constrained.tolist()}],
           "regions":[{"type":"sphere","center":[2,2,1],"radius":.9,"kind":"void"}]}
        q=custom_case(d)
        self.assertEqual(q.bc.to_dict(),p.bc.to_dict())
        np.testing.assert_array_equal(q.void_mask,sphere_mask(q.grid,[2,2,1],.9))
        d["boundary"]=p.bc.to_dict()
        self.assertEqual(custom_case(d).bc.to_dict(),p.bc.to_dict())

    def test_feasibility_and_mask_gradient(self):
        import torch
        p=paper_case("mbb",dims=(4,4,2),no_design=True)
        x=torch.full(p.shape,.5,requires_grad=True)
        out=p.apply_masks_torch(x)
        self.assertTrue(torch.all(out[torch.as_tensor(p.void_mask)]==0))
        out.sum().backward()
        self.assertTrue(torch.all(x.grad[torch.as_tensor(p.void_mask)]==0))
        with self.assertRaisesRegex(ValueError,"overlap"):
            ProblemSpec(p.grid,p.bc,.3,np.ones(p.shape,bool),np.ones(p.shape,bool))
        with self.assertRaisesRegex(ValueError,"exceeds"):
            ProblemSpec(p.grid,p.bc,.3,solid_mask=np.ones(p.shape,bool))
        with self.assertRaisesRegex(ValueError,"infeasible"):
            ProblemSpec(p.grid,p.bc,.3,void_mask=np.ones(p.shape,bool))

if __name__=="__main__":
    unittest.main()
