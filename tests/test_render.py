"""Coordinate and offscreen-image checks; run in a PyVista-enabled environment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from boxyto.render import density_to_surface, render_boundary_conditions, render_density


@unittest.skipUnless(importlib.util.find_spec("pyvista"), "PyVista is optional")
class TestRendering(unittest.TestCase):
    def test_full_box_bounds_and_closed_surface(self):
        mesh = density_to_surface(np.ones((3, 5, 2)), extent=(10, 9, 4))
        np.testing.assert_allclose(mesh.bounds, (0, 10, 0, 9, 0, 4))
        self.assertEqual(mesh.n_open_edges, 0)
        self.assertGreater(mesh.volume, 0)

    def test_asymmetric_voxel_has_correct_xyz_and_y_flip(self):
        rho = np.zeros((3, 5, 2))
        rho[0, 3, 1] = 1
        mesh = density_to_surface(rho, extent=(10, 9, 4))
        # x=3, y=row0 (physical top), z=1; unequal xyz spacing exposes swaps.
        np.testing.assert_allclose(mesh.bounds, (6, 8, 6, 9, 2, 4))
        np.testing.assert_allclose(mesh.center, (7, 7.5, 3))

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(density_to_surface(np.zeros((2,3,4))).n_points, 0)
        for rho in (np.ones((3,4)), np.full((2,3,4), np.nan), np.ones((2,3,4))*2):
            with self.assertRaises(ValueError):
                density_to_surface(rho)
        with self.assertRaises(ValueError):
            density_to_surface(np.ones((2,3,4)), cutoff=1)

    def test_curvature_saturation_uses_requested_limit(self):
        from PIL import Image
        rho = np.ones((4,7,3))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"saturated.png"
            render_density(rho, output, curvature=lambda pts: np.full(len(pts),.04),
                curvature_limit=.04, size=(256,256))
            with Image.open(output) as im:
                image = np.asarray(im).astype(np.int16)
                visible = image[...,3] > 200
                rgb = image[visible,:3]
                self.assertGreater(len(rgb),100)
                self.assertGreater(float(np.mean(rgb[:,0]-rgb[:,1])),100)

    def test_illustrative_collar_preserves_exact_export(self):
        import pyvista as pv
        rho = np.ones((12,20,6))
        exact = density_to_surface(rho)
        with tempfile.TemporaryDirectory() as folder:
            mesh_path = Path(folder)/"exact.vtp"
            report = render_density(rho, Path(folder)/"assembly.png",
                cylinders=[[6,6,8,6]], curvature=lambda pts: pts[:,0]*.002,
                mesh_output=mesh_path, size=(256,256))
            exported = pv.read(mesh_path)
            np.testing.assert_allclose(exported.points, exact.points)
            np.testing.assert_array_equal(exported.faces, exact.faces)
            np.testing.assert_allclose(exported["absolute_gaussian_curvature"], exported.points[:,0]*.002)
            self.assertEqual(report["illustrative_collar_count"],1)
            self.assertEqual(report["points"],exact.n_points)

    def test_offscreen_screenshots_and_scalar_mesh(self):
        from PIL import Image
        rho = np.zeros((9,15,5))
        rho[1:-1,1:-1,1:-1] = 1
        rho[3:6,4:8,:] = 0
        bc = {"fixed_points_xyz": [[0,0,0],[0,9,0],[0,0,5],[0,9,5]],
              "loads_points_xyz": [[15,0,2.5]], "load_vectors_xyz": [[0,1,0]]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"curvature.png"
            mesh_output = Path(folder)/"surface.vtp"
            report = render_density(rho, output, curvature=lambda pts: pts[:,0]*.003,
                bc=bc, size=(400,400), mesh_output=mesh_output)
            self.assertGreater(report["triangles"], 0)
            self.assertTrue(mesh_output.is_file())
            with Image.open(output) as im:
                self.assertEqual(im.size, (400,400))
                self.assertEqual(im.mode, "RGBA")
                rgb = np.asarray(im)[...,:3].astype(np.int16)
                self.assertGreater(float(rgb.std()), 10)
                self.assertGreater(int((rgb[...,0] > rgb[...,1]+25).sum()), 100)
            render_boundary_conditions((15,9,5), bc, Path(folder)/"bc.png",size=(256,256))
            with self.assertRaises(ValueError):
                render_density(rho, Path(folder)/"wrong.jpg", size=(128,128))
            with self.assertRaises(ValueError):
                render_density(rho, output, curvature=np.zeros(2), size=(128,128))


if __name__ == "__main__":
    unittest.main()
