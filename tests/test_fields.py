"""Numerical checks of the notebook formulation and trainable derivatives."""
import unittest
import os
import numpy as np
import torch

from boxyto.fields import TrimmedNeuralField, adjugate, shadows


class FieldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def make_model(self):
        model = TrimmedNeuralField(7, global_frequency=0.8, bands=2,
                                   field_scales=((1.,1.,1.),(.7,1.2,.5),(1.1,.8,1.3)),
                                   dtype=torch.float64)
        generator = torch.Generator().manual_seed(93)
        with torch.no_grad():
            model.weights.copy_(torch.randn(model.weights.shape, generator=generator, dtype=torch.float64) * 0.35)
        return model

    def test_original_initialization_and_density(self):
        model = TrimmedNeuralField(40, dtype=torch.float64)
        axis = np.linspace([-.6,0.], [0.,.6], 4).flatten()
        mesh = np.meshgrid(axis, axis, axis, indexing="ij")
        expected = np.stack([v.reshape(-1) for v in mesh])
        np.testing.assert_array_equal(model.kernels[0].detach().numpy(), expected)
        np.testing.assert_allclose(model.kernels[1].detach().numpy(), expected * .95, rtol=0, atol=0)
        self.assertEqual(model.weights.shape, (3,512,1))
        coords = torch.tensor([[0.,0.,0.],[-12.,22.,5.]], dtype=torch.float64)
        torch.testing.assert_close(model(coords), torch.full((2,1), .125, dtype=torch.float64))
        self.assertTrue(torch.equal(model.gradient(coords), torch.zeros((2,3), dtype=torch.float64)))
        self.assertTrue(torch.equal(model.hessian(coords), torch.zeros((2,3,3), dtype=torch.float64)))
        self.assertTrue(torch.isfinite(model.gaussian_curvature(coords)).all())

    def test_forward_matches_notebook_expression_with_feature_scales(self):
        model = self.make_model()
        coords = torch.tensor([[.2,-1.3,2.4],[-3.,1.1,.7]], dtype=torch.float64)
        reference = []
        for i in range(3):
            basis = torch.cos((coords * model.field_scales[i]) / (2.*model.nelm) @ (50.*model.kernels[i]) + 1.)
            reference.append(torch.sigmoid(basis @ model.weights[i]))
        expected = reference[0] * reference[1] * reference[2]
        torch.testing.assert_close(model.density(coords), expected, atol=2e-14, rtol=2e-13)

    def test_analytic_spatial_derivatives_match_autograd_and_finite_difference(self):
        model = self.make_model()
        coords = torch.tensor([[.2,-1.3,2.4],[-3.,1.1,.7]], dtype=torch.float64, requires_grad=True)
        value = model.density(coords)
        gradient = torch.autograd.grad(value.sum(), coords, create_graph=True)[0]
        hessian = torch.stack([torch.autograd.grad(gradient[:,i].sum(), coords, retain_graph=True)[0] for i in range(3)], dim=1)
        analytic_value, analytic_gradient, analytic_hessian = model.derivatives(coords)
        torch.testing.assert_close(analytic_value, value)
        torch.testing.assert_close(analytic_gradient, gradient, atol=2e-13, rtol=2e-12)
        torch.testing.assert_close(analytic_hessian, hessian, atol=2e-12, rtol=2e-11)
        epsilon = 1e-5
        for i in range(3):
            shift = torch.zeros_like(coords)
            shift[:,i] = epsilon
            finite = (model.density(coords + shift) - model.density(coords - shift)) / (2.*epsilon)
            torch.testing.assert_close(analytic_gradient[:,i:i+1], finite, atol=2e-10, rtol=2e-8)

    def test_adjugate_handles_singular_hessians(self):
        matrix = torch.tensor([[[2.,1.,0.],[1.,3.,1.],[0.,1.,2.]],
                               [[1.,2.,3.],[2.,4.,6.],[0.,0.,0.]]], dtype=torch.float64)
        expected = torch.eye(3, dtype=torch.float64).expand(2,3,3) * torch.linalg.det(matrix)[:,None,None]
        torch.testing.assert_close(matrix @ adjugate(matrix), expected, atol=1e-13, rtol=1e-13)
        self.assertTrue(torch.isfinite(adjugate(matrix)).all())

    def test_geometry_losses_reuse_matches_individual_terms(self):
        model = self.make_model()
        coords = torch.tensor([[.2,-1.3,2.4],[-3.,1.1,.7],[2.1,.4,-1.5]], dtype=torch.float64)
        losses = model.geometry_losses(coords, 75., 83.)
        mask = 1. - model.seams(coords)
        torch.testing.assert_close(losses["curvature"], (model.gaussian_curvature(coords).abs()*mask).mean())
        torch.testing.assert_close(losses["intersection"], model.intersection(coords,75.).mean())
        torch.testing.assert_close(losses["alignment"], (model.axis_alignment(coords,83.)*mask).mean())
        inactive = model.geometry_losses(coords, None, None)
        self.assertEqual(float(inactive["intersection"].detach()), 0.)
        self.assertEqual(float(inactive["alignment"].detach()), 0.)
        # The fixed machining axis is unoriented, including when the user flips it.
        torch.testing.assert_close(model.axis_alignment(coords,83.,(0,0,1)),
                                   model.axis_alignment(coords,83.,(0,0,-1)))

    def test_geometry_gradients_match_parameter_finite_difference(self):
        model = self.make_model()
        coords = torch.tensor([[.2,-1.3,2.4],[-3.,1.1,.7],[2.1,.4,-1.5]], dtype=torch.float64)
        def objective():
            loss = model.geometry_losses(coords,75.,83.)
            return loss["curvature"] + loss["intersection"] + loss["alignment"]
        objective().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        index = (0,4,0)
        analytic = model.weights.grad[index].item()
        saved = model.weights[index].item()
        epsilon = 1e-5
        with torch.no_grad():
            model.weights[index] = saved + epsilon
            plus = objective().item()
            model.weights[index] = saved - epsilon
            minus = objective().item()
            model.weights[index] = saved
        self.assertAlmostEqual(analytic, (plus-minus)/(2.*epsilon), delta=max(1e-6, abs(analytic)*1e-5))

    def test_directional_shadow_and_flag_equivalence(self):
        density = torch.tensor([[[1.,0.,0.]]], dtype=torch.float64, requires_grad=True)
        forward = shadows(density, ("z+",))
        reverse = shadows(density, ("z-",))
        self.assertGreater(float(forward[0,0,2].detach()), .99)
        self.assertLess(float(reverse[0,0,2].detach()), .01)
        torch.testing.assert_close(reverse, shadows(density,(0,0,0,0,0,1)))
        both = shadows(density,("z+","z-"))
        self.assertTrue((both <= forward).all())
        self.assertTrue((both <= reverse).all())
        both.sum().backward()
        self.assertTrue(torch.isfinite(density.grad).all())
        for directions in [(), ("diagonal",), ("z+","z+"), (0,0,0,0,0,2)]:
            with self.assertRaises(ValueError):
                shadows(density,directions)

    @unittest.skipUnless(os.environ.get("BOXYTO_TEST_GPU") == "1", "set BOXYTO_TEST_GPU=1 for CUDA validation")
    def test_cuda_field_geometry_forward_and_backward(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA validation requested but no GPU is available")
        for dtype in (torch.float32, torch.float64):
            reference = self.make_model().to(dtype=dtype)
            model = self.make_model().to(device="cuda", dtype=dtype)
            cpu_coords = torch.tensor([[.2,-1.3,2.4],[-3.,1.1,.7],[2.1,.4,-1.5]], dtype=dtype)
            coords = cpu_coords.to("cuda")
            expected = reference.derivatives(cpu_coords)
            actual = model.derivatives(coords)
            tolerance = {"atol":2e-5,"rtol":2e-4} if dtype == torch.float32 else {"atol":2e-12,"rtol":2e-10}
            torch.testing.assert_close(model.density(coords).cpu(), reference.density(cpu_coords), **tolerance)
            for cpu_value, gpu_value in zip(expected, actual):
                torch.testing.assert_close(gpu_value.cpu(), cpu_value, **tolerance)
            cpu_losses = reference.geometry_losses(cpu_coords,75.,83.)
            gpu_losses = model.geometry_losses(coords,75.,83.)
            for name in cpu_losses:
                torch.testing.assert_close(gpu_losses[name].cpu(), cpu_losses[name], **tolerance)
            sum(cpu_losses.values()).backward()
            sum(gpu_losses.values()).backward()
            for (_, cpu_param), (_, gpu_param) in zip(reference.named_parameters(), model.named_parameters()):
                self.assertIsNotNone(gpu_param.grad)
                self.assertTrue(torch.isfinite(gpu_param.grad).all())
                torch.testing.assert_close(gpu_param.grad.cpu(), cpu_param.grad, **tolerance)
            rho = torch.tensor([[[1.,0.,0.]]], dtype=dtype, device="cuda", requires_grad=True)
            shadow = shadows(rho,("z+","z-"))
            shadow.sum().backward()
            self.assertTrue(torch.isfinite(rho.grad).all())
        torch.cuda.synchronize()

    def test_invalid_field_configuration(self):
        for kwargs in [{"nelm":0}, {"nelm":10,"global_frequency":float("nan")},
                       {"nelm":10,"bands":1}, {"nelm":10,"field_scales":[[1.,0.,1.]]*3}]:
            with self.assertRaises(ValueError):
                TrimmedNeuralField(**kwargs)


if __name__ == "__main__":
    unittest.main()

