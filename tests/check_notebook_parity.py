"""Optional CPU parity check against the unchanged TensorFlow notebook.

This is separate from portable unittest discovery. It needs both installed
TensorFlow and PyTorch and a supplied original objective_tuning notebook.
It extracts only known field methods, never executes setup or optimization
cells, and compares nonzero randomly weighted fields in float64.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import torch
from boxyto.fields import TrimmedNeuralField


METHODS = {
    "rbnn1", "rbnn2", "rbnn3", "rbnn", "drbnn", "hrbnn", "gcurv",
    "gbnn1", "gbnn2", "gbnn3", "gbnn", "drbnn1", "drbnn2", "drbnn3",
    "intersect_2field", "intersect", "axis_align2", "adjoint",
}


def original_field_class(path):
    notebook = json.loads(Path(path).read_text())
    methods = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "class Opt" not in source and "class Utils" not in source:
            continue
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name in ("Utils", "Opt"):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name in METHODS:
                        methods[method.name] = method
    missing = METHODS - set(methods)
    if missing:
        raise RuntimeError(f"Notebook does not contain required original field methods: {sorted(missing)}")
    cls = ast.ClassDef(name="OriginalField", bases=[], keywords=[], body=list(methods.values()), decorator_list=[])
    namespace = {"tf": tf, "np": np}
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["OriginalField"]


def check(path):
    torch.set_num_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    scales = ((1.,1.,1.),(.6,1.2,.75),(1.1,.9,1.3))
    model = TrimmedNeuralField(40, .9, bands=4, field_scales=scales, dtype=torch.float64)
    generator = torch.Generator().manual_seed(735)
    with torch.no_grad():
        model.weights.copy_(torch.randn(model.weights.shape, dtype=torch.float64, generator=generator)*.09)
    reference = original_field_class(path)()
    reference.nelm = 40
    reference.a_dir = tf.constant([[0.,0.,1.]], dtype=tf.float64)
    for i in range(3):
        setattr(reference, f"kernel{i+1}", tf.Variable(model.kernels[i].detach().numpy(), dtype=tf.float64))
        setattr(reference, f"weights{i+1}", tf.Variable(model.weights[i].detach().numpy(), dtype=tf.float64))
        setattr(reference, f"scalingX{i+1}", tf.constant([scales[i]], dtype=tf.float64))
    xyz = np.array([[1.2,-10.3,2.4],[-13.1,25.1,.7],[19.3,-4.3,-3.9],[-8.1,-24.2,6.1],[5.2,31.4,-6.3]], dtype=np.float64)
    coords = torch.tensor(xyz, dtype=torch.float64)
    tfcoords = tf.constant(xyz, dtype=tf.float64)
    comparisons = {
        "density": (model.density(coords), reference.rbnn(tfcoords)),
        "gradient": (model.gradient(coords), reference.drbnn(tfcoords)),
        "hessian": (model.hessian(coords), reference.hrbnn(tfcoords)),
        "gaussian_curvature": (model.gaussian_curvature(coords), reference.gcurv(tfcoords)),
        "seams": (model.seams(coords), reference.gbnn(tfcoords)),
        "intersection": (model.intersection(coords,75.), reference.intersect(tfcoords,tf.constant(75., dtype=tf.float64))),
        "axis_alignment": (model.axis_alignment(coords,83.), reference.axis_align2(reference.drbnn(tfcoords),tf.constant(83., dtype=tf.float64))),
    }
    errors = {}
    for name, (actual, expected) in comparisons.items():
        actual, expected = actual.detach().numpy(), expected.numpy()
        # Original notebook approximates radians by angle*.0174533.
        rtol = 5e-6 if name in ("intersection", "axis_alignment") else 2e-10
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=2e-12, err_msg=name)
        errors[name] = float(np.max(np.abs(actual-expected)))
    result = {"passed":True, "samples":len(xyz), "dtype":"float64", "features_per_field":512,
              "torch":torch.__version__, "tensorflow":tf.__version__, "max_absolute_errors":errors}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path,
                        help="Original boxy_objective_tuning.ipynb path; source remains unchanged")
    args = parser.parse_args()
    check(args.notebook)

