import unittest
import numpy as np
from boxyto.optimize import Options, field_coordinates, build_problem

class TestConfiguration(unittest.TestCase):
    def test_centers_and_coordinate_order(self):
        c=field_coordinates((4,2,3))
        self.assertEqual(c.shape,(24,3))
        np.testing.assert_array_equal(c[0],[1,-3,-2])
        np.testing.assert_array_equal(c[-1],[-1,3,2])
    def test_custom_region_feasibility(self):
        config = {"problem": {"type": "custom", "dims": [4,2,2], "volume_fraction": .3,
                             "load_regions": [{"box": [[4,4],[0,0],[0,2]], "force": [0,1,0]}],
                             "support_regions": [{"box": [[0,0],[0,2],[0,2]]}],
                             "regions": [{"type": "box", "bounds": [[0,4],[0,2],[0,2]], "kind": "solid"}]}}
        with self.assertRaises(ValueError):
            build_problem(config)

    def test_every_example(self):
        from pathlib import Path
        from boxyto.optimize import read_config
        for path in (Path(__file__).resolve().parents[1]/"examples").glob("*.json"):
            with self.subTest(example=path.name):
                spec = build_problem(read_config(path))
                self.assertTrue(np.any(~(spec.void_mask | spec.solid_mask)))

    def test_reject_mistyped_mode_and_scales(self):
        with self.assertRaises(ValueError):
            Options(mode="5-axis").validate()
        with self.assertRaises(ValueError):
            Options(field_scales=((1,1,0),)*3).validate()

if __name__ == "__main__":
    unittest.main()
