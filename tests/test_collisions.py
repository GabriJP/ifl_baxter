import unittest

import numpy as np
import wandb

from utils.fed import Cube


class LoadHelpTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        wandb.init(mode="disabled")

    def test_init(self) -> None:
        with self.subTest("Small point"), self.assertRaises(ValueError):
            Cube(np.stack([np.array((0, 0, 0)), np.array((0, 1))]))

        with self.subTest("Overlapping point"), self.assertRaises(ValueError):
            Cube(np.stack([np.array((0, 0, 0)), np.array((0, 1, 0))]))

    def test_volume(self) -> None:
        tests = (
            ((0, 0, 0), (1, 1, 1), 1.0),
            ((0, 0, 0), (1, 2, 1), 2.0),
            ((-1, -1, -1), (1, 1, 1), 8.0),
        )
        for i, (p_a, p_b, vol) in enumerate(tests):
            with self.subTest(f"{i}"):
                cube = Cube.from_tuples(p_a, p_b)
                self.assertEqual(cube.volume, vol)

    def test_intersects(self) -> None:
        cube_a = Cube.from_tuples((0, 0, 0), (1, 1, 1))
        cube_b = Cube.from_tuples((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
        cube_c = Cube.from_tuples((2, 2, 2), (3, 3, 3))

        with self.subTest("Intersection of cubes"):
            self.assertTrue(cube_a.intersects(cube_b))

        with self.subTest("Non intersection of cubes"):
            self.assertFalse(cube_a.intersects(cube_c))

    def test_intersection_cube(self) -> None:
        cube_a = Cube.from_tuples((0, 0, 0), (1, 1, 1))
        cube_b = Cube.from_tuples((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
        cube_c = Cube.from_tuples((2, 2, 2), (3, 3, 3))

        with self.subTest("Intersection of cubes"):
            intersection = cube_a.intersection_cube(cube_b)
            self.assertTupleEqual(tuple(intersection.descriptor[0]), (0.5, 0.5, 0.5))
            self.assertEqual(tuple(intersection.descriptor[1]), (1.0, 1.0, 1.0))

        with self.subTest("Non intersection of cubes"), self.assertRaises(ValueError):
            cube_a.intersection_cube(cube_c)
