#!/usr/bin/python3

import numpy as np
import unittest
import emtoolbox.fields.generate_points as gp


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_line(self):
        r0 = np.array([-1.0, -2.0, -3.0])
        r1 = np.array([1.0, 2.0, 3.0])
        s = 5
        points = gp.generate_line(r0, r1, s)

        expected = np.array([[-1.0, -2.0, -3.0],
                             [-0.5, -1.0, -1.5],
                             [0.0, 0.0, 0.0],
                             [0.5, 1.0, 1.5],
                             [1.0, 2.0, 3.0]])

        np.testing.assert_allclose(points, expected, rtol=0.01)

    def test_ring_z(self):
        a = 2
        r0 = np.zeros(3)
        n = np.array([0.0, 0.0, 1.0])
        s = 4
        points = gp.generate_ring(a, r0, n, s)

        expected = np.array([[a, 0.0, 0.0],
                             [0.0, a, 0.0],
                             [-a, 0.0, 0.0],
                             [0.0, -a, 0.0]])

        np.testing.assert_allclose(points, expected, atol=0.01)  # rtol fails?

    def test_ring_z_offset(self):
        a = 2
        r0 = np.array([0.0, 1.0, 1.0])
        n = np.array([0.0, 0.0, 1.0])
        s = 4
        points = gp.generate_ring(a, r0, n, s)

        expected = np.array([[a, 1.0, 1.0],
                             [0.0, 1.0 + a, 1.0],
                             [-a, 1.0, 1.0],
                             [0.0, 1.0 - a, 1.0]])

        np.testing.assert_allclose(points, expected, atol=0.01)  # rtol fails?

    def test_ring_x(self):
        a = 3
        r0 = np.zeros(3)
        n = np.array([1.0, 0.0, 0.0])
        s = 4
        points = gp.generate_ring(a, r0, n, s)

        expected = np.array([[0.0, 0.0, a],
                             [0.0, a, 0],
                             [0.0, 0.0, -a],
                             [0.0, -a, 0.0]])

        np.testing.assert_allclose(points, expected, atol=0.01)  # rtol fails?

    def test_ring_y(self):
        a = 5
        r0 = np.zeros(3)
        n = np.array([0.0, 1.0, 0.0])
        s = 4
        points = gp.generate_ring(a, r0, n, s)

        expected = np.array([[a, 0.0, 0.0],
                             [0.0, 0.0, a],
                             [-a, 0.0, 0.0],
                             [0.0, 0.0, -a]])

        np.testing.assert_allclose(points, expected, atol=0.01)  # rtol fails?


"""
    def test_ring_x_y(self):
        a = 1
        r0 = np.zeros(3)
        n = np.array([0.707, 0.707, 0.0])
        s = 4
        points = generate_ring(a, r0, n, s)

        expected = np.array([[a, 0.0, 0.0],
                             [0.0, 0.0, a],
                             [-a, 0.0, 0.0],
                             [0.0, 0.0, -a]])

        np.testing.assert_allclose(points, expected, atol=0.01)  # rtol fails?
"""

if __name__ == '__main__':
    unittest.main()
