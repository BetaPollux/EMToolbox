#!/usr/bin/python3

from electrostatics import *
import numpy as np
import unittest


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_3dvector(self):
        # Schaum Electromagnetics 2e  P2.7
        q = 64.4e-9
        rf = np.array([0.0, 0.0, 0.0])
        rq = np.array([-4.0, 3.0, 2.0])
        E = efield_point(q, rq, rf)

        expected = 20.0 / np.sqrt(29) * np.array([4.0, -3.0, -2.0])

        np.testing.assert_allclose(E, expected, rtol=0.01)

    def test_1dvector(self):
        q = 5e-9
        rf = np.array([0.5])
        rq = np.array([0.1])
        E = efield_point(q, rq, rf)

        expected = np.array([280.9])

        np.testing.assert_allclose(E, expected, rtol=0.001)

    def test_null_coll(self):
        q = np.array([10e-9, 10e-9])
        rf = np.array([0.0, 0.0, 0.0])
        rq = np.array([[0.5, 0.5, 0.5],
                       [-0.5, -0.5, -0.5]])
        E = efield_point_coll(q, rq, rf)

        expected = np.zeros_like(rf)

        np.testing.assert_allclose(E, expected, rtol=0.001)

    def test_1dvector_coll(self):
        rf = np.array([0.5])
        rq = np.array([0.1, 0.3])
        q = np.array([1e-9, 1e-9])
        E = efield_point_coll(q, rq, rf)

        expected = np.array([280.9])

        np.testing.assert_allclose(E, expected, rtol=0.001)

    def test_3dvector_coll(self):
        # Schaum Electromagnetics 2e P2.8
        rf = np.array([0.0, 0.0, 5.0])
        rq = np.array([[0.0, 4.0, 0.0],
                       [3.0, 0.0, 0.0]])
        q = np.array([0.35e-6, -0.55e-6])
        E = efield_point_coll(q, rq, rf)

        expected = np.array([74.9, -48.0, -64.9])

        np.testing.assert_allclose(E, expected, rtol=0.01)

    def test_line(self):
        # Schaum Electromagnetics 2e P2.10
        ql = 20e-9
        rq = np.array([2.0, -4.0, 0.0])
        nq = np.array([0.0, 0.0, 1.0])
        rf = np.array([-2.0, -1.0, 4.0])
        E = efield_line(ql, rq, nq, rf)

        expected = np.array([-57.6, 43.2, 0.0])

        np.testing.assert_allclose(E, expected, rtol=0.01)

    def test_line_two(self):
        # Schaum Electromagnetics 2e P3.9
        ql = 20e-6
        rq = np.zeros(3)
        nq1 = np.array([1.0, 0.0, 0.0])
        nq2 = np.array([0.0, 1.0, 0.0])
        rf = np.array([3.0, 3.0, 3.0])
        E1 = efield_line(ql, rq, nq1, rf)
        E2 = efield_line(ql, rq, nq2, rf)
        E = E1 + E2
        expected = (0.5305e-6 / eps0) * np.array([1.0, 1.0, 2.0])

        np.testing.assert_allclose(E, expected, rtol=0.01)

    def test_plane(self):
        # Schaum Electromagnetics 2e P2.24
        qs = 0.3e-9
        rq = np.array([3.0, 0.0, 0.0])
        nq = np.array([2.0, -3.0, 1.0]) / np.sqrt(14)
        rf1 = np.zeros(3)                # Below plane
        rf2 = np.array([3.0, 0.0, 4.0])  # Above plane
        rf3 = np.array([0.0, 0.0, 6.0])  # On plane
        E1 = efield_plane(qs, rq, nq, rf1)
        E2 = efield_plane(qs, rq, nq, rf2)
        E3 = efield_plane(qs, rq, nq, rf3)

        expected = 17.0 / np.sqrt(14) * np.array([-2.0, 3.0, -1.0])

        np.testing.assert_allclose(E1, expected, rtol=0.01)
        np.testing.assert_allclose(E2, -expected, rtol=0.01)
        np.testing.assert_allclose(E3, np.zeros(3), rtol=0.01)

    def test_ring(self):
        # Schaum Electromagnetics 2e P2.47
        ql = 10e-9
        rq = np.zeros(3)
        nq = np.array([0.0, 0.0, 1.0])
        a = 2
        rf = np.array([0.0, 0.0, 5.0])
        E = efield_ring(ql, rq, nq, a, rf)

        expected = efield_point(100.5e-9, rq, rf)

        np.testing.assert_allclose(E, expected, rtol=0.01)

    def test_disk(self):
        # Schaum Electromagnetics 2e P2.4
        a = 5
        qs = 500e-6 / a ** 2
        rq = np.zeros(3)
        nq = np.array([0.0, 0.0, 1.0])
        rf = np.array([0.0, 0.0, 5.0])
        E = efield_disk(qs, rq, nq, a, rf)

        expected = np.array([0.0, 0.0, (16.56 / 50e-6)])

        np.testing.assert_allclose(E, expected, rtol=0.01)


if __name__ == '__main__':
    unittest.main()
