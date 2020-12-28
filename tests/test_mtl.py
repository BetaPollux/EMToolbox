#! /usr/bin/python3

import numpy as np
import unittest
from emtoolbox.tline.mtl import Mtl, meters_from_mils, WideSep_Wire, WideSep_Plane, WideSep_Shield


class TestWideSep_Wire_Three(unittest.TestCase):
    def setUp(self):
        pos = np.array([[50, 0],
                        [0, 0],
                        [100, 0]])

        wire_radius = np.array([7.5,
                                7.5,
                                7.5])

        self.p = meters_from_mils(pos)
        self.r = meters_from_mils(wire_radius)

    def tearDown(self):
        pass

    def test_self_inductance(self):
        l1 = WideSep_Wire.self_inductance(self.p[0], self.r[0],
                                          self.p[1], self.r[1])
        l2 = WideSep_Wire.self_inductance(self.p[0], self.r[0],
                                          self.p[2], self.r[2])

        np.testing.assert_approx_equal(l1, 0.759e-6, significant=3)
        np.testing.assert_approx_equal(l2, 0.759e-6, significant=3)

    def test_mutual_inductance(self):
        l21 = WideSep_Wire.mutual_inductance(self.p[0], self.r[0],
                                             self.p[1], self.p[2])

        np.testing.assert_approx_equal(l21, 0.241e-6, significant=3)

    def test_assemble_inductance_matrix(self):
        L = Mtl.assemble_inductance_matrix([0.759e-6, 0.759e-6], [0.241e-6])

        expected = np.array([[0.759e-6, 0.241e-6],
                             [0.241e-6, 0.759e-6]])

        np.testing.assert_allclose(L, expected, rtol=0, atol=0.001e-6)

    def test_generate_capacitance_matrix(self):
        L = np.array([[0.759e-6, 0.241e-6],
                      [0.241e-6, 0.759e-6]])

        C = Mtl.generate_capacitance_matrix(L, 1.0)

        expected = np.array([[1.63e-11, -5.17e-12],
                             [-5.17e-12, 1.63e-11]])

        np.testing.assert_allclose(C, expected, atol=0.01e-12)


class TestWideSep_Plane_Three(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[-0.01, 0.02],
                           [0.01, 0.02]])

        wire_radius = np.array([16,
                                16])

        self.r = meters_from_mils(wire_radius)

    def tearDown(self):
        pass

    def test_self_inductance(self):
        l1 = WideSep_Plane.self_inductance(self.p[0], self.r[0])
        l2 = WideSep_Plane.self_inductance(self.p[1], self.r[1])

        np.testing.assert_approx_equal(l1, 0.918e-6, significant=3)
        np.testing.assert_approx_equal(l2, 0.918e-6, significant=3)

    def test_mutual_inductance(self):
        l21 = WideSep_Plane.mutual_inductance(self.p[0], self.p[1])

        np.testing.assert_approx_equal(l21, 0.161e-6, significant=3)


class TestWideSep_Plane_Twisted(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[-0.01, 0.02],
                           [0.01, 0.02 + 8.382e-4],
                           [0.01, 0.02 - 8.382e-4]])

        wire_radius = np.array([16,
                                16,
                                16])

        self.r = meters_from_mils(wire_radius)

    def tearDown(self):
        pass

    def test_self_inductance(self):
        l1 = WideSep_Plane.self_inductance(self.p[0], self.r[0])
        l2 = WideSep_Plane.self_inductance(self.p[1], self.r[1])
        l3 = WideSep_Plane.self_inductance(self.p[2], self.r[2])

        np.testing.assert_approx_equal(l1, 0.9179e-6, significant=3)
        np.testing.assert_approx_equal(l2, 0.9261e-6, significant=3)
        np.testing.assert_approx_equal(l3, 0.9093e-6, significant=3)

    def test_mutual_inductance(self):
        l21 = WideSep_Plane.mutual_inductance(self.p[0], self.p[1])
        l31 = WideSep_Plane.mutual_inductance(self.p[0], self.p[2])
        l23 = WideSep_Plane.mutual_inductance(self.p[1], self.p[2])

        np.testing.assert_approx_equal(l21, 0.1641e-6, significant=3)
        np.testing.assert_approx_equal(l31, 0.1574e-6, significant=3)
        np.testing.assert_approx_equal(l23, 0.6344e-6, significant=3)


class TestWideSep_Shield_Three(unittest.TestCase):
    def setUp(self):
        pos = np.array([15,
                        15])
        self.phi = np.array([0,
                             np.pi])
        wire_radius = np.array([7.5,
                                7.5])
        self.p = meters_from_mils(pos)
        self.r = meters_from_mils(wire_radius)
        self.r_sh = meters_from_mils(30)

    def tearDown(self):
        pass

    def test_self_inductance(self):
        l1 = WideSep_Shield.self_inductance(self.r_sh, self.p[0], self.r[0])
        l2 = WideSep_Shield.self_inductance(self.r_sh, self.p[1], self.r[1])

        np.testing.assert_approx_equal(l1, 220e-9, significant=3)
        np.testing.assert_approx_equal(l2, 220e-9, significant=3)

    def test_mutual_inductance(self):
        l21 = WideSep_Shield.mutual_inductance(self.r_sh,
                                               self.p[0], self.phi[0],
                                               self.p[1], self.phi[1])

        np.testing.assert_approx_equal(l21, 44.6e-9, significant=3)


if __name__ == '__main__':
    unittest.main()
