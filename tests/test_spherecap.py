#!/usr/bin/python3

import numpy as np
import pytest
from pytest import approx
from emtoolbox.fields.spherecap import SphereCapacitor
from emtoolbox.utils.constants import EPS0


def test_capacitance():
    sc = SphereCapacitor(30e-3, 2.6, 20e-3)
    assert sc.capacitance() == approx(21.7e-12, abs=0.1e-12)


def test_get_arrays():
    N = 11
    sc = SphereCapacitor(30e-3, 2.6, 20e-3)
    X, er = sc.get_arrays(N=N)
    assert X == approx(np.linspace(30e-3, 50e-3, N))
    assert er == approx(2.6 * np.ones(N-1))


def test_efield():
    ri = 30e-3
    ro = 50e-3
    N = 5
    sc = SphereCapacitor(ri, 2.6, ro - ri)
    X = np.linspace(ri, ro, N)
    Va = 10.0
    Q = sc.capacitance() * Va
    expected = Q / (4 * np.pi * EPS0 * 2.6 * X**2)
    efield = sc.efield(X, Va=Va)
    assert efield == approx(expected)


def test_potential():
    ri = 30e-3
    ro = 50e-3
    N = 5
    sc = SphereCapacitor(ri, 2.6, ro - ri)
    X = np.linspace(ri, ro, N)
    Va = 10.0
    Q = sc.capacitance() * Va
    expected = Va - Q / (4 * np.pi * EPS0 * 2.6) * (1/ri - 1/X)
    potential = sc.potential(X, Va=Va)
    assert potential == approx(expected)


def test_charge():
    sc = SphereCapacitor(0.5e-3, 5.2, 3.5e-3)
    Va = 10.0
    expected = sc.capacitance() * Va
    assert sc.charge(Va) == approx(expected)


def test_energy():
    sc = SphereCapacitor(0.5e-3, 5.2, 3.5e-3)
    Va = 10.0
    expected = 0.5 * sc.capacitance() * Va ** 2
    assert sc.energy(Va) == approx(expected)
