#!/bin/usr/python3

import numpy as np
import pytest
from pytest import approx
from emtoolbox.fields.coaxcap import CoaxCapacitor


def test_capacitance():
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3)
    assert cc.capacitance() == approx(139.1e-12)


def test_capacitance_length():
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3, length=10.0)
    assert cc.capacitance() == approx(1.391e-9)


def test_get_arrays():
    N = 11
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3)
    X, er = cc.get_arrays(N=N)
    assert X == approx(np.linspace(0.5e-3, 4e-3, N))
    assert er == approx(5.2 * np.ones(N-1))


def test_efield():
    ri = 0.5e-3
    ro = 4.0e-3
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    X = np.linspace(ri, ro, N)
    Va = 10.0
    expected = -Va / X / np.log(ri / ro)
    efield = cc.efield(X, Va=Va)
    assert efield == approx(expected)


def test_potential():
    ri = 0.5e-3
    ro = 4.0e-3
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    X = np.linspace(ri, ro, N)
    Va = 10.0
    expected = Va / np.log(ri / ro) * np.log(X / ro)
    potential = cc.potential(X, Va=Va)
    assert potential == approx(expected)


def test_potential_ref():
    ri = 0.5e-3
    ro = 4.0e-3
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    X = np.linspace(ri, ro, N)
    Va = 10.0
    Vref = 5.0
    expected = Vref + Va / np.log(ri / ro) * np.log(X / ro)
    potential = cc.potential(X, Va=Va, Vref=Vref)
    assert potential == approx(expected)


def test_potential_inner():
    ri = 0.5e-3
    ro = 4.0e-3
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    X = np.linspace(0, ri, N)
    Va = 10.0
    potential = cc.potential(X, Va=Va)
    assert potential == approx(Va)


def test_potential_outer():
    ri = 0.5e-3
    ro = 4.0e-3
    Vref = 5.0
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    X = np.linspace(ro, 10 * ro, N)
    Va = 10.0
    potential = cc.potential(X, Va=Va, Vref=Vref)
    assert potential == approx(Vref)


def test_potential_diagonal():
    ri = 0.5e-3
    ro = 4.0e-3
    N = 5
    cc = CoaxCapacitor(ri, 5.2, ro - ri)
    R = np.linspace(ri, ro, N)
    theta = np.pi / 4
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Va = 10.0
    expected = Va / np.log(ri / ro) * np.log(R / ro)
    potential = cc.potential(X, Y, Va=Va)
    assert potential == approx(expected)


def test_charge():
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3)
    Va = 10.0
    expected = cc.capacitance() * Va
    assert cc.charge(Va) == approx(expected)


def test_energy():
    cc = CoaxCapacitor(0.5e-3, 5.2, 3.5e-3)
    Va = 10.0
    expected = 0.5 * cc.capacitance() * Va ** 2
    assert cc.energy(Va) == approx(expected)


def test_2layer():
    ri = 20e-3
    er1, t1 = 2.0, 2.5e-3
    er2, t2 = 5.0, 2.5e-3
    cc = CoaxCapacitor(ri, (er1, er2), (t1, t2))
    X, er = cc.get_arrays(N=5)
    assert cc.ro == approx(ri + t1 + t2)
    assert X == approx([20e-3, 21.25e-3, 22.5e-3, 23.75e-3, 25e-3])
    assert er == approx([er1, er1, er2, er2])


def test_2layer_mixed_type():
    er1, t1 = 5.0, 1e-3
    er2 = 1.0
    with pytest.raises(Exception):
        CoaxCapacitor(20e-3, (er1, er2), t1)


def test_2layer_mixed_length():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    with pytest.raises(Exception):
        CoaxCapacitor(20e-3, (er1, er2), (t1, t2, t2))


def test_capacitance_2layer():
    ri = 20e-3
    er1, t1 = 2.0, 2.5e-3
    er2, t2 = 5.0, 2.5e-3
    cc = CoaxCapacitor(ri, (er1, er2), (t1, t2))
    assert cc.capacitance() == approx(695.7e-12)
