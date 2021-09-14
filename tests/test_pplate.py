#!/bin/usr/python3

import numpy as np
import pytest
from pytest import approx
from emtoolbox.fields.pplate import ParallelPlates


def test_capacitance():
    pp = ParallelPlates(3.5, 1e-3)
    assert pp.capacitance() == approx(30.989e-9)


def test_capacitance_area():
    pp = ParallelPlates(3.5, 1e-3, area=10.0)
    assert pp.capacitance() == approx(309.89e-9, rel=0.001)


def test_get_arrays():
    N = 11
    pp = ParallelPlates(3.5, 1e-3)
    X, er = pp.get_arrays(N=N)
    assert X == approx(np.linspace(0, 1e-3, N))
    assert er == approx(3.5 * np.ones(N-1))


def test_efield():
    t = 1e-3
    N = 5
    pp = ParallelPlates(3.5, t)
    X = np.linspace(0, t, N)
    Va = 10.0
    expected = -Va / t * np.ones(N)
    efield = pp.efield(X, Va)
    assert efield == approx(expected)


def test_potential():
    t = 1e-3
    N = 5
    pp = ParallelPlates(3.5, t)
    X = np.linspace(0, t, N)
    Va = 10.0
    expected = Va / t * X
    potential = pp.potential(X, Va)
    assert potential == approx(expected)


def test_potential_ref():
    t = 1e-3
    N = 5
    pp = ParallelPlates(3.5, t)
    X = np.linspace(0, t, N)
    Va = 10.0
    Vref = 5.0
    expected = Va / t * X + Vref
    potential = pp.potential(X, Va, Vref=Vref)
    assert potential == approx(expected)


def test_2layer():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    pp = ParallelPlates((er1, er2), (t1, t2))
    X, er = pp.get_arrays(N=5)
    assert pp.thickness == approx(t1 + t2)
    assert X == approx([0.0, 1e-3, 2e-3, 3e-3, 4e-3])
    assert er == approx([er1, er2, er2, er2])


def test_2layer_mixed_type():
    er1, t1 = 5.0, 1e-3
    er2 = 1.0
    with pytest.raises(Exception):
        ParallelPlates((er1, er2), t1)


def test_2layer_mixed_length():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    with pytest.raises(Exception):
        ParallelPlates((er1, er2), (t1, t2, t2))


def test_capacitance_2layer():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    pp = ParallelPlates((er1, er2), (t1, t2))
    assert pp.capacitance() == approx(2.767e-9)


def test_potential_2layer():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    pp = ParallelPlates((er1, er2), (t1, t2))
    X = np.linspace(0, 4e-3, 5)
    V = pp.potential(X, 200)
    assert V == approx([0, 12.5, 75, 137.5, 200])


def test_efield_2layer():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    pp = ParallelPlates((er1, er2), (t1, t2))
    X = np.linspace(0, 4e-3, 5)
    E = pp.efield(X, 200)
    E1 = -12.5 / t1
    E2 = -187.5 / t2
    assert E == approx([E1, E1, E2, E2, E2])


def test_capacitance_3layer():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    pp = ParallelPlates((er2, er1, er2), (0.5 * t2, t1, 0.5 * t2))
    assert pp.capacitance() == approx(2.767e-9)


def test_charge():
    pp = ParallelPlates(3.5, 1e-3)
    Va = 10.0
    expected = pp.capacitance() * Va
    assert pp.charge(Va) == approx(expected)


def test_energy():
    pp = ParallelPlates(3.5, 1e-3)
    Va = 10.0
    expected = 0.5 * pp.capacitance() * Va ** 2
    assert pp.energy(Va) == approx(expected)


def test_2layer_area():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    A = 3.0
    pp1 = ParallelPlates((er1, er2), (t1, t2))
    pp2 = ParallelPlates((er1, er2), (t1, t2), area=A)
    Va = 25.0
    X = np.linspace(0, t1 + t2, 11)
    assert pp1.capacitance() == approx(pp2.capacitance() / A)
    assert pp1.efield(X, Va) == approx(pp2.efield(X, Va))
    assert pp1.potential(X, Va) == approx(pp2.potential(X, Va))
