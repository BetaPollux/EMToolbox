#!/usr/bin/python3

import numpy as np
from pytest import approx
import emtoolbox.fields.efield as em
from emtoolbox.fields.pplate import ParallelPlates
import emtoolbox.fields.poisson_fdm as fdm


def test_efield_1d():
    X, dx = np.linspace(0, 1, 101, retstep=True)
    V = 10.0 * X - 1.0
    # E = -d/dx V
    Ea = -10.0
    E = em.efield_1d(V, dx)
    assert E == approx(Ea)


def test_efield_2d():
    x, dx = np.linspace(0, 1, 101, retstep=True)
    y, dy = np.linspace(0, 2, 101, retstep=True)
    X, Y = np.meshgrid(x, y)
    V = 10.0 * X + 5.0 * Y - 1.0
    # Ex = -d/dx V
    # Ey = -d/dy V
    Exa = -10.0
    Eya = -5.0
    Ex, Ey = em.efield_2d(V, dx, dy)
    assert Ex == approx(Exa)
    assert Ey == approx(Eya)


def test_efield_pplate():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    Va = 200
    pp = ParallelPlates((er1, er2), (t1, t2))
    X, dx = np.linspace(0, 4e-3, 21, retstep=True)
    V = pp.potential(X, Va)
    E = pp.efield(X, Va)
    Ec = em.efield_1d(V, dx)
    assert Ec[:5] == approx(E[:5])
    # discontinuity at boundary [5]
    assert Ec[6:] == approx(E[6:])


def test_efield_pplate_fdm():
    er1, t1 = 5.0, 1e-3
    er2, t2 = 1.0, 3e-3
    Va = 200
    pp = ParallelPlates((er1, er2), (t1, t2))
    X, dx = np.linspace(0, 4e-3, 21, retstep=True)
    V = pp.potential(X, Va)
    E = em.efield_1d(V, dx)

    er = np.where(X[:-1] < t1, er1, er2)
    Vfdm = fdm.poisson_1d(X, dielectric=er, v_right=Va)
    Efdm = em.efield_1d(Vfdm, dx)
    assert Efdm == approx(E, rel=0.001)

