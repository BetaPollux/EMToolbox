#!/usr/bin/python3

import numpy as np
from pytest import approx
import emtoolbox.fields.efield as em


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
