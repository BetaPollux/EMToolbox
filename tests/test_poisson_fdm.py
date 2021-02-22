#!/usr/bin/python3

import numpy as np
from emtoolbox.fields.pplate import ParallelPlates
import emtoolbox.fields.poisson_fdm as fdm
import pytest
from pytest import approx


def test_trough_analytical():
    w = 3.0
    h = 4.0
    x = np.linspace(0, w, 4)
    y = np.linspace(0, h, 5)
    X, Y = np.meshgrid(x, y)
    V = fdm.trough_analytical(X, Y, v_left=10, v_bottom=0, v_top=20, v_right=30)
    assert V[3, 1] == approx(16.4784, abs=1e-4)
    assert V[3, 2] == approx(21.8499, abs=1e-4)
    assert V[2, 1] == approx(14.1575, abs=1e-4)
    assert V[2, 2] == approx(20.4924, abs=1e-4)
    assert V[1, 1] == approx(9.60942, abs=1e-4)
    assert V[1, 2] == approx(14.9810, abs=1e-4)


def test_poisson_1d():
    pp = ParallelPlates(1.0, 5e-3)
    X, _ = pp.get_arrays(101)
    v0 = -2
    v1 = 5
    V = fdm.poisson_1d(X, v_left=v0, v_right=v1, conv=1e-3)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_dielectric():
    pp = ParallelPlates((5.0, 1.0), (1e-3, 3e-3))
    v0 = 0
    v1 = 200
    X, er = pp.get_arrays(101)
    V = fdm.poisson_1d(X, dielectric=er, v_left=v0, v_right=v1, conv=1e-4)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_dielectric2():
    pp = ParallelPlates((1.0, 5.0, 1.0), (2e-3, 1e-3, 1e-3))
    v0 = 0
    v1 = 200
    X, er = pp.get_arrays(101)
    V = fdm.poisson_1d(X, dielectric=er, v_left=v0, v_right=v1, conv=1e-4)
    Va = pp.potential(X, v1 - v0, v0)
    assert V == approx(Va, abs=0.01)


def test_poisson_2d():
    w = 2.0
    h = 1.0
    x = np.linspace(0, w, 101)
    y = np.linspace(0, h, 51)
    X, Y = np.meshgrid(x, y)
    bc = {'v_top': 10, 'v_left': 5, 'v_right': -2, 'v_bottom': -4}
    V = fdm.poisson_2d(X, Y, **bc, conv=1e-3)
    Va = fdm.trough_analytical(X, Y, **bc)
    # Exclude boundaries due to analytical error at corners
    assert V[1:-1, 1:-1] == approx(Va[1:-1, 1:-1], abs=0.1)  
