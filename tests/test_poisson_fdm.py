#!/usr/bin/python3

import numpy as np
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


def test_plates_analytical():
    w = 4e-3
    X = np.linspace(0, w, 101)
    m = int(0.5 * len(X))
    V = fdm.plates_analytical(X, v_left=50, v_right=150)
    assert V[m] == approx(100.0)


def test_plates_dielectric_analytical():
    w = 4e-3
    X = np.linspace(0, w, 101)
    b = int(0.25 * len(X))
    er1 = 5.0
    er2 = 1.0
    V = fdm.plates_dielectric_analytical(X, er1, er2, X[b], v_left=0, v_right=200)
    assert V[0] == approx(0)
    assert V[b] == approx(12.5)
    assert V[-1] == approx(200)


def test_poisson_1d():
    w = 5.0
    X = np.linspace(0, w, 101)
    bc = {'v_left': -2, 'v_right': 5}
    V = fdm.poisson_1d(X, **bc, conv=1e-3)
    Va = fdm.plates_analytical(X, **bc)
    assert V == approx(Va, abs=0.01)


def test_poisson_1d_dielectric():
    w = 4e-3
    X = np.linspace(0, w, 101)
    bc = {'v_left': 0, 'v_right': 200}
    b = int(0.25 * len(X))
    er1 = 5.0
    er2 = 1.0
    er = np.where(X[:-1] < X[b], er1, er2)

    V = fdm.poisson_1d(X, dielectric=er, **bc, conv=1e-4)
    Va = fdm.plates_dielectric_analytical(X, er1, er2, X[b], **bc)
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
