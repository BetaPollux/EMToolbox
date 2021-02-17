#!/usr/bin/python3

import numpy as np
from emtoolbox.fields.poisson_fdm import trough_analytical
import pytest
from pytest import approx


def test_trough_analytical():
    w = 3.0
    h = 4.0
    x = np.linspace(0, w, 4)
    y = np.linspace(0, h, 5)
    X, Y = np.meshgrid(x, y)
    V = trough_analytical(w, h, X, Y, v_left=10, v_bottom=0, v_top=20, v_right=30)
    assert V[3, 1] == approx(16.4784, abs=1e-4)
    assert V[3, 2] == approx(21.8499, abs=1e-4)
    assert V[2, 1] == approx(14.1575, abs=1e-4)
    assert V[2, 2] == approx(20.4924, abs=1e-4)
    assert V[1, 1] == approx(9.60942, abs=1e-4)
    assert V[1, 2] == approx(14.9810, abs=1e-4)
